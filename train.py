import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer
from dataset import get_train_test_datasets
from model import ViralPredictor
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Constants
CSV_PATH = 'data/raw'
IMG_DIR = 'data/raw/thumbnails/'
BATCH_SIZE = 16  # Increased batch size
EPOCHS = 100
LEARNING_RATE = 3e-4  # Higher LR for classifier only
TEST_SPLIT = 0.1
PATIENCE = 10  # More patience

# Training transforms with stronger Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize larger first
    transforms.RandomCrop(224),  # Random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)  # Random erasing for regularization
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load and split data (90% train, 10% test)
train_dataset, test_dataset = get_train_test_datasets(
    CSV_PATH, IMG_DIR, tokenizer, train_transform, test_size=TEST_SPLIT
)
# Use different transform for test set (no augmentation)
test_dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

model = ViralPredictor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Freeze pretrained models - only train classifier
print("Freezing ResNet50 and DistilBERT layers...")
for param in model.visual_model.parameters():
    param.requires_grad = False
for param in model.text_model.parameters():
    param.requires_grad = False

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)

            outputs = model(imgs, input_ids, mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


# Training loop with Early Stopping
print("\nStarting training (frozen backbone)...")
print("-" * 70)

best_test_loss = float('inf')
best_test_acc = 0
patience_counter = 0
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        imgs = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].unsqueeze(1).to(device)

        outputs = model(imgs, input_ids, mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Update scheduler
    scheduler.step(test_loss)

    # Early stopping check (based on accuracy now)
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_test_loss = test_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), "data/models/viral_predictor_best.pth")
        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Acc: {test_acc:.2%} | ★ Best")
    else:
        patience_counter += 1
        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Acc: {test_acc:.2%} | ({patience_counter}/{PATIENCE})")

    if patience_counter >= PATIENCE:
        print(f"\n⚠ Early stopping at epoch {epoch + 1}. Best model was from epoch {best_epoch}.")
        break

# Load best model and final evaluation
print("-" * 70)
model.load_state_dict(torch.load("data/models/viral_predictor_best.pth"))
final_loss, final_acc = evaluate(model, test_loader, criterion, device)
print(f"\n✓ Best model from epoch {best_epoch}")
print(f"  Test Loss: {final_loss:.4f}")
print(f"  Test Accuracy: {final_acc:.2%}")

# Save final model
torch.save(model.state_dict(), "data/models/viral_predictor.pth")
print("\nModel saved to data/models/viral_predictor.pth")
