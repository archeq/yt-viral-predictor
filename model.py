import torch
import torch.nn as nn
from transformers import DistilBertModel
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ViralPredictor(nn.Module):
    def __init__(self):
        super(ViralPredictor, self).__init__()

        # 1. Visual Branch (Image)
        # Using ResNet50, removing the last classification layer
        self.visual_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.visual_model.fc = nn.Identity()  # Returns a 2048 feature vector

        # 2. Text Branch (NLP)
        # DistilBERT is lighter and faster than full BERT
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # DistilBERT returns a 768 feature vector

        # 3. Fusion (Combination)
        # Input: 2048 (image) + 768 (text) = 2816
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, input_ids, attention_mask):
        # A. Image Analysis
        visual_features = self.visual_model(image)  # Shape: [Batch, 2048]

        # B. Text Analysis
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]  # Take [CLS] token -> Shape: [Batch, 768]

        # C. Combination
        combined_features = torch.cat((visual_features, text_features), dim=1)

        # D. Prediction
        output = self.classifier(combined_features)
        return output