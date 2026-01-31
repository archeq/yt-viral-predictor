import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class ViralDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, transform=None):
        # 1. Load CSV data
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer

        # 2. Filter mid-range v-scores (0.8 to 1.5)
        self.data = self.data[
            (self.data['V-Score'] > 1.5) | (self.data['V-Score'] < 0.8)
            ].reset_index(drop=True)

        # 3. If v-score is >1.5, label as 1 (viral), else 0 (non-viral)
        self.data['label'] = (self.data['V-Score'] > 1.5).astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get row data
        row = self.data.iloc[idx]
        video_id = row['Video ID']
        title_text = str(row['Title'])
        label = row['label']

        # Load image
        img_path = os.path.join(self.img_dir, f"{video_id}.jpg")

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback if image was not found
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        # Tokenize title text for BERT
        text_inputs = self.tokenizer(
            title_text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=50
        )

        return {
            'image': image,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }