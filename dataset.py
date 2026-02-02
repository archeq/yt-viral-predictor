# dataset.py - Dataset class for viral video prediction

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import glob
from sklearn.model_selection import train_test_split


class ViralDataset(Dataset):
    def __init__(self, data, img_dir, tokenizer, transform=None):
        """
        Initialize dataset with pre-processed DataFrame.

        Args:
            data: pandas DataFrame with columns ['Video ID', 'Title', 'label']
            img_dir: path to thumbnails directory
            tokenizer: HuggingFace tokenizer
            transform: image transformations
        """
        self.data = data.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row['Video ID']
        title_text = str(row['Title'])
        label = row['label']

        img_path = os.path.join(self.img_dir, f"{video_id}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        text_inputs = self.tokenizer(
            title_text, return_tensors="pt", padding='max_length', truncation=True, max_length=50
        )

        return {
            'image': image,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


def get_train_test_datasets(csv_dir, img_dir, tokenizer, transform=None, test_size=0.1, random_state=42):
    """
    Load data, balance classes, and split into train/test datasets.

    Args:
        csv_dir: path to CSV files directory
        img_dir: path to thumbnails directory
        tokenizer: HuggingFace tokenizer
        transform: image transformations
        test_size: fraction of data for testing (default 0.1 = 10%)
        random_state: random seed for reproducibility

    Returns:
        train_dataset, test_dataset: ViralDataset instances
    """
    # 1. Load ALL CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    df_list = []
    for filename in csv_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception:
            pass

    data = pd.concat(df_list, ignore_index=True)

    # 2. Remove rows with null V-Score
    null_count = data['V-Score'].isna().sum()
    if null_count > 0:
        print(f"Removing {null_count} rows with null V-Score (insufficient history)")
    data = data.dropna(subset=['V-Score'])

    # 3. Filter data (Remove "average" performers)
    # V-Score > 1.0 = viral, V-Score < -0.5 = weak
    data = data[
        (data['V-Score'] > 1.0) | (data['V-Score'] < -0.5)
    ].reset_index(drop=True)

    # 4. Create labels
    data['label'] = (data['V-Score'] > 1.0).astype(int)

    # 5. Balance data
    df_viral = data[data['label'] == 1]
    df_weak = data[data['label'] == 0]
    min_len = min(len(df_viral), len(df_weak))

    print(f"Stats before balancing: Viral={len(df_viral)}, Weak={len(df_weak)}")

    df_viral = df_viral.sample(n=min_len, random_state=random_state)
    df_weak = df_weak.sample(n=min_len, random_state=random_state)

    balanced_data = pd.concat([df_viral, df_weak]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"DATA BALANCED: {min_len} viral vs {min_len} weak. Total: {len(balanced_data)}")

    # 6. Split into train/test (90/10)
    train_data, test_data = train_test_split(
        balanced_data,
        test_size=test_size,
        random_state=random_state,
        stratify=balanced_data['label']  # Maintain class balance in both sets
    )

    print(f"Train set: {len(train_data)} samples ({100*(1-test_size):.0f}%)")
    print(f"Test set: {len(test_data)} samples ({100*test_size:.0f}%)")

    # 7. Create datasets
    train_dataset = ViralDataset(train_data, img_dir, tokenizer, transform)
    test_dataset = ViralDataset(test_data, img_dir, tokenizer, transform)

    return train_dataset, test_dataset
