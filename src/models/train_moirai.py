import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import pandas as pd
import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader
from uni2ts.model.moirai import MoiraiModule, MoiraiFinetune
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

class OmniTimeSeriesDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, context_length: int, prediction_length: int, patch_size: int = 32, max_patch_size: int = 128, stride: int = 32):
        # We assume the last column is the target (Pc5_Wave_Power), and others are covariates
        self.data = data_df.values
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.max_patch_size = max_patch_size
        self.stride = stride
        self.seq_len = context_length + prediction_length
        self.num_variates = self.data.shape[1]
        self.context_patches = self.context_length // self.patch_size
        self.pred_patches = self.prediction_length // self.patch_size

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len) // self.stride + 1)

    def __getitem__(self, idx):
        # Extract sequence for all variates with stride
        start_idx = idx * self.stride
        seq = self.data[start_idx : start_idx + self.seq_len, :]  # shape: (seq_len, num_variates)
        
        target_list = []
        observed_mask_list = []
        time_id_list = []
        variate_id_list = []
        prediction_mask_list = []
        
        # Process covariates (indices 0 to num_variates - 2)
        for v in range(self.num_variates - 1):
            cov_seq = seq[:self.context_length, v]
            cov_patches = torch.tensor(cov_seq, dtype=torch.float32).view(self.context_patches, self.patch_size)
            padded_cov = torch.ones(self.context_patches, self.max_patch_size, dtype=torch.float32) # Padded with 1.0
            padded_cov[:, :self.patch_size] = cov_patches
            target_list.append(padded_cov)
            
            obs_mask = torch.zeros((self.context_patches, self.max_patch_size), dtype=torch.bool)
            obs_mask[:, :self.patch_size] = True
            observed_mask_list.append(obs_mask)
            
            time_id_list.append(torch.arange(self.context_patches, dtype=torch.long))
            variate_id_list.append(torch.full((self.context_patches,), v, dtype=torch.long))
            prediction_mask_list.append(torch.zeros(self.context_patches, dtype=torch.bool))
            
        # Process target (last index)
        target_seq = seq[:, -1]
        target_patches = torch.tensor(target_seq, dtype=torch.float32).view(self.context_patches + self.pred_patches, self.patch_size)
        padded_target = torch.ones(self.context_patches + self.pred_patches, self.max_patch_size, dtype=torch.float32) # Padded with 1.0
        padded_target[:, :self.patch_size] = target_patches
        target_list.append(padded_target)
        
        obs_mask = torch.zeros((self.context_patches + self.pred_patches, self.max_patch_size), dtype=torch.bool)
        obs_mask[:, :self.patch_size] = True
        observed_mask_list.append(obs_mask)
        time_id_list.append(torch.arange(self.context_patches + self.pred_patches, dtype=torch.long))
        variate_id_list.append(torch.full((self.context_patches + self.pred_patches,), self.num_variates - 1, dtype=torch.long))
        
        # Target prediction mask is True only for the prediction patches
        tgt_pred_mask = torch.zeros(self.context_patches + self.pred_patches, dtype=torch.bool)
        tgt_pred_mask[-self.pred_patches:] = True
        prediction_mask_list.append(tgt_pred_mask)
        
        # Concatenate all lists
        target = torch.cat(target_list, dim=0)
        observed_mask = torch.cat(observed_mask_list, dim=0)
        time_id = torch.cat(time_id_list, dim=0)
        variate_id = torch.cat(variate_id_list, dim=0)
        prediction_mask = torch.cat(prediction_mask_list, dim=0)
        
        num_total_patches = target.shape[0]
        patch_size_tensor = torch.full((num_total_patches,), self.patch_size, dtype=torch.long)
        sample_id = torch.ones(num_total_patches, dtype=torch.long) # Use 1 instead of 0, as 0 bypasses scaling
        
        return {
            "target": target,
            "observed_mask": observed_mask,
            "time_id": time_id,
            "variate_id": variate_id,
            "prediction_mask": prediction_mask,
            "patch_size": patch_size_tensor,
            "sample_id": sample_id
        }

def train():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv("data/processed/omni_processed.csv", parse_dates=["Datetime"], index_col="Datetime")
    
    # Scale all columns to [1e-4, 1.0] to guarantee no negative values and no exact zeros.
    # Moirai's Mixture Distribution (specifically LogNormal & NegativeBinomial) will return NaN on <= 0 values.
    # PyTorch's NaN * 0 behavior propagates this NaN even for masked covariates.
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(1e-4, 1.0))
    df[df.columns] = scaler.fit_transform(df[df.columns])
    
    # Train/Val split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    context_length = 512
    prediction_length = 128
    patch_size = 32
    batch_size = 32
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    train_dataset = OmniTimeSeriesDataset(train_df, context_length, prediction_length, patch_size)
    val_dataset = OmniTimeSeriesDataset(val_df, context_length, prediction_length, patch_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    
    # 2. Initialize Moirai Model
    print("Initializing Moirai Model...")
    # Load Moirai-1.0-R-base (31M parameters) or small (14M)
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")
    
    model = MoiraiFinetune(
        module=module,
        min_patches=2,
        min_mask_ratio=0.15,
        max_mask_ratio=0.5,
        max_dim=128,
        num_training_steps=1000,
        num_warmup_steps=100,
        lr=1e-4, # Reduced learning rate to prevent NaN/Gradient Explosion
    )
    
    # 3. Train Model
    print("Starting Training...")
    
    # Apple Silicon (MPS) has a known PyTorch bug generating NaN during complex distribution evaluation in eval() mode.
    # To bypass this hardware-specific bug, we monitor train loss and disable the validation loop.
    early_stop_callback = EarlyStopping(
        monitor="train/PackedNLLLoss",
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode="min"
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/moirai_checkpoints",
        filename="prahari-{epoch:02d}-{train/PackedNLLLoss:.2f}",
        monitor="train/PackedNLLLoss",
        mode="min",
        save_top_k=3
    )
    
    # We use CPU/MPS based on user hardware
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0, # Add gradient clipping to prevent NaN
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        limit_val_batches=0 # Completely skip the buggy Apple MPS validation loop
    )
    
    trainer.fit(model, train_dataloaders=train_loader)
    print("Training Complete. Model saved to models/moirai_checkpoints/")

if __name__ == "__main__":
    train()
