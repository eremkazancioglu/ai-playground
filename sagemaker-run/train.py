#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import seaborn as sns
import time
import boto3
from botocore.exceptions import ClientError

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeciesDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_name']
        img_path = os.path.join(self.data_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        label = self.df.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SageMakerCheckpoint:
    """Handle checkpointing for SageMaker spot instances"""
    
    def __init__(self, checkpoint_dir, s3_bucket=None, s3_prefix=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3') if s3_bucket else None
        
    def save_checkpoint(self, state, filename):
        """Save checkpoint locally and optionally to S3"""
        local_path = self.checkpoint_dir / filename
        torch.save(state, local_path)
        logger.info(f"Saved checkpoint to {local_path}")
        
        # Upload to S3 if configured
        if self.s3_client and self.s3_bucket:
            try:
                s3_key = f"{self.s3_prefix}/{filename}" if self.s3_prefix else filename
                self.s3_client.upload_file(str(local_path), self.s3_bucket, s3_key)
                logger.info(f"Uploaded checkpoint to s3://{self.s3_bucket}/{s3_key}")
            except ClientError as e:
                logger.error(f"Failed to upload checkpoint to S3: {e}")
    
    def load_checkpoint(self, filename):
        """Load checkpoint from local or S3"""
        local_path = self.checkpoint_dir / filename
        
        # Try loading from local first
        if local_path.exists():
            return torch.load(local_path, map_location='cpu')
        
        # Try downloading from S3
        if self.s3_client and self.s3_bucket:
            try:
                s3_key = f"{self.s3_prefix}/{filename}" if self.s3_prefix else filename
                self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
                logger.info(f"Downloaded checkpoint from s3://{self.s3_bucket}/{s3_key}")
                return torch.load(local_path, map_location='cpu')
            except ClientError as e:
                logger.info(f"No checkpoint found in S3: {e}")
        
        return None
    
    def list_checkpoints(self):
        """List available checkpoints"""
        return list(self.checkpoint_dir.glob("*.pth"))

def prepare_data(csv_file, target_species, downsample_other=False, downsample_frac=0.1):
    """Load and prepare data with 'other' class for non-target species"""
    logger.info("=== PREPARE_DATA DEBUG ===")
    logger.info(f"target_species received: {target_species}")
    logger.info(f"target_species type: {type(target_species)}")

    df = pd.read_csv(csv_file)
    logger.info(f"CSV loaded, shape: {df.shape}")
    logger.info(f"Species in CSV (first 10): {df['species_name'].unique()[:10]}")
    logger.info(f"Number of rows with Lycorma delicatula: {(df['species_name'] == 'Lycorma delicatula').sum()}")
    
    # Create binary classification: target species vs 'other'
    df['label_name'] = df['species_name'].apply(
        lambda x: x if x in target_species else 'other'
    )
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label_name'])
    
    logger.info("Class distribution:")
    logger.info(df['label_name'].value_counts())

    if downsample_other:
        logger.info(f"Downsampling 'other' to {downsample_frac}")
        downsampled_target = df[df['label_name'] == 'other'].sample(frac=downsample_frac, random_state=42)

        # Combine with other labels
        other_labels = df[df['label_name'] != 'other']
        df = pd.concat([other_labels, downsampled_target]).reset_index(drop=True)

        logger.info("New class distribution:")
        logger.info(df['label_name'].value_counts())

    species_to_label = dict(zip(le.classes_, range(len(le.classes_))))
    
    return df, species_to_label

def get_transforms(img_size):
    """Define training and validation transforms"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model_efficientnet(num_classes):
    """Create EfficientNet model"""
    model = models.efficientnet_b1(pretrained=True)
    
    # Explicitly unfreeze all layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    return model

def get_simplified_differential_lr_efficientnet(model, base_lr=0.0001):
    """Simplified version with fewer parameter groups for 9 blocks"""
    param_groups = [
        # Early blocks (0-2) - preserve low-level features
        {'params': [p for i in range(3) for p in model.features[i].parameters()], 
         'lr': base_lr * 0.1},
        
        # Middle blocks (3-5) - moderate adaptation
        {'params': [p for i in range(3, 6) for p in model.features[i].parameters()], 
         'lr': base_lr * 0.5},
        
        # Late blocks (6-8) - higher adaptation
        {'params': [p for i in range(6, 9) for p in model.features[i].parameters()], 
         'lr': base_lr * 1.0},
        
        # Classifier - highest LR
        {'params': model.classifier.parameters(), 'lr': base_lr * 10.0}
    ]
    
    return optim.Adam(param_groups)

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    class_counts = Counter(labels)
    total = len(labels)
    weights = {cls: total / (len(class_counts) * count) 
              for cls, count in class_counts.items()}
    return torch.FloatTensor([weights[i] for i in range(len(class_counts))])

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    # Debug: Check first batch
    first_batch = True
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        if first_batch:
            logger.info(f"First batch - Images shape: {images.shape}")
            logger.info(f"First batch - Labels shape: {labels.shape}")
            logger.info(f"First batch - Labels values: {labels.cpu().numpy()}")
            logger.info(f"First batch - Labels unique: {torch.unique(labels).cpu().numpy()}")
            logger.info(f"First batch - Images min/max: {images.min().item():.3f}/{images.max().item():.3f}")
            first_batch = False
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Debug: Check if loss is actually being computed
        if batch_idx == 0:
            logger.info(f"Raw loss value: {loss.item()}")
            logger.info(f"Outputs shape: {outputs.shape}")
            logger.info(f"Outputs sample: {outputs[0].detach().cpu().numpy()}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 100 == 0:
            logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
    
    return epoch_loss, epoch_f1

def validate_epoch(model, val_loader, criterion, device, return_predictions=False):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
    
    if return_predictions:
        return epoch_loss, epoch_f1, all_predictions, all_labels
    else:
        return epoch_loss, epoch_f1

def train_fold(train_df, val_df, fold_num, species_to_label, class_weights, 
               config, checkpoint_manager, device):
    """Train and validate one fold with checkpointing"""
    logger.info(f"\n=== Fold {fold_num + 1} ===")
    
    # Create transforms
    train_transform, val_transform = get_transforms(config['img_size'])
    
    # Create datasets
    train_dataset = SpeciesDataset(train_df, config['data_dir'], train_transform)
    val_dataset = SpeciesDataset(val_df, config['data_dir'], val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = create_model_efficientnet(len(species_to_label)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = get_simplified_differential_lr_efficientnet(model, base_lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Try to load checkpoint
    checkpoint_file = f"fold_{fold_num + 1}_checkpoint.pth"
    checkpoint = checkpoint_manager.load_checkpoint(checkpoint_file)
    
    start_epoch = 0
    best_f1 = 0.0
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    best_f1_epochs_ago = 0
    
    if checkpoint:
        logger.info(f"Resuming from checkpoint at epoch {checkpoint['epoch']}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_f1s = checkpoint.get('train_f1s', [])
        val_f1s = checkpoint.get('val_f1s', [])
        best_f1_epochs_ago = checkpoint.get('best_f1_epochs_ago', 0)
        best_model_state = checkpoint['model_state_dict'].copy()  # ← Add this line
    else:
        # Initialize best_model_state for new training
        best_model_state = model.state_dict().copy()
    
    # Training loop
    for epoch in range(start_epoch, config['num_epochs']):
        logger.info(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_f1)
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        
        logger.info(f'  Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        logger.info(f'  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        
        # Save checkpoint every epoch
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_f1s': train_f1s,
            'val_f1s': val_f1s,
            'best_f1_epochs_ago': best_f1_epochs_ago,
            'fold_num': fold_num
        }
        checkpoint_manager.save_checkpoint(checkpoint_state, checkpoint_file)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_f1_epochs_ago = 0
            best_model_state = model.state_dict().copy()
            # Save best model
            best_model_file = f"best_model_fold_{fold_num + 1}.pth"
            checkpoint_manager.save_checkpoint(best_model_state, best_model_file)
        else:
            best_f1_epochs_ago += 1
        
        if best_f1_epochs_ago >= config['epoch_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and get final predictions
    model.load_state_dict(best_model_state)
    _, _, final_predictions, final_labels = validate_epoch(
        model, val_loader, criterion, device, return_predictions=True)
    
    return best_f1, final_predictions, final_labels, (train_losses, val_losses, train_f1s, val_f1s)

def plot_training_curves(fold_metrics, k_folds, output_dir):
    """Plot training curves for all folds"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for fold in range(k_folds):
        train_losses, val_losses, train_f1s, val_f1s = fold_metrics[fold]
        epochs = range(1, len(train_losses) + 1)
        
        axes[0, 0].plot(epochs, train_losses, label=f'Fold {fold+1}', alpha=0.7)
        axes[0, 1].plot(epochs, val_losses, label=f'Fold {fold+1}', alpha=0.7)
        axes[1, 0].plot(epochs, train_f1s, label=f'Fold {fold+1}', alpha=0.7)
        axes[1, 1].plot(epochs, val_f1s, label=f'Fold {fold+1}', alpha=0.7)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    axes[1, 0].set_title('Training F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(fold_results, species_to_label, output_dir):
    """Plot confusion matrices for each fold and average"""
    label_names = [''] * len(species_to_label)
    for species, label in species_to_label.items():
        label_names[label] = species
    
    n_folds = len(fold_results)
    n_classes = len(species_to_label)
    
    # Calculate individual fold confusion matrices
    fold_cms = []
    for fold_predictions, fold_labels in fold_results:
        cm = confusion_matrix(fold_labels, fold_predictions, labels=range(n_classes))
        fold_cms.append(cm)
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot individual fold confusion matrices
    for i, cm in enumerate(fold_cms):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_names, yticklabels=label_names,
                    ax=axes[i])
        axes[i].set_title(f'Fold {i+1}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Calculate and plot average confusion matrix
    avg_cm = np.mean(fold_cms, axis=0)
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[n_folds])
    axes[n_folds].set_title('Average Confusion Matrix')
    axes[n_folds].set_xlabel('Predicted')
    axes[n_folds].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a separate figure for normalized confusion matrices (all folds + average)
    n_folds = len(fold_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Create label names
    label_names = [''] * len(species_to_label)
    for species, label in species_to_label.items():
        label_names[label] = species
    
    # Plot individual fold normalized confusion matrices
    fold_cms_norm = []
    for i, cm in enumerate(fold_cms):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
        fold_cms_norm.append(cm_norm)
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=label_names, yticklabels=label_names,
                    ax=axes[i], vmin=0, vmax=1)
        axes[i].set_title(f'Fold {i+1} (Normalized)')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Calculate and plot average normalized confusion matrix
    avg_cm_norm = np.mean(fold_cms_norm, axis=0)
    sns.heatmap(avg_cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                ax=axes[n_folds], vmin=0, vmax=1)
    axes[n_folds].set_title('Average Normalized Confusion Matrix')
    axes[n_folds].set_xlabel('Predicted')
    axes[n_folds].set_ylabel('Actual')
    
    # Hide any unused subplots
    for i in range(n_folds + 1, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fold_cms, avg_cm

def main():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--checkpoint-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    # Training arguments
    parser.add_argument('--target-species', type=str, 
                       default='"Lycorma delicatula,Zenaida macroura,Acer rubrum,Erigeron canadensis"')
    parser.add_argument('--csv-file', type=str, default='new_balto_species_photos_augmented_subsample_other.csv')
    parser.add_argument('--img-size', type=int, default=240)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--k-folds', type=int, default=3)
    parser.add_argument('--epoch-patience', type=int, default=5)
    parser.add_argument('--downsample-other', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--downsample-frac', type=float, default=0.5)
    
    # S3 checkpointing
    parser.add_argument('--s3-bucket', type=str, default=None)
    parser.add_argument('--s3-checkpoint-prefix', type=str, default='checkpoints')
    
    args = parser.parse_args()

    # Convert comma-separated string to list
    args.target_species = [s.strip() for s in args.target_species.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device with SageMaker-specific optimizations
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Enable optimizations for SageMaker GPU instances
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using GPU device: {device}")
        logger.info(f"GPU device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        # Optimize CPU performance for SageMaker CPU instances
        torch.set_num_threads(os.cpu_count())
        logger.info(f"Using CPU device with {os.cpu_count()} threads")
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'csv_file': os.path.join(args.data_dir, args.csv_file),
        'target_species': args.target_species,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'k_folds': args.k_folds,
        'epoch_patience': args.epoch_patience,
    }
    
    # Setup checkpoint manager
    checkpoint_manager = SageMakerCheckpoint(
        args.checkpoint_dir, 
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_checkpoint_prefix
    )
    
    # Load and prepare data
    logger.info("Loading data...")
    df, species_to_label = prepare_data(
        config['csv_file'], 
        config['target_species'], 
        downsample_other=args.downsample_other, 
        downsample_frac=args.downsample_frac
    )

    # Debug: Print dataset info
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Class distribution after downsampling:")
    logger.info(df['label_name'].value_counts())
    logger.info(f"Numeric labels distribution:")
    logger.info(df['label'].value_counts())
    logger.info(f"Species to label mapping: {species_to_label}")
    
    num_classes = len(species_to_label)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {list(species_to_label.keys())}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(df['label'].values)
    logger.info(f"Class weights: {class_weights}")
    
    # Prepare for stratified k-fold
    X = df.index.values
    y = df['label'].values
    
    if config['k_folds'] == 1:
        # Simple train/test split for dry run
        from sklearn.model_selection import train_test_split
        
        train_idx, val_idx = train_test_split(
            X, test_size=0.3, stratify=y, random_state=42
        )
        
        fold_splits = [(train_idx, val_idx)]
        logger.info("Using simple train/test split (70/30)")
        
    else:
        # Regular k-fold cross validation
        skf = StratifiedKFold(n_splits=config['k_folds'], shuffle=True, random_state=42)
        fold_splits = list(skf.split(X, y))
        logger.info(f"Using {config['k_folds']}-fold cross validation")
    
    fold_f1_scores = []
    fold_results = []
    fold_metrics = []
    
    # K-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        fold_f1, fold_predictions, fold_labels, fold_training_metrics = train_fold(
            train_df, val_df, fold, species_to_label, class_weights,
            config, checkpoint_manager, device
        )
        
        fold_f1_scores.append(fold_f1)
        fold_results.append((fold_predictions, fold_labels))
        fold_metrics.append(fold_training_metrics)
        
        logger.info(f"Fold {fold + 1} F1 Score: {fold_f1:.4f}")
    
    # Calculate overall results
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    
    logger.info(f"\n=== FINAL RESULTS ===")
    logger.info(f"F1 Scores by fold: {[f'{score:.4f}' for score in fold_f1_scores]}")
    logger.info(f"Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Generate plots
    plot_training_curves(fold_metrics, config['k_folds'], output_dir)
    fold_cms, avg_cm = plot_confusion_matrices(fold_results, species_to_label, output_dir)
    
    # Save final results
    results = {
        'fold_f1_scores': fold_f1_scores,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'fold_results': fold_results,
        'fold_confusion_matrices': fold_cms,
        'average_confusion_matrix': avg_cm,
        'species_to_label': species_to_label,
        'config': config
    }
    
    # Save to both model dir and output dir
    torch.save(results, os.path.join(args.model_dir, 'training_results.pth'))
    torch.save(results, output_dir / 'training_results.pth')
    
    # Save species mapping for inference
    with open(os.path.join(args.model_dir, 'species_to_label.json'), 'w') as f:
        json.dump(species_to_label, f)
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main()