import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import zipfile
import urllib.request
from tqdm import tqdm
import sys 
import os

project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.append(project_root) 
    
project_root
train_data_path = os.path.join(project_root, "PlantCLEF2025_data/images_max_side_800")
saved_model_path = os.path.join(project_root, "resnet50")

from loading.data_loader import SinglePlantDataLoader
from resnet50 import resnet50_trainhead

#os.chdir(train_data_path)
#os.chdir("/scratch/hl9h/images_max_side_800")
DATA_DIR = train_data_path

RESIZE_SIZE = 256
IMG_SIZE = 224 
BATCH_SIZE = 32
NUM_CLASSES = 7806  # As per the challenge overview
NUM_WORKERS = os.cpu_count()  # Use all available CPU cores for loading

data_splitter = SinglePlantDataLoader(
        data_dir=DATA_DIR,
        resize_size=RESIZE_SIZE,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

# Get the dataloaders
train_loader, val_loader, test_loader = data_splitter.get_dataloaders()


#--------------------------------------------------------------------------
# Traing the Classification Head of ResNet50 Model
#--------------------------------------------------------------------------

# Initialize Configuations and Data Sets
NUM_CLASS = 7806
NUM_EPOCH = 20
LR = 0.01
DEVICE = resnet50_trainhead.device
print(f'Using device: {DEVICE}')


# Traing the Classification Head of ResNet50 with PlantCLEF Training Data
print("\n" + "="*60)
print("Transfer Learning - Training the Classification Head of ResNet50 with PlantCLEF Training Data")
print("="*60)

try:
    print("\nTraining the Head of ResNet50 ...")
    
    pretrained_resnet = resnet50_trainhead.get_resnet50_pretrained(num_classes=NUM_CLASS, fine_tune=False)
    num_params, trainable_params = resnet50_trainhead.count_parameters(pretrained_resnet)
    model_size = resnet50_trainhead.get_model_size_mb(pretrained_resnet)
    print(f"Total parameters: {num_params:,d}")
    print(f"Trainable parameters: {trainable_params:,d}")
    print(f"Model size: {model_size:.2f} MB")
    
    trainhead_history = resnet50_trainhead.train_model(pretrained_resnet, 
                                   train_loader, 
                                   val_loader,
                                   device=DEVICE,
                                   num_epochs=NUM_EPOCH, 
                                   lr=LR)
    
    #resnet50.plot_training_history(finetune_history, title="Fine Tuning History")
    results_df = pd.DataFrame(trainhead_history)
    results_df.to_csv(f'{saved_model_path}/trainhead_history.csv', index=False)
    print(f"\nTraining {NUM_EPOCH} epochs finished! Head training history saved to 'trainhead_history.csv'")
    
except Exception as e:
    print(f"Error in Transfer Learning (Train-head): {e}")
    trainhead_history = {}


