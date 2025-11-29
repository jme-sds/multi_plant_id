import os
import sys
from torch.utils.data import DataLoader, Dataset

project_root = os.path.abspath('..')

if project_root not in sys.path:
    sys.path.append(project_root)

from loading.data_loader import SinglePlantDataLoader
from loading.quadrat import QuadratTilingDataset_Inference

import torch
from torchvision import transforms, datasets
import timm
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Subset
from collections import defaultdict
from sklearn.metrics import silhouette_score
import csv
from pathlib import Path
import argparse

from matplotlib import pyplot as plt
import random
from PIL import Image


assert "PLANT_HOME" in os.environ, f"Please set home/root directory of PlantCLEF files to the environment variable PLANT_HOME. (globus share in scratch)"
PLANT_HOME = os.getenv("PLANT_HOME")

def load_model(device):
    model = timm.create_model("timm/vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True)
    #model = timm.create_model("vit_base_patch16_224", pretrained=True)
    checkpoint_path = os.path.join(PLANT_HOME, "dinov2_model/model_best.pth.tar")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # Load to CPU first

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint # Assume the checkpoint itself is the state_dict

    # 3. Load the state dictionary into the model
    model.load_state_dict(state_dict,strict=False)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model

def extract_embeddings(dataloader, dataset,model, device):
    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, idx in tqdm(dataloader):
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)
            if feats.ndim == 3:
                cls_embs = feats[:, 0, :]  # [CLS] token
            else:
                cls_embs = feats

            true_ids = [class_to_speciesid[int(c)] for c in idx]

            all_embs.append(cls_embs.cpu().numpy())
            all_labels.append(true_ids)
    return np.concatenate(all_embs), np.concatenate(all_labels)

def extract_unlabeled_embeddings(dataloader, model, device):
    all_embs, all_paths = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):

            imgs = batch[0]      # shape (B, T, 3, 518, 518)
            paths = batch[1]     # list of length B

            # Flatten tiles:
            # imgs: (B, T, C, H, W) → (B*T, C, H, W)
            T = TILES_PER_SIDE**2
#            B, T, C, H, W = imgs.shape
#            imgs = imgs.view(B * T, C, H, W).to(device)

            B, C, H, W = imgs.shape
            imgs = imgs.view(B, C, H, W).to(device)
            feats = model.forward_features(imgs)
            cls_embs = feats[:, 0, :] if feats.ndim == 3 else feats

            # Save embeddings
            all_embs.append(cls_embs.cpu().numpy())

            # Save one path for every tile
            # If image X has 25 tiles → repeat the path 25 times
            for p in paths:
                all_paths.extend([p] * T)

    return np.concatenate(all_embs), all_paths

def build_faiss_index(embs, device):
    # Build index
    faiss_file = os.path.join(PLANT_HOME, "knn/faiss_index/faiss.idx")
    if os.path.exists(faiss_file):  # if index exists, load and use GPU if available
        index = faiss.read_index(faiss_file)
        print("Loaded FAISS index from {}".format(faiss_file))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            device_id = 0
            gpu_index = faiss.index_cpu_to_gpu(res, device_id, index)
            return gpu_index
        else:
            return index
    else:   # if index does not exist, build and save for later use, optionally use GPU if available
        index = faiss.IndexFlatIP(embs.shape[1])
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            device_id = 0
            gpu_index = faiss.index_cpu_to_gpu(res, device_id, index)
            gpu_index.add(embs.astype("float32"))
            faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), faiss_file)
            print("Saved FAISS index to {}".format(faiss_file))
            return gpu_index
        else:
            index.add(embs.astype("float32"))
            faiss.write_index(index, faiss_file)
            print("Saved FAISS index to {}".format(faiss_file))
            return index

def knn_predict(embs, index, faiss_labels, k=5):
    distances, indices = index.search(embs.astype("float32"), k)
    # Convert indices → species IDs
    preds = np.array([[faiss_labels[i] for i in row] for row in indices])
    return preds


def aggregate_predictions(tile_preds, tile_paths, tiles_per_quadrat=9, min_votes=1):
    quadrat_to_species = defaultdict(list)
    for i in range(0, len(tile_preds), tiles_per_quadrat):
        preds_for_quadrat = tile_preds[i:i+tiles_per_quadrat]
        path = tile_paths[i]
        quadrat_id = Path(path).stem
        # Flatten tile predictions: each tile has k neighbors
        flat_preds = preds_for_quadrat.flatten()
        # Voting
        counter = Counter(flat_preds)
        species = [sp for sp, count in counter.items() if count >= min_votes]
        quadrat_to_species[quadrat_id] = species
    return quadrat_to_species

def write_submission(pred_dict, out_csv="submission.csv"):
    print("Writing submission to {}".format(out_csv))
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quadrat_id", "species_ids"])
        for quad, species in pred_dict.items():
            s = "[" + ", ".join(str(x) for x in species) + "]"
            writer.writerow([quad, s])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_per_side", type=int, default=3, help="Number of tiles per side.")
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors to consider in KNN.")
    parser.add_argument("--votes", type=int, default=1, help="Number of votes to consider in voting for quadrat prediction.")
    parser.add_argument("--subset", type=bool, default=False, help="Use a subset of the training data.")
    parser.add_argument("--n_samples", type=int, default=640, help="Number of samples to use in the subset.")
    args = parser.parse_args()


    # Config Settings
    TILES_PER_SIDE = args.tiles_per_side
    SUBSET = args.subset
    N_SAMPLES = args.n_samples # only if SUBSET = True
    NEIGHBORS = args.neighbors
    VOTES = args.votes

    RESIZE_SIZE = 256
    IMG_SIZE = 518 
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count() # Use all available CPU cores for loading

#    transform = transforms.Compose([
#        transforms.Resize((518,518)),
#        transforms.ToTensor(),
#        transforms.Normalize(
#            mean=(0.485, 0.456, 0.406),
#            std=(0.229, 0.224, 0.225)
#        )
#    ])

    inference_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Crucial
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

#    class_to_speciesid = {class_idx: int(folder_name) for folder_name, class_idx in train_dataset.class_to_idx.items()}

    DATA_DIR = os.path.join(PLANT_HOME,"images_max_side_800")

    data_splitter = SinglePlantDataLoader(
        data_dir=DATA_DIR,
        resize_size=RESIZE_SIZE,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Get the dataloaders
    train_loader, val_loader, test_loader = data_splitter.get_dataloaders()


    QUADRAT_DIR = os.path.join(PLANT_HOME, "data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images")


    quadrat_test_set = QuadratTilingDataset_Inference(
        data_dir=QUADRAT_DIR,
        grid_size=(TILES_PER_SIDE, TILES_PER_SIDE),
        transform=inference_transform
    )

    assert len(quadrat_test_set) != 0, "Dataset is empty. Exiting."

    # 3. Instantiate the DataLoader
    # shuffle=False is critical for inference to keep tiles in order
    quadrat_loader = DataLoader(
        dataset=quadrat_test_set,
        batch_size=32, # Batch size of 32 *tiles*
        shuffle=False, 
        num_workers=1,
        pin_memory=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(device)

    train_file = os.path.join(PLANT_HOME,"knn/embeddings/train_embs.npz")
    if os.path.exists(train_file):
        with np.load(train_file) as data:
            train_embs = data['embs']
            train_labels = data['labels']
            print("Train embeddings loaded from", train_file)
    else:
        train_embs, train_labels = extract_embeddings(train_loader, train_dataset, model, device)
        np.savez(train_file, embs=train_embs, labels=train_labels)



    filename = os.path.join(PLANT_HOME, "knn/embeddings/quadrat_embs_"+str(TILES_PER_SIDE)+"x"+str(TILES_PER_SIDE)+".npz")
    if os.path.exists(filename):
        with np.load(filename) as data:
            quadrat_embs = data['embs']
            quadrat_paths = data['paths']
            print("Quadrat embeddings loaded from", filename)
    else:
        print(filename, " not found. Extracting embeddings...")
        quadrat_embs, quadrat_paths = extract_unlabeled_embeddings(quadrat_loader, model, device)
        np.savez(filename, embs=quadrat_embs, paths=quadrat_paths)


    faiss.normalize_L2(quadrat_embs)
    faiss.normalize_L2(train_embs)

    index = build_faiss_index(train_embs, device)

    # Tile-level predictions
    tile_preds = knn_predict(quadrat_embs, index = index, faiss_labels=train_labels, k=NEIGHBORS)
    # Quadrat-level species predictions
    quadrat_preds = aggregate_predictions(tile_preds, quadrat_paths, tiles_per_quadrat=TILES_PER_SIDE*TILES_PER_SIDE, min_votes=VOTES)
    # Write CSV
    write_submission(quadrat_preds,out_csv= os.path.join(PLANT_HOME,"submissions/knn_submission_"+str(TILES_PER_SIDE)+"x"+str(TILES_PER_SIDE)+"_v"+str(VOTES)+"_n"+str(NEIGHBORS)+".csv"))
