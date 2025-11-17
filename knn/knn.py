import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import timm
import numpy as np
import pandas as pd
import os
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

def tile_image_nxn(img, tiles_per_side=3, target_size=518):
    w, h = img.size

    tile_w = w / tiles_per_side
    tile_h = h / tiles_per_side

    tiles = []

    for row in range(tiles_per_side):
        for col in range(tiles_per_side):
            left = int(col * tile_w)
            top = int(row * tile_h)
            right = int((col + 1) * tile_w)
            bottom = int((row + 1) * tile_h)

            tile = img.crop((left, top, right, bottom))
            tile = tile.resize((target_size, target_size), Image.BICUBIC)
            tiles.append(tile)

    return tiles

class QuadratNxNDataset(Dataset):
    def __init__(self, root, transform, tiles_per_side=3, target_size=518):
        self.paths = sorted([str(p) for p in Path(root).glob("*.*")])
        self.transform = transform
        self.tiles_per_side = tiles_per_side
        self.target_size = target_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        tiles = tile_image_nxn(
            img,
            tiles_per_side=self.tiles_per_side,
            target_size=self.target_size
        )

        tiles = [self.transform(t) for t in tiles]
        tiles = torch.stack(tiles)  # shape [N*N, 3, 518, 518]

        return tiles, path


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
            B, T, C, H, W = imgs.shape
            imgs = imgs.view(B * T, C, H, W).to(device)

            feats = model.forward_features(imgs)
            cls_embs = feats[:, 0, :] if feats.ndim == 3 else feats

            # Save embeddings
            all_embs.append(cls_embs.cpu().numpy())

            # Save one path for every tile
            # If image X has 25 tiles → repeat the path 25 times
            for p in paths:
                all_paths.extend([p] * T)

    return np.concatenate(all_embs), all_paths

def knn_predict(embs, k=5):
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

    transform = transforms.Compose([
        transforms.Resize((518,518)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])


    train_dataset = datasets.ImageFolder(os.path.join(PLANT_HOME, "images_max_side_800"), transform=transform)

    if SUBSET:
        n_samples = N_SAMPLES
        indices = np.random.choice(len(train_dataset), n_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)
        class_to_speciesid = {class_idx: int(folder_name) for folder_name, class_idx in train_dataset.dataset.class_to_idx.items()}
    else:
        class_to_speciesid = {class_idx: int(folder_name) for folder_name, class_idx in train_dataset.class_to_idx.items()}

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)


    quadrat_path = os.path.join(PLANT_HOME, "data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images")

    quadrat_loader = DataLoader(
        QuadratNxNDataset(
            quadrat_path,
            transform,
            tiles_per_side=TILES_PER_SIDE,
            target_size=518
        ),
        batch_size=1,
        shuffle=False,
        num_workers=4
    )


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        quadrat_embs, quadrat_paths = extract_unlabeled_embeddings(quadrat_loader, model, device)
        np.savez(filename, embs=quadrat_embs, paths=quadrat_paths)

    faiss.normalize_L2(quadrat_embs)
    faiss.normalize_L2(train_embs)
    index = faiss.IndexFlatIP(train_embs.shape[1])
    index.add(train_embs)
    #D, I = index.search(quadrat_embs, k=5)
    faiss_labels = train_labels



    # Tile-level predictions
    tile_preds = knn_predict(quadrat_embs, k=NEIGHBORS)
    # Quadrat-level species predictions
    quadrat_preds = aggregate_predictions(tile_preds, quadrat_paths, tiles_per_quadrat=TILES_PER_SIDE*TILES_PER_SIDE, min_votes=VOTES)
    # Write CSV
    write_submission(quadrat_preds,out_csv= os.path.join(PLANT_HOME,"submissions/knn_submission_"+str(TILES_PER_SIDE)+"x"+str(TILES_PER_SIDE)+"_v"+str(VOTES)+"_n"+str(NEIGHBORS)+".csv"))
