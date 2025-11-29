import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Dataset
import os
import glob
from PIL import Image


# os.chdir("/scratch/ezq9qu/images_max_side_800")
# DATA_DIR = os.getcwd()


# RESIZE_SIZE = 256
# IMG_SIZE = 224 
# BATCH_SIZE = 32
# NUM_CLASSES = 7806 # As per the challenge overview
# NUM_WORKERS = os.cpu_count() # Use all available CPU cores for loading



      
class QuadratTilingDataset_Inference(Dataset):
  """
  A custom PyTorch Dataset for loading and tiling UNLABELED quadrat images.
  
  This version tiles an image into a grid (e.g., 3x3) and is used for
  inference. It returns the tile and the file path of its parent image.
  """
  def __init__(self, data_dir, grid_size=(3, 3), transform=None):
      """
      Initializes the QuadratTilingDataset for inference.

      Args:
          data_dir (str): Path to the folder containing quadrat images.
          grid_size (tuple): A (rows, cols) tuple for the tiling grid.
          transform (callable, optional): A transform to be applied to each tile.
                                          (Must include Resize, ToTensor, Normalize)
      """
      self.data_dir = data_dir
      self.transform = transform
      self.samples = []
      
      # --- Grid Logic ---
      self.num_rows, self.num_cols = grid_size
      self.num_tiles = self.num_rows * self.num_cols
      print(f"Initializing inference dataset for a {self.num_rows}x{self.num_cols} grid.")
      
      self._create_samples()

  def _create_samples(self):
      """
      --- MODIFIED ---
      Scans the data_dir for all images and creates (image_path, tile_index) samples.
      """
      
      img_paths = []
      # Find all common image types
      for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.JPG", "*.JPEG"]:
           img_paths.extend(glob.glob(os.path.join(self.data_dir, ext)))
  

      for img_path in sorted(img_paths): # Sort to make it reproducible
          for i in range(self.num_tiles): # e.g., 0-8
              self.samples.append((img_path, i))
              
      print(f"Found {len(img_paths)} images, creating {len(self.samples)} total tiles.")

  def __len__(self):
      """Returns the total number of tiles."""
      return len(self.samples)

  def __getitem__(self, idx):
      """
      --- MODIFIED ---
      Fetches the tile at the given index.
      Returns the tile tensor and the original image path.
      """
      img_path, tile_index = self.samples[idx]

      try:
          img = Image.open(img_path).convert("RGB")
          
          # Extract the tile from the grid
          tile = self._get_tile(img, tile_index)

          if self.transform:
              tile = self.transform(tile)
          
          # --- MODIFIED RETURN ---
          # Return the tile and its source path
          # The path is crucial for grouping predictions later
          return tile, img_path

      except Exception as e:
          print(f"Error loading tile for image {img_path} at index {idx}: {e}")
          # Assumes IMG_SIZE is defined in your script's scope
          dummy_tile = torch.zeros((3, IMG_SIZE, IMG_SIZE)) 
          return dummy_tile, "error_path"


  def _get_tile(self, img, tile_index):
      """
      Extracts a specific tile from the image based on the (rows, cols) grid.
      """
      img_width, img_height = img.size
      
      # Calculate the size of each tile
      tile_width = img_width // self.num_cols
      tile_height = img_height // self.num_rows
      
      # Calculate the row and column of this tile_index
      row = tile_index // self.num_cols
      col = tile_index % self.num_cols
      
      # Calculate the crop box coordinates (left, top, right, bottom)
      left = col * tile_width
      top = row * tile_height
      
      # Handle edge cases for images not perfectly divisible
      if col == self.num_cols - 1:
          right = img_width
      else:
          right = (col + 1) * tile_width
          
      if row == self.num_rows - 1:
          bottom = img_height
      else:
          bottom = (row + 1) * tile_height
          
      tile = img.crop((left, top, right, bottom))
      
      return tile


def main():
  """
  Main function
  """



  # 1. Define the transform
  # This MUST match the transform you use to build the memory bank
  # It must include Resize() because the grid tiles are not square
  inference_transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)), # Crucial
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])
  

  #QUADRAT_DIR = ""

  quadrat_test_set = QuadratTilingDataset_Inference(
      data_dir=QUADRAT_DIR,
      grid_size=(3, 3),
      transform=inference_transform
  )
  
  if len(quadrat_test_set) == 0:
      print("Dataset is empty. Exiting.")
      return

  # 3. Instantiate the DataLoader
  # shuffle=False is critical for inference to keep tiles in order
  quadrat_loader = DataLoader(
      dataset=quadrat_test_set,
      batch_size=32, # Batch size of 32 *tiles*
      shuffle=False, 
      num_workers=4,
      pin_memory=True
  )


if __name__ == "__main__":
  main()
