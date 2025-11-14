import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import os


# os.chdir("/scratch/ezq9qu/images_max_side_800")
# DATA_DIR = os.getcwd()


# RESIZE_SIZE = 256
# IMG_SIZE = 224 
# BATCH_SIZE = 32
# NUM_CLASSES = 7806 # As per the challenge overview
# NUM_WORKERS = os.cpu_count() # Use all available CPU cores for loading


class SinglePlantDataLoader:
    """
    A class to create training, validation, and test DataLoaders
    from a single directory containing subfolders for each class.
    Splits the data based on 80/10/10 Train/Test/Val
    """
    def __init__(self, data_dir, resize_size, img_size, batch_size, num_workers, train_split=0.8, val_split=0.1, test_split=0.1):
        """
        Initializes the SinglePlantDataLoader.

        Args:
            data_dir (str): Path to the root directory containing class subfolders.
            resize_size (int): The size to which the smaller edge of the image will be resized.
            img_size (int): The size of the center crop or random resized crop.
            batch_size (int): The number of images per batch.
            num_workers (int): How many subprocesses to use for data loading.
            train_split (float): The proportion of the dataset to use for training.
            val_split (float): The proportion of the dataset to use for validation.
            test_split (float): The proportion of the dataset to use for testing.
        """
        self.data_dir = data_dir
        self.resize_size = resize_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.classes = None

        self._load_and_split_data()


    def _load_and_split_data(self):
        """
        Loads the dataset, splits it, and creates the DataLoaders.
        """
        # Define the standard PyTorch transforms for ImageNet-pretrained models
        val_test_transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"Loading data from: {self.data_dir}")
        try:
            # Create the full dataset
            full_dataset = datasets.ImageFolder(
                root=self.data_dir,
                transform=val_test_transform # Use validation transform for initial loading
            )

            # Check if dataset is empty
            if not full_dataset.samples:
                print(f"ERROR: No images found in {self.data_dir}.")
                return

            self.classes = full_dataset.classes


        except Exception as e:
            print(f"An unexpected error occurred loading data: {e}")
            return

        # --- Split dataset into train, val, and test ---
        dataset_size = len(full_dataset)
        train_size = int(self.train_split * dataset_size)
        val_size = int(self.val_split * dataset_size)
        test_size = dataset_size - train_size - val_size # Ensure all data is used

        print(f"\nSplitting dataset")

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size]
        )

        # Apply the training transform to the training split
        train_dataset.dataset.transform = train_transform




        print(f"\nFound {len(full_dataset)} total images belonging to {len(self.classes)} classes.")

        # Create the DataLoaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,      # Shuffle the training data
            num_workers=self.num_workers,
            pin_memory=True    # Speeds up CPU-to-GPU data transfer
        )

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,     
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,     
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_dataloaders(self):
        """
        Returns the training, validation, and test DataLoaders.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader

    def get_classes(self):
        """
        Returns the list of class names found in the dataset.

        Returns:
            list: A list of class names.
        """
        return self.classes
        
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
            for i in range(self.num_tiles): 
                self.samples.append((img_path, i))
                
        

    def __len__(self):
        """Returns the total number of tiles."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
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
            
            # Return the tile and its source path
            # The path is crucial for grouping predictions later
            return tile, img_path

        except Exception as e:
            print(f"Error loading tile for image {img_path} at index {idx}: {e}")

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

    # Create an instance of the SinglePlantDataLoader
    data_splitter = SinglePlantDataLoader(
        data_dir=DATA_DIR,
        resize_size=RESIZE_SIZE,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Get the dataloaders
    train_loader, val_loader, test_loader = data_splitter.get_dataloaders()

    if train_loader is None or val_loader is None or test_loader is None:
        return

    print("\n--- Dataloader Pipelines Created ---")

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
    #IMG_SIZE = 244

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
