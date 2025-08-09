# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
import cv2
import numpy as np
import logging
import os
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Optional, Tuple

# logging setup  to track executions and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers = [
                        logging.StreamHandler(), #to show logs in console 
                        logging.FileHandler('yolo_training.log') # save logs to a file
                    ] )

logger = logging.getLogger(__name__)

# configurations class for project settings 
class Config:
    def __init__(self):
        self.data_dir = "./foof/food-101/images"  # Updated to match your structure
        self.classes = ['pizza','grilled_salmon','sushi','ice_cream','hamburger']
        self.max_images_per_class = 100
        self.test_images_per_class = 20
        self.epochs = 30
        self.batch_size = 5 # batch size optimized for gpu 
        self.img_size = 224 # image size for yolo v8 
        self.model_path_nano = "yolov8n_food_classifier.pt" #nano model save path 
        self.model_path_small = "yolov8s_food_classifier.pt" #small model save path 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 0.001
        logging.info(f"Using device : {self.device}")
       
# custom dataset class to load the food-101 dataset 
class FoodDataset(Dataset):
    def __init__(self, image_paths: List[str] , labels: List[int] , transform: Optional[A.Compose]=None):
        self.image_paths = image_paths
        self.labels = labels  # list of labels from 0 to 4 obviously but comment keep us in touch so it is important to write comments
        self.transform = transform #albumentations transform
        logger.info(f"Initialized dataset with {len(image_paths)} images ")
    
    
    def __len__(self) -> int:
        return len(self.image_paths)  #count of total images 
    
    
    def __getitem__(self , idx: int) -> Tuple[Optional[torch.Tensor], Optional[int]]: # Returns a single data sample (image tensor and label) for the given index
        try: 
            image_path = self.image_paths[idx]
            image = cv2.imread(image_path) # load the image
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
            
            if self.transform:
                augmented = self.transform(image=image)
                image =  augmented['image'] # Apply image transformations (e.g., resize, normalize, flip) if defined
            label = self.labels[idx] # fetching the label ( class label )
            return image,label 
        
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None , None 
        
        
# Advanced Augmentations using Albumentations
transform = A.Compose([
    A.Resize(224,224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit = 15 , p=0.5)  ,# randomly rotate by 15 degrees
    A.RandomBrightnessContrast(brightness_limit = 0.2 , contrast_limit=0.2 , p =0.5),
    A.GaussNoise(sigma_limit=(0.1, 2.0), p=0.3),  # Fixed to use sigma_limit
    A.MotionBlur(blur_limit=7, p =0.3),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.2225]),  # imagenet normalization 
    ToTensorV2()
])

# food-101 dataset load and subset function
def load_food101_subset(config: Config) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dict[int, str]]]:
    try:
        logger.info("Loading food-101 dataset")
        
        # Check if data directory exists
        if not os.path.exists(config.data_dir):
            raise FileNotFoundError(f"Dataset directory {config.data_dir} not found. Ensure dataset is downloaded and path is correct.")

        # Manually load images from class folders under config.data_dir
        train_image_paths = []
        train_labels = []
        test_image_paths = []
        test_labels = []
        
        # Split into train and test based on your max_images_per_class and test_images_per_class
        for class_name in config.classes:
            class_dir = os.path.join(config.data_dir, class_name)
            if not os.path.exists(class_dir):
                raise ValueError(f"Class directory {class_dir} not found")
            
            # Get all image files in the class folder
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  # Case-insensitive
            if not image_files:
                raise ValueError(f"No images found in {class_dir}")
            
            # Sort files to ensure consistent splitting
            image_files.sort()
            
            # Split into train and test (simple split based on counts)
            total_images = len(image_files)
            train_count = min(config.max_images_per_class, total_images)
            test_count = min(config.test_images_per_class, total_images - train_count) if total_images > train_count else 0
            
            # Assign labels
            class_idx = config.classes.index(class_name)
            
            # Train images
            for img in image_files[:train_count]:
                image_path = os.path.join(class_dir, img)
                train_image_paths.append(image_path)
                train_labels.append(class_idx)
            
            # Test images
            for img in image_files[train_count:train_count + test_count]:
                image_path = os.path.join(class_dir, img)
                test_image_paths.append(image_path)
                test_labels.append(class_idx)
        
        train_subset = FoodDataset(train_image_paths, train_labels, transform)
        test_subset = FoodDataset(test_image_paths, test_labels, transform)
        
        class_names = {i: cls for i, cls in enumerate(config.classes)}
        logger.info(f"Loaded {len(train_image_paths)} train and {len(test_image_paths)} test images")
        return train_subset, test_subset, class_names
    
    except Exception as e:
        logger.error(f"error loading Food-101 dataset: {str(e)}")
        return None, None, None

def validate_model(model: YOLO , dataloader: DataLoader , class_names:Dict[int , str], config: Config) -> float:
    try:
        logger.info("Starting Validating mogger, TIME TO SEE THE FIRE SHI I BUILT LESS GO BUD ")
        model.eval()  # model is in evaluation mode
        correct = 0 
        total = 0
        with torch.no_grad(): #disable graadient 
            for images, labels in dataloader:
                if images is None or labels is None:
                    continue
                images, labels = images.to(config.device), labels.to(config.device)
                results = model(images) # YOLOv8 predictions
                
                predicted = torch.tensor([r.probs.top1 for r in results])
    # top predictions images
                total+= labels.size(0)
                correct+= (predicted == labels).sum().item()
        accuracy = 100 * correct / total if total>0 else 0
        logger.info(f"Validations accuracy : {accuracy:.2f}%")
        return accuracy
    
    except Exception as e:
        logger.error(f"Error in validation : {str(e)}")
        return 0.0

# training function for yolov8 with advance features:
def train_yolo_model(config: Config , train_dataset: Dataset , test_dataset: Dataset) -> Tuple[Optional[YOLO], Optional[YOLO]]:
    try:
        logger.info("Setting up YOLOv8 models (nano and samll)")
        
        model_nano = YOLO('yolov8n-cls.pt')
        model_small = YOLO('yolov8s-cls.pt')

        # Making DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )  
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
        # Train nano model 
        logger.info("Starting YOLOv8 nano training")
        model_nano.train(
            imgsz=config.img_size,
            epochs=config.epochs,
            batch=config.batch_size,
            device=config.device,
            patience=5,
            augment=True,
            save=True,
            project="runs/train",
            name="food_classifier_nano",
            optimizer='SGD',
            lr0=config.lr,
            cos_lr=True, # Cosine Annealing LR 
            data=config.data_dir  # Added to point to custom dataset directory
        )
        logger.info("Starting YOLOv8 small training")  # Corrected "YOLOv9" to "YOLOv8"
        model_small.train(
            imgsz=config.img_size,
            epochs=config.epochs,
            batch=config.batch_size,
            device=config.device,
            patience=5,
            augment=True,
            save=True,
            project="runs/train",
            name="food_classifier_small",
            optimizer='SGD',
            lr0=config.lr,
            cos_lr=True, # Cosine Annealing LR 
            data=config.data_dir  # Added to point to custom dataset directory
        )
    
        # Validate both models
        nano_accuracy = validate_model(model_nano, test_loader, config.classes, config)
        small_accuracy = validate_model(model_small, test_loader, config.classes, config)
    
        logger.info(f"Nano model validation accuracy: {nano_accuracy:.2f}%")
        logger.info(f"Small model validation accuracy: {small_accuracy:.2f}%")
    
        model_nano.save(config.model_path_nano)
        model_small.save(config.model_path_small)
    
        logger.info(f"Nano model saved at {config.model_path_nano}")
        logger.info(f"Small model saved at {config.model_path_small}")  # Corrected typo "Nano" to "Small")
        
        return model_nano, model_small
    except Exception as e:
        logger.error(f"Error training YOLOv8: {str(e)}")
        return None, None

# Ensemble prediction function
def ensemble_predict(models: List[YOLO], image: torch.Tensor, class_names: Dict[int, str], config: Config) -> Optional[str]:
    try:
        logger.info("Starting ensemble prediction")
        probs = []
        for model in models:
            model.eval()
            with torch.no_grad():
                results = model(image)
                probs.append(results[0].probs.data.cpu().numpy())  # Probabilities for all classes
        # Average probabilities
        avg_probs = np.mean(probs, axis=0)
        predicted_idx = np.argmax(avg_probs)  # Top prediction index
        confidence = avg_probs[predicted_idx]
        logger.info(f"Ensemble prediction: {class_names[predicted_idx]} (Confidence: {confidence:.2f})")
        return class_names[predicted_idx]
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {str(e)}")
        return None

# Inference function for single image
def classify_image(models: List[YOLO], image_path: str, transform: A.Compose, class_names: Dict[int, str], config: Config) -> Optional[str]:
    try:
        logger.info(f"Classifying image: {image_path}")
        image = cv2.imread(image_path)  # Image load karo
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR se RGB
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(config.device)  # Transform aur batch dimension
        prediction = ensemble_predict(models, image_tensor, class_names, config)
        return prediction
    except Exception as e:
        logger.error(f"Error classifying image: {str(e)}")
        return None

# Main execution
if __name__ == "__main__":
    # Initialize configuration
    config = Config()

    # Load dataset
    train_dataset, test_dataset, class_names = load_food101_subset(config)
    if train_dataset is None or test_dataset is None:
        logger.error("Failed to load dataset")
        exit()

    # Train and validate models
    model_nano, model_small = train_yolo_model(config, train_dataset, test_dataset)
    if model_nano is None or model_small is None:
        logger.error("Failed to train models")
        exit()

    # Test inference with ensemble
    test_image = "./foof/food-101/images/pizza/1001116.jpg"  # Updated path
    prediction = classify_image([model_nano, model_small], test_image, transform, class_names, config)
    print(f"Predicted class: {prediction}")