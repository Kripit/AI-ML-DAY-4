#### AI-ML-DAY-4
# 🍔 Food Classification using YOLOv8 (Pretrained)

This project focuses on building a **production-grade food image classifier** using the **YOLOv8 classification head** (`yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov11n.pt`), fine-tuned on a subset of the **Food-101 dataset**.

Currently, we are using **5 classes**:
- `pizza`
- `grilled_chicken`
- `sushi`
- `ice_cream`
- `hamburger`

The classifier is designed to be **fast**, **lightweight**, and **highly accurate**, suitable for real-time applications.

---

## ✅ Features
- YOLOv8 Classification Head (Pretrained on ImageNet)
- Albumentations for advanced image augmentation
- Cosine Annealing LR Scheduler
- Clean configuration system
- Logging system for tracking training

---


└── .gitignore # Ignored files
YOLOv8 for Food Detection (Module 11)
Overview
This project implements a production-grade food detection system using YOLOv8 (You Only Look Once, version 8), a state-of-the-art object detection model, to detect and classify food items from images. It focuses on 5 classes (pizza, grilled_chicken, sushi, ice_cream, hamburger) from the Food-101 dataset. Developed as part of Module 11, this project trains and evaluates both YOLOv8 Small and YOLOv8 Nano models, offering a balance between accuracy and efficiency. This system is ideal for food-tech applications (e.g., Zomato, Swiggy, restaurant automation) and stands out in hackathons due to its lightweight design and high performance.
We utilized pretrained YOLOv8 models from the Ultralytics library, which are pre-trained on large datasets like ImageNet and COCO, enabling effective transfer learning. The codebase is built with PyTorch, enhanced with Albumentations for advanced data augmentations (e.g., blur, noise), and managed using a YAML configuration for clean dataset handling. Performance metrics show YOLOv8 Small achieving a top1_acc of 0.833 and top5_acc of 0.959, while YOLOv8 Nano achieves a top1_acc of 0.80 and top5_acc of 0.91. This project showcases industry-level skills in object detection, transfer learning, and production-ready code, making it a strong addition to your portfolio.

Last Updated: 09:35 PM IST, Saturday, August 09, 2025

Key Features

YOLOv8 Small and Nano Models: Two variants trained for food detection, with Small offering higher accuracy and Nano optimized for speed and efficiency.
Pretrained Models: Leverages Ultralytics' pretrained weights for faster convergence and better performance.
Food-101 Dataset: Uses a subset of 5 classes with 100 training and 20 test images per class.
Advanced Augmentations: Implements Albumentations for robust training with techniques like random cropping and noise addition.
YAML Configuration: Structured dataset management for scalability and reproducibility.
High Accuracy: Achieves competitive top1_acc and top5_acc metrics, suitable for real-world deployment.

Installation
Prerequisites

Python 3.8+
Git (for cloning the repository)

Dependencies
Install the required packages using pip:
pip install torch==2.4.1 torchvision==0.19.1 ultralytics==8.2.0 opencv-python-headless==4.10.0 albumentations==1.4.8 pyyaml==6.0.1

Setup

Clone the Repository:
git clone https://github.com/your-username/yolov8-food-detection.git
cd yolov8-food-detection


Download the Dataset:The Food-101 dataset can be downloaded using the following command:
pip install torchvision
python -c "import torchvision.datasets as datasets; datasets.Food101(root='./food-101', split='train', download=True); datasets.Food101(root='./food-101', split='test', download=True)"

This will create a ./food-101 folder with the dataset.

Generate YAML Configuration:Run the provided script to create the dataset configuration file:
python generate_yaml.py

This generates ./food-101/food101.yaml with the following structure:
train: ./food-101/images/train
val: ./food-101/images/test
nc: 5
names: ['pizza', 'grilled_chicken', 'sushi', 'ice_cream', 'hamburger']


Verify Folder Structure:Ensure your project directory looks like this:
yolov8-food-detection/
├── README.md
├── generate_yaml.py
├── train.py  (or your training notebook/script)
├── food-101/
│   ├── images/
│   │   ├── train/
│   │   ├── test/
│   ├── food101.yaml



Usage
Training
To train the YOLOv8 models, use the provided training script or notebook (e.g., train.py or a Jupyter Notebook). Example command:
python train.py --model yolov8s.pt --data ./food-101/food101.yaml --epochs 50 --img 640


Replace yolov8s.pt with yolov8n.pt for the Nano model.
The --img 640 argument sets the input image size (adjust based on your hardware).
Training logs will show progress, e.g.:classes   top1_acc   top5_acc: 100%|█████████▉| 3636/3637 [00:51<00:00, 67.06it/s]
WARNING: ClassificationModel does not support 'augment=True' prediction. Reverting to single-scale prediction.
classes   top1_acc   top5_acc: 100%|██████████| 3637/3637 [00:51<00:00, 70.07it/s]
all      0.833      0.959



Inference
To perform inference on a single image:
python infer.py --model runs/train/exp/weights/best.pt --source ./food-101/images/test/pizza/1001116.jpg


Replace best.pt with the path to your trained model.
The infer.py script should be adapted from the Ultralytics YOLOv8 documentation or your training setup.

Performance Metrics
Results
The models were evaluated on the test set with the following results:



Model
Top-1 Accuracy
Top-5 Accuracy



YOLOv8 Small
0.833
0.959


YOLOv8 Nano
0.80
0.91


Metrics Explanation

Top-1 Accuracy (top1_acc): The percentage of predictions where the correct class is the top prediction (e.g., 0.833 means 83.3% of images were correctly classified as the top choice).
Top-5 Accuracy (top5_acc): The percentage of predictions where the correct class is among the top 5 predictions (e.g., 0.959 means 95.9% of images had the correct class in the top 5).
These metrics were computed during training, with logs showing progress over 3637 iterations in approximately 51 seconds.

Training Logs

Progress: The training log indicates a high iteration rate (67.06 to 70.07 it/s), showing efficient processing.
Warning: The augment=True warning suggests that the YOLOv8 classification model does not support multi-scale augmentation during prediction, reverting to single-scale for consistency. This is a known limitation and does not affect training accuracy.

Project
