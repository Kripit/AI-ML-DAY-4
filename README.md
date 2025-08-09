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

## 📂 Project Structure

├── foof/
│ ├── food-101/ # Dataset folder
│ └── food101.yaml # Dataset config
├── runs/train/ # Training logs & checkpoints
├── main.py # Training script
├── generate_yaml.py # Utility to generate dataset YAML
├── yolo8v.py # YOLO training logic
├── best_food_model.pth # Saved best model weights
├── yolov8n-cls.pt # Pretrained small model
├── yolov8s-cls.pt # Pretrained medium model
├── yolov11n.pt # Pretrained large model
└── .gitignore # Ignored files
