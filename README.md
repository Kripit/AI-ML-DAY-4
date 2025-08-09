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






#### Performance Metrics

# Results

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



### Overview

This project implements a production-grade food detection system using YOLOv8 (You Only Look Once, version 8), a state-of-the-art object detection model, to detect and classify food items from images. It focuses on 5 classes (pizza, grilled_chicken, sushi, ice_cream, hamburger) from the Food-101 dataset. Developed as part of Module 11, this project trains and evaluates both YOLOv8 Small and YOLOv8 Nano models, offering a balance between accuracy and efficiency. This system is ideal for food-tech applications (e.g., Zomato, Swiggy, restaurant automation) and stands out in hackathons due to its lightweight design and high performance.

We utilized pretrained YOLOv8 models from the Ultralytics library, which are pre-trained on large datasets like ImageNet and COCO, enabling effective transfer learning. The codebase is built with PyTorch, enhanced with Albumentations for advanced data augmentations (e.g., blur, noise), and managed using a YAML configuration for clean dataset handling. Performance metrics show YOLOv8 Small achieving a top1_acc of 0.833 and top5_acc of 0.959, while YOLOv8 Nano achieves a top1_acc of 0.80 and top5_acc of 0.91. This project showcases industry-level skills in object detection, transfer learning, and production-ready code, making it a strong addition to your portfolio.






### Key Features





. YOLOv8 Small and Nano Models: Two variants trained for food detection, with Small offering higher accuracy and Nano optimized for speed and efficiency.



. Pretrained Models: Leverages Ultralytics' pretrained weights for faster convergence and better performance.



. Food-101 Dataset: Uses a subset of 5 classes with 100 training and 20 test images per class.



. Advanced Augmentations: Implements Albumentations for robust training with techniques like random cropping and noise addition.



. YAML Configuration: Structured dataset management for scalability and reproducibility.



. High Accuracy: Achieves competitive top1_acc and top5_acc metrics, suitable for real-world deployment.
