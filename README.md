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



These metrics were computed during training, with logs showing progress over 3637 iterations in approximately 51 seconds.
