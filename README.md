# FEKD: A Lightweight Camouflaged Object Detection Model Based on Frequency-Domain Enhanced Distillation
### 基于频域增强蒸馏的轻量级伪装目标检测模型
This is the official PyTorch implementation of our paper:
**A lightweight camouflage object detection model based on frequency-domain enhanced distillation（FEKD）**
<img width="933" height="531" alt="image" src="https://github.com/user-attachments/assets/a89a7dd7-11fe-4ccc-89d4-59090fbc259a" />
## 📌 Abstract
Camouflaged object detection aims to precisely segment camouflaged targets hidden in complex backgrounds. The core challenge lies in the high similarity of texture and edge features between the target and the background, which weakens the discriminative representation of the target and makes the foreground response exhibit low contrast and weak saliency, thereby increasing the difficulty of distinguishing the foreground from the background. Existing methods mostly focus on spatial-domain feature modeling and adaptation, while to some extent neglecting the potential advantages of frequency-domain information in alleviating boundary blurring and enhancing the representation of high-frequency details. As a result, the modeling of high-frequency structures and fine details is insufficient, leading to inaccurate boundary localization, inadequate detail recovery, and response diffusion in target regions. To address this issue, a lightweight camouflaged object detection model based on frequency-domain enhanced distillation is proposed. By jointly constraining the features of the teacher and student networks in both the spatial domain and the frequency-domain amplitude spectrum, effective transfer of high-frequency details and global structures is achieved. Meanwhile, a gated frequency-domain enhancement fusion mec[hanism is introduced to enhance the student feature representation, and channel semantic inconsistency in cross-architecture distillation is alleviated by a multi-scale adaptive channel alignment module. Experimental results demonstrate that this method achieves precise distillation of frequency-domain knowledge with 49.7 M parameters and performs favorably on datasets such as COD10K and CAMO. On CAMO, the average E-measure reaches 90.30%, the structural similarity reaches 84.7%, and the mean absolute error (MAE) is reduced to 5%, achieving an effective balance between detection accuracy and inference efficiency.
## 📌 Usage
Datasets and datasets are suggested to be downloaded from official pages. 
## 📌 Environment Setup
```bash
# Pytorch==2.5.1+CUDA12.4 (or 2.0.1+CUDA11.8)
conda create -n lckd python=3.10 -y && conda activate lckd
pip install -r requirements.txt
## 📌 Quantitative Results
<img width="774" height="499" alt="image" src="https://github.com/user-attachments/assets/d2e64b28-51f2-46ba-aae9-32b52b787ea4" />
