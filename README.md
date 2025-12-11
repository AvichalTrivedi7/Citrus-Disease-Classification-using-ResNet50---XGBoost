# 🍊 Citrus Disease Classification

### **ResNet50 + XGBoost Hybrid Model (Fresh, Blackspot, Canker, Grenning)**

A deep learning + machine learning hybrid pipeline for citrus fruit disease detection.

---

## 📌 **Overview**

This project implements a **hybrid architecture** combining:

* **ResNet50 (Transfer Learning)** → Feature extraction
* **Dense(128) Head** → Learn citrus-specific representations
* **XGBoost Classifier** → Final disease classification

The model predicts **percentage probabilities** for:

* **Blackspot**
* **Canker**
* **Greening (Huanglongbing)**
* **Fresh / Healthy**

And can classify **any real-life citrus image** (even phone photos).

---

## 📂 **Project Structure**

```
Citrus-Disease-Classification/
│
├── run_resnet_xgb.py           # Main training + feature extractor + XGBoost pipeline
├── predict_demo.py             # Single-image disease prediction script
├── requirements.txt            # Required Python packages
├── README.md                   # This file
│
├── final_runs_resnet/          # (Empty now) Download trained models & place here
│   ├── cnn_resnet_ep50.h5
│   ├── featmod_resnet_ep50.h5
│   ├── xgb_resnet_ep50.joblib
│   └── ... (other models)
│
└── Orange Dataset/             # (Excluded from Git) Full dataset of images
```

---

## 📥 **Download Trained Models (Required)**

Because GitHub cannot store files over 100MB, all trained models are hosted on Google Drive.

### 👉 **Download here:**

🔗 **[Google Drive Link](https://drive.google.com/drive/folders/1C8eT3iwqfmHIHHkHYK_jOcgbHVjd0qlr?usp=sharing)**

After downloading:

### **Extract the folder into:**

```
final_runs_resnet/
```

You should see files like:

```
cnn_resnet_ep5.h5
cnn_resnet_ep10.h5
cnn_resnet_ep20.h5
cnn_resnet_ep50.h5
featmod_resnet_ep50.h5
xgb_resnet_ep50.joblib
results_summary.csv
confusion_resnet_ep50.png
```

---

## ⚙️ **Installation**

### 1️⃣ Create a virtual environment

```
python -m venv venv
```

### 2️⃣ Activate it

```
venv\Scripts\activate
```

### 3️⃣ Install required packages

```
pip install -r requirements.txt
```

---

## 🚀 **How to Train the Model**

If you want to retrain:

```
python run_resnet_xgb.py
```

The script automatically:

✔ Performs a **70/30 stratified split**
✔ Runs experiments for **5, 10, 20, 50 epochs**
✔ Saves:

* Trained CNN head
* Feature extractor
* XGBoost classifier
* Confusion matrices
* Summary CSV

---

## 🔍 **How to Predict on a Single Image**

Edit `predict_demo.py`:

```python
from run_resnet_xgb import predict_single_image

image_path = r"path_to_image.jpg"
result = predict_single_image(image_path, epochs_of_model=50)
print(result)
```

Run:

```
python predict_demo.py
```

Example output:

```
{
  "blackspot": 2.1,
  "canker": 88.4,
  "fresh": 1.7,
  "grenning": 7.8
}
```

---

## 📊 **Confusion Matrices & Performance**

The results for all epoch experiments are saved in:

```
final_runs_resnet/results_summary.csv
```

And individual confusion matrices:

```
final_runs_resnet/confusion_resnet_ep*.png
```

---

## **Summarized Usage Instructions**

1. Clone the GitHub repo
2. Create & activate a virtual environment
3. Install dependencies
4. Download trained models from Google Drive
5. Place models inside `final_runs_resnet/`
6. Run `predict_demo.py` to test real-life images

Your system is now fully functional.

---

## 📝 **Acknowledgments**

This project was built for testing how well simple **image analysis models** can perform, showcasing hybrid deep learning and machine learning techniques for real-world agriculture and potential **medical applications** (v.imp) in similar fashion.

