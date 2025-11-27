
# 🧠 AlzheimerDetect —  Alzheimer’s Prediction Made by Dhanush Gowda s
### Handwriting Image Classification Using MobileNetV2 + Grad-CAM  
### Flask Web Application with Multi-Model Selection & Admin Panel

---

## 📌 Overview

**AlzheimerDetect** is an advanced AI-based web application designed to detect **Alzheimer’s Disease** from **handwriting images**.  
It uses:

- MobileNetV2 (TensorFlow 2.17)
- Explainable AI (Grad-CAM)
- Multi-model selection (6 trained models)
- User authentication system
- Admin dashboard
- Modern Bootstrap UI

The system processes a handwriting image, runs it through a selected model, generates a prediction, and displays a **heatmap explanation**.

---

# 📂 Project Structure

```
AlzheimerDetect/
│ app.py
│ model_Train.py
│ README.md
│ requirements.txt
│ alzheimer.db
│
├── saved_models/
│   ├── model1_TASK_02.h5
│   ├── model2_TASK_03.h5
│   ├── model3_TASK_04.h5
│   ├── model4_TASK_05.h5
|│
├── Dataset/
│   ├── TASK_02/
│   │   ├── AD/
│   │   └── HC/
│   ├── TASK_03/
│   ├── TASK_04/
│   ├── TASK_05/
│   ├── TASK_21/
│   └── TASK_24/
│
├── static/
│   ├── uploads/            # Prediction images + Grad-CAM
│   ├── images/
│   │   └── brain-scan.png
│   ├── css/
│   └── js/
│
└── templates/
    ├── *.html (User pages)
    └── admin/
         ├── login.html
         ├── dashboard.html
         ├── users.html
         ├── predictions.html
         └── prediction_detail.html
```

---

# ⚙️ Installation Guide

## 1️⃣ Install Python  
Recommended version:

```
Python 3.12.6
```

---

## 2️⃣ Install dependencies

Run:

```
pip install -r requirements.txt
```

If TensorFlow fails on Windows:

```
pip install tensorflow==2.17.0
pip install keras==3.3.3
pip install opencv-python
pip install pillow
pip install flask
```

---

## 3️⃣ Start the Flask server

```
python app.py
```

Server runs at:

```
http://127.0.0.1:5001
```

---

# 🔑 Login Credentials

## 👤 User Login
Register normally.

## 🔐 Admin Login

Visit:

```
http://127.0.0.1:5001/admin
```

Default credentials:

```
Username: admin
Password: admin123
```

---

# 🧪 Supported AI Models (6-Model Architecture)

### Model files:
| Model Key | File | Task |
|----------|--------------------------|--------|
| model1 | model1_TASK_02.h5 | TASK_02 |
| model2 | model2_TASK_03.h5 | TASK_03 |
| model3 | model3_TASK_04.h5 | TASK_04 |
| model4 | model4_TASK_05.h5 | TASK_05 |


### Selection modes:
✔ Manual (dropdown)  
✔ Auto detection (if filename contains TASK_XX)

---

# 🌟 Features

### ✔ Handwriting Image Upload  
Accepts: JPG, JPEG, PNG

### ✔ Deep Learning Prediction  
Outputs:
- Alzheimer’s Disease (AD)
- Healthy Control (HC)
- Confidence score

### ✔ Grad-CAM Explanation  
Highlights image regions influencing the model.

### ✔ User Features  
- Register/Login  
- Prediction history  
- Model selection  
- View heatmaps  

### ✔ Admin Panel  
Admin can:
- View user list  
- View all predictions  
- View prediction details  
- Delete users or predictions  

---

# 📸 Grad-CAM Example

```
Original Image     →     Grad-CAM Heatmap
```

Used to visualize important handwriting regions detected by MobileNetV2.

---

# 🧠 Model Training (MobileNetV2)

To train:

```
python model_Train.py
```

Training script includes:
- Preprocessing  
- MobileNetV2 base  
- GlobalAveragePooling  
- Hyperparameter tuning  
- Automatic saving in `/saved_models`  

---

# 🚀 Deployment Options

### ✔ Local Windows (recommended)
### ✔ Docker  
### ✔ Gunicorn + Nginx  
### ✔ Railway.app / Render.com  
### ✔ Convert to EXE (PyInstaller)

If you want a deployment guide, ask:

**“Generate deployment guide”**

---

# 🛠 Troubleshooting

### ❌ TensorFlow errors
Install:

```
pip install tensorflow==2.17.0 --upgrade
```

### ❌ Keras/TensorFlow mismatch
```
pip install keras==3.3.3
```

### ❌ Database issues
Delete `alzheimer.db` and restart app.

### ❌ Grad-CAM black image
Use correct layer:

```
layer_name="Conv_1"
```

---

# ❤️ Credits

Developed by **Dhanush Gowda S**  
Powered by **Flask, TensorFlow, MobileNetV2, Bootstrap, Grad-CAM**

---

** If you need the exe file of this project then install pyinstaller and run "pyinstaller app.spec"**
you will get the exe file in the Alzheimer file inside the dist
