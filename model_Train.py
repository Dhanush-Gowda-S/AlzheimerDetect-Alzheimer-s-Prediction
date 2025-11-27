import os
import numpy as np
import cv2
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =============================
# CONFIG
# =============================


datasetno=input("Enter dataset number:-")
modelno=input("enter model number:-")

DATASET_DIR = "./Dataset/TASK_"+datasetno
SAVE_DIR = "./saved_models"
SAVE_DIR_report ="./saved_models/model"+modelno+"_TASK_"+datasetno+" report"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_report, exist_ok=True)
IMG_SIZE = 224

# =============================
# Preprocessing Function
# =============================

def preprocess_gaussian_clahe_then_otsu(image):
    gray = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    normalized = cv2.divide(gray, blurred, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(normalized)

    _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    resized = cv2.resize(otsu, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb / 255.0


# =============================
# Dataset Loader
# =============================

def load_dataset():
    X, y = [], []

    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                
                # Determine label from folder name
                folder = os.path.basename(root).upper()
                
                if "AD" in folder:
                    label = 0
                elif "HC" in folder:
                    label = 1
                else:
                    continue  # ignore unexpected folders

                path = os.path.join(root, file)

                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = preprocess_gaussian_clahe_then_otsu(img)

                X.append(img)
                y.append(label)

    return np.array(X), np.array(y)


print("Loading dataset...")
X, y = load_dataset()
print("Loaded:", X.shape, y.shape)

# =============================
# Train / Test Split
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=42)

y_train_c = tf.keras.utils.to_categorical(y_train, 2)
y_val_c = tf.keras.utils.to_categorical(y_val, 2)
y_test_c = tf.keras.utils.to_categorical(y_test, 2)

# =============================
# Build MobileNetV2 Model
# =============================

def build_model():
    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False  # freeze for stability & speed
#if you want higher accuracy remove the hash of the loop but training takes more time 
#you can only increase upto 50 or it will be overfitted
    for layer in base.layers[-50:]:
        layer.trainable = True


    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation="softmax")(x)

    model = Model(base.input, output)
    model.compile(
        optimizer=Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


model = build_model()
model.summary()

# =============================
# Train the Model
# =============================

history = model.fit(
    X_train, y_train_c,
    validation_data=(X_val, y_val_c),
    epochs=15,
    batch_size=16,
    verbose=1
)

# =============================
# Save Model
# =============================
modelname="model"+modelno+"_TASK_"+datasetno+".h5"
model.save(os.path.join(SAVE_DIR, modelname))
print("Model saved at:", SAVE_DIR)

# =============================
# Evaluate
# =============================

loss, acc = model.evaluate(X_test, y_test_c)
print("Test Accuracy:", acc)

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# =============================
# Plot Training Curves
# =============================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()
plt.savefig(os.path.join(SAVE_DIR_report, "accuracy_plot.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss")
plt.legend()
plt.savefig(os.path.join(SAVE_DIR_report, "loss_plot.png"))
plt.close()

print("Training complete!")
