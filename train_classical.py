import os
import cv2
import numpy as np
import pandas as pd
import mahotas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import time

# Step 1: Load dataset into DataFrame

DATA_DIR = "C:/Users/aksha/OneDrive/Desktop/Major/dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

def create_dataframe(DATA_DIR):
    data = []
    for label in ["benign", "malignant"]:
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.exists(label_dir):
            continue
        # Walk all subfolders
        for root, _, files in os.walk(label_dir):
            for file in files:
                if file.lower().endswith(".png"):
                    data.append({
                        "image_path": os.path.join(root, file),
                        "label": label
                    })
    return pd.DataFrame(data)

df = create_dataframe(DATA_DIR)
print(f"Total images found: {len(df)}")
print(df.head())

# Step 2: Feature extraction using mahotas Haralick

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    glcm = mahotas.features.texture.haralick(img)
    return np.mean(glcm, axis=0)

# Extract features for all images
feature_list = []
for idx, row in df.iterrows():
    features = extract_features(row["image_path"])
    feature_list.append(features)

X = np.array(feature_list)
y = np.array([1 if label == "malignant" else 0 for label in df["label"]])

# Step 3: Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Train classical ML models

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear', probability=True),
    "NaiveBayes": GaussianNB()
}

metrics = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    start_time = time.time()   
    model.fit(X_train, y_train)
    end_time = time.time()     
    
    training_time = end_time - start_time   
    print(f"{name} training time: {training_time:.4f} seconds")  
    
    y_pred = model.predict(X_test)
    metrics[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "training_time_sec": round(training_time, 4)
    }

# Step 5: Save metrics for Streamlit

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nTraining complete. Metrics with training time saved to metrics.json")
print(metrics)