# train_cnn.py
import os
import cv2
import numpy as np
import pandas as pd
import json
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load dataset into DataFrame

DATA_DIR = "C:/Users/aksha/OneDrive/Desktop/Major/dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"

def create_dataframe(data_dir):
    data = []
    for label in ["benign", "malignant"]:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            continue
        for root, _, files in os.walk(label_dir):
            for file in files:
                if file.lower().endswith(".png"):
                    data.append({
                        "path": os.path.join(root, file),
                        "label": label
                    })
    return pd.DataFrame(data)

df = create_dataframe(DATA_DIR)
print(f"Total images found: {len(df)}")
print(df.head())

# Step 2: Train-test split

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Step 3: Image data generators

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='label',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='path',
    y_col='label',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Step 4: CNN model

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 5: Callbacks for best training

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Step 6: Train model (with timing)

start_time = time.time()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=200,  # high number; EarlyStopping will stop at best epoch
    callbacks=[early_stop, reduce_lr]
)

end_time = time.time()
training_time = end_time - start_time
print(f"⏱ CNN training time: {training_time:.2f} seconds")

# Step 7: Save best model

model.save("cnn_model.h5")
print("✅ CNN model saved as cnn_model.h5")

# Step 8: Evaluate on validation set

val_gen.reset()
y_pred_prob = model.predict(val_gen)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = val_df['label'].apply(lambda x: 1 if x=="malignant" else 0).values

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['benign','malignant']))

# Step 9: Save CNN metrics to metrics.json (with training time)

last_epoch_acc = history.history['val_accuracy'][-1]
last_epoch_loss = history.history['val_loss'][-1]

cnn_metrics_last_epoch = {
    "accuracy": float(last_epoch_acc),
    "loss": float(last_epoch_loss),
    "precision": float(precision_score(y_true, y_pred)),
    "recall": float(recall_score(y_true, y_pred)),
    "f1_score": float(f1_score(y_true, y_pred)),
    "training_time_sec": round(training_time, 2)
}

metrics_path = "metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)
else:
    all_metrics = {}

all_metrics["CNN"] = cnn_metrics_last_epoch

with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=4)

print("✅ CNN metrics (with training time) saved to metrics.json")
