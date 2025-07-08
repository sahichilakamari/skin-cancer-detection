import os
import pandas as pd
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# --- CONFIG ---
IMAGE_SIZE = 128
EPOCHS = 25  # Increased epochs
BATCH_SIZE = 32
CLASS_WEIGHT = {0: 1.0, 1: 5.0}  # Higher weight for malignant

def load_data():
    df = pd.read_csv("HAM10000_metadata.csv")
    df['image_path'] = df['image_id'].apply(lambda x: f"HAM10000_images/{x}.jpg")
    
    malignant_dx = ['mel', 'bcc', 'akiec']
    df['binary_label'] = df['dx'].apply(lambda x: 1 if x in malignant_dx else 0)
    
    # Calculate and display class distribution
    malignant_count = df['binary_label'].sum()
    benign_count = len(df) - malignant_count
    print(f"Class distribution - Benign: {benign_count}, Malignant: {malignant_count}")
    
    os.makedirs("model", exist_ok=True)
    with open("model/labels_map.json", "w") as f:
        json.dump({"0": "benign", "1": "malignant"}, f)
    
    return df

def create_generators():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect'
    )

def image_generator(image_paths, labels, batch_size=32, augment=False):
    datagen = create_generators() if augment else None
    
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            batch_images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                    img = np.array(img) / 255.0
                    if augment:
                        img = datagen.random_transform(img)
                    batch_images.append(img)
                except:
                    batch_images.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)))
            
            yield np.array(batch_images), np.array(batch_labels)

def build_model():
    base_model = MobileNetV2(
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        weights='imagenet'
    )
    
    # Fine-tune last 50 layers
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),  # Focal loss
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall')  # Focus on recall
        ]
    )
    return model

def main():
    start = time.time()
    
    print("📦 Loading data...")
    df = load_data()
    
    train_paths, test_paths, y_train, y_test = train_test_split(
        df['image_path'].values,
        df['binary_label'].values,
        test_size=0.2,
        stratify=df['binary_label'],
        random_state=42
    )
    
    train_gen = image_generator(train_paths, y_train, BATCH_SIZE, augment=True)
    test_gen = image_generator(test_paths, y_test, BATCH_SIZE)
    
    train_steps = len(train_paths) // BATCH_SIZE
    test_steps = len(test_paths) // BATCH_SIZE
    
    print("🏋️ Training model...")
    model = build_model()
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=test_gen,
        validation_steps=test_steps,
        epochs=EPOCHS,
        class_weight=CLASS_WEIGHT,
        callbacks=[
            EarlyStopping(monitor='val_recall', patience=5, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_recall', factor=0.5, patience=2, mode='max')
        ],
        verbose=1
    )
    
    model.save("model/skinsafe_optimized_model.h5")
    
    print("📊 Evaluating model...")
    test_gen = image_generator(test_paths, y_test, BATCH_SIZE)
    y_pred = (model.predict(test_gen, steps=test_steps) > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test[:len(y_pred)], y_pred, target_names=['Benign', 'Malignant']))
    print("ROC AUC:", roc_auc_score(y_test[:len(y_pred)], y_pred))
    
    # Save confusion matrix
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test[:len(y_pred)], 
        y_pred, 
        display_labels=['Benign', 'Malignant'], 
        cmap='Blues'
    )
    plt.title("Confusion Matrix")
    plt.savefig("model/confusion_matrix.png")
    plt.close()
    
    # Save precision-recall curve
    probs = model.predict(test_gen, steps=test_steps)
    precision, recall, thresholds = precision_recall_curve(y_test[:len(probs)], probs)
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig("model/precision_recall_curve.png")
    plt.close()
    
    print(f"\n⏱️ Total training time: {(time.time() - start)/60:.2f} minutes")

if __name__ == "__main__":
    main()