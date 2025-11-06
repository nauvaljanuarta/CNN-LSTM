# lstm.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

print("🚀 MEMUAT DATA UCI HAR DATASET...")

# 1. LOAD DATA
def load_har_data():
    """Load dataset UCI HAR"""
    try:
        # Load training data
        X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
        y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', header=None)
        
        # Load testing data  
        X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
        y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', header=None)
        
        # Load activity labels
        activity_labels = pd.read_csv('UCI HAR Dataset/activity_labels.txt', 
                                     delim_whitespace=True, header=None)
        
        print("✅ Data loaded successfully!")
        return X_train, y_train, X_test, y_test, activity_labels
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None, None

# Load data
X_train, y_train, X_test, y_test, activity_labels = load_har_data()

if X_train is None:
    exit()

# 2. CEK DATA
print("\n📊 DATA EXPLORATION:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

print("\n🎯 ACTIVITY LABELS:")
for i, label in activity_labels.iterrows():
    print(f"  {label[0]}: {label[1]}")

print("\n📈 LABEL DISTRIBUTION:")
print("Training:")
print(y_train[0].value_counts().sort_index())
print("\nTesting:")
print(y_test[0].value_counts().sort_index())

# 3. PREPROCESSING
print("\n🔧 PREPROCESSING DATA...")

# Normalisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape untuk LSTM (samples, timesteps, features)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Encode labels (1-6 menjadi 0-5)
y_train_encoded = y_train[0] - 1
y_test_encoded = y_test[0] - 1

print(f"✅ Data reshaped: {X_train_reshaped.shape}")

# 4. SPLIT DATA SESUAI SPEC
print("\n📊 SPLITTING DATA...")

# Gabungkan semua data
X_all = np.vstack([X_train_reshaped, X_test_reshaped])
y_all = np.hstack([y_train_encoded, y_test_encoded])

# Split: 70% training, 30% temporary
X_train_final, X_temp, y_train_final, y_temp = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
)

# Split 30% temp: 2/3 untuk testing (20% total), 1/3 untuk validation (10% total ≈ 20% dari training)
X_test_final, X_val_final, y_test_final, y_val_final = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
)

print("✅ FINAL DATA SPLIT:")
total_samples = len(X_all)
print(f"Training: {len(X_train_final)} samples ({len(X_train_final)/total_samples*100:.1f}%)")
print(f"Validation: {len(X_val_final)} samples ({len(X_val_final)/total_samples*100:.1f}%)")
print(f"Testing: {len(X_test_final)} samples ({len(X_test_final)/total_samples*100:.1f}%)")

# 5. BUILD LSTM MODEL
print("\n🧠 BUILDING LSTM MODEL...")

model = Sequential([
    LSTM(128, input_shape=(X_train_final.shape[1], X_train_final.shape[2]), 
         return_sequences=True, activation='relu'),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='softmax')  # 6 classes
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled!")
model.summary()

# 6. TRAINING
print("\n🎯 TRAINING MODEL...")

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_val_final, y_val_final),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 7. EVALUATION
print("\n📊 EVALUATING MODEL...")

# Predict
y_pred_proba = model.predict(X_test_final)
y_pred = np.argmax(y_pred_proba, axis=1)

# Accuracy
accuracy = accuracy_score(y_test_final, y_pred)
print(f"🎯 TEST ACCURACY: {accuracy:.4f}")

# Classification Report
print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test_final, y_pred, 
                          target_names=activity_labels[1].tolist()))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_final, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=activity_labels[1].tolist(),
            yticklabels=activity_labels[1].tolist())
plt.title('Confusion Matrix - LSTM Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. PLOT TRAINING HISTORY
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. FINAL RESULTS
print("\n" + "="*50)
print("🎉 FINAL RESULTS")
print("="*50)
print(f"✅ Final Test Accuracy: {accuracy:.4f}")
print(f"✅ Training Samples: {len(X_train_final)}")
print(f"✅ Validation Samples: {len(X_val_final)}") 
print(f"✅ Testing Samples: {len(X_test_final)}")
print(f"✅ Number of Features: {X_train_final.shape[2]}")
print(f"✅ Number of Classes: 6")

print("\n📁 Output files saved:")
print("   - confusion_matrix.png")
print("   - training_history.png")

print("\n🚀 PROGRAM SELESAI!")