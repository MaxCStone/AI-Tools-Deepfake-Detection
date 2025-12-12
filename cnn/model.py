import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, GlobalAveragePooling2D
import traceback
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

print("TF version:", tf.__version__)
print("GPUs:", gpus)

img_size = 256
batch_size = 8 
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "data/Dataset/Train",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    "data/Dataset/Validation",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    "data/Dataset/Test",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("Train samples:", train_generator.samples, "classes:", train_generator.class_indices)
print("Val samples:", val_generator.samples, "classes:", val_generator.class_indices)
print("Test samples:", test_generator.samples, "classes:", test_generator.class_indices)

model = Sequential([
    Input(shape=(img_size, img_size, 3)),
    Conv2D(8, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(),
    GlobalAveragePooling2D(),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()

try:
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        verbose=2
    )

except Exception:
    print("Training raised an exception:")
    traceback.print_exc()

print("\nGenerating confusion matrix...")

try:
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    y_true = test_generator.classes
    
    cm = confusion_matrix(y_true, y_pred)
    
    class_names = list(test_generator.class_indices.keys())
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to confusion_matrix.png")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClass names: {class_names}")
    
    model.save_weights('cnn.weights.h5')
    print(f"\nModel weights saved to cnn.weights.h5")
    
except Exception:
    print("Error generating confusion matrix:")
    traceback.print_exc()
