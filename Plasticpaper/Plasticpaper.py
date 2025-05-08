import kagglehub
import os
import numpy
from PIL import Image
# tensorflow is not supported by python 3.13 go down to 3.12
# install scipy too
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

supported_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

def validate_images(folder_path, supported_exts=('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
    print(f"Checking images in: {folder_path}")
    bad_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_exts):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()  # verify image integrity
                except Exception as e:
                    print(f"[BROKEN] {file_path}: {e}")
                    bad_files.append(file_path)
    print(f"Finished checking. Found {len(bad_files)} bad files.")
    return bad_files                

# === CONFIGURATION ===
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

base_dir = r"C:\Users\22-0042c\source\repos\Plasticpaper\Plasticpaper\garbage-dataset\garbage-dataset"

# === LOAD DATASET ===
bad_images = validate_images(base_dir)
if bad_images:
    print("You have broken or unsupported images. Consider removing or replacing them.")
else:
    print("All images are valid!")
    
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir,
    labels='inferred',
    label_mode='binary',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

# === SPLIT DATA ===
total_batches = tf.data.experimental.cardinality(dataset).numpy()

train_size = int(0.8 * total_batches)
temp_size = total_batches - train_size
val_size = temp_size // 2
test_size = temp_size - val_size

train_dataset = dataset.take(train_size)
temp_dataset = dataset.skip(train_size)

validation_dataset = temp_dataset.take(val_size)
test_dataset = temp_dataset.skip(val_size)

print(f"Total batches: {total_batches}")
print(f"Train: {train_size}, Validation: {val_size}, Test: {test_size}")

# === PREFETCHING ===
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# === MODEL ===
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# === TRAINING ===
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)

# === EVALUATE ON TEST ===
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc*100:.2f}%")

# === SAVE MODEL ===
model.save('paper_plastic_model.h5')
print("Model saved as paper_plastic_model.h5")