import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, RandomFlip, RandomRotation,
                                     RandomZoom, RandomContrast)
from tensorflow.keras.optimizers import Adam

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# === SUPPORTED IMAGE EXTENSIONS ===
supported_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

def load_image(file_path):
    try:
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        return image
    except tf.errors.InvalidArgumentError:
        print(f"Invalid image file: {file_path}")
        return None    

# === CONFIGURATION ===
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.0005

base_dir = r"C:\Users\Luis Oliver\source\repos\Plasticpaper\Plasticpaper\garbage-dataset\garbage-dataset"

# === LOAD VALID IMAGE PATHS AND LABELS ===
def get_image_paths_and_labels(base_dir):
    image_paths, labels = [], []
    class_names = sorted(os.listdir(base_dir))
    class_to_label = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(supported_exts):
                path = os.path.join(class_dir, fname)
                image = load_image(path)
                if image is not None:
                    image_paths.append(path)
                    labels.append(class_to_label[class_name])
                else:
                    print(f"Skipping invalid image: {path}")
    return image_paths, labels, class_names

image_paths, labels, class_names = get_image_paths_and_labels(base_dir)
print(f"Found {len(image_paths)} valid images belonging to {len(class_names)} classes.")

# === DATA PIPELINE ===
path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img, label

dataset = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=len(image_paths), seed=123)

# === DATASET SPLITTING ===
total_count = len(image_paths)
train_count = int(0.8 * total_count)
val_count = int(0.1 * total_count)
test_count = total_count - train_count - val_count

train_dataset = dataset.take(train_count)
rest_dataset = dataset.skip(train_count)
validation_dataset = rest_dataset.take(val_count)
test_dataset = rest_dataset.skip(val_count)

# === DATA AUGMENTATION ===
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomContrast(0.1),
])

train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# === BATCH AND PREFETCH ===
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === MODEL ===
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")

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
])

if num_classes == 2:
    model.add(Dense(1, activation='sigmoid'))
    loss_fn = 'binary_crossentropy'
else:
    model.add(Dense(num_classes, activation='softmax'))
    loss_fn = 'sparse_categorical_crossentropy'

optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

model.summary()

# === TRAINING ===
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)

# === EVALUATION ===
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc*100:.2f}%")

# === SAVE MODEL ===
model.save('paper_plastic_model.h5')
print("Model saved as paper_plastic_model.h5")
