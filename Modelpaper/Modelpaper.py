import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model
model = tf.keras.models.load_model("paper_plastic_model.h5")
print("Model loaded successfully!")

# Preprocessing function for input images
def preprocess_input_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (128, 128))  # Match training IMAGE_SIZE
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Function to run prediction on a single image
def predict_image_class(image_path, class_names):
    img = preprocess_input_image(image_path)
    img_expanded = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img_expanded)
    if len(class_names) == 2:  # Binary classification
        predicted_label = class_names[int(predictions[0] > 0.5)]
    else:  # Multi-class classification
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]
    return predicted_label

# Optional: Visualize image with prediction label
def show_prediction(image_path, predicted_label):
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title(f'Predicted Class: {predicted_label}')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Provide path to your test image here
    test_image_path = 'C:\\Users\\22-0042c\\source\\repos\\Plasticpaper\\Modelpaper\\Test_Image.jpg'  
    # Class names for your dataset
    class_names = ['paper', 'plastic']

    predicted_label = predict_image_class(test_image_path, class_names)
    print(f"Predicted label: {predicted_label}")
    show_prediction(test_image_path, predicted_label)
