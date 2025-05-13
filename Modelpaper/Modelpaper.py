import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Load the saved model ===
model = tf.keras.models.load_model(
    r"C:\Users\Luis Oliver\source\repos\Plasticpaper\Plasticpaper\paper_plastic_model.h5"
)
print("Model loaded successfully!")

# === Preprocessing function for input images ===
def preprocess_input_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (128, 128))  # Match training IMAGE_SIZE
    img = img / 255.0  # Normalize to [0, 1]
    return img

# === Function to run prediction on a single image and get confidence ===
def predict_image_class(image_path, class_names):
    img = preprocess_input_image(image_path)
    img_expanded = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img_expanded)

    if len(class_names) == 2:  # Binary classification
        confidence = float(predictions[0])
        predicted_label = class_names[int(confidence > 0.5)]
        confidence_score = confidence if confidence > 0.5 else 1 - confidence
    else:  # Multi-class classification
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]
        confidence_score = float(np.max(predictions))

    return predicted_label, confidence_score

# === Optional: Visualize image with prediction label and confidence ===
def show_prediction(image_path, predicted_label, confidence):
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_label} ({confidence*100:.2f}%)')
    plt.axis('off')
    plt.show()

# === MAIN EXECUTION ===
if __name__ == '__main__':
    # Path to the test image
    test_image_path = r'C:\Users\Luis Oliver\source\repos\Plasticpaper\Modelpaper\Test_Image.jpg'

    # Update this list based on your dataset classes
    class_names = ['paper', 'plastic']

    predicted_label, confidence = predict_image_class(test_image_path, class_names)
    print(f"Predicted label: {predicted_label} (Confidence: {confidence*100:.2f}%)")

    show_prediction(test_image_path, predicted_label, confidence)
