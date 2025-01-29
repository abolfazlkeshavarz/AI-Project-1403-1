import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Load and preprocess your image
def preprocess_image(img_path):
    # Open the image
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img)  # Convert to numpy array
    
    # Invert the image if background is black and digits are white
    #img = 255 - img  # Invert the image to make the background white and digits black
    
    # Normalize the image to be between 0 and 1
    img = img / 255.0
    
    # Reshape to match the model's input shape
    img = img.reshape(1, 28, 28, 1)
    
    return img

# Function to test multiple images
def test_multiple_images(image_folder):
    # Get list of image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        
        # Preprocess the image
        img = preprocess_image(img_path)
        
        # Predict the digit in the image
        predictions = model.predict(img)
        predicted_label = np.argmax(predictions)
        
        # Plot the processed image and predicted label
        plt.figure(figsize=(2, 2))
        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Predicted Label: {predicted_label}")
        plt.axis('off')  # Hide axis
        plt.show()

        # Print out the result in the console
        print(f"Image: {img_file} | Predicted label: {predicted_label}")

# Test multiple images from a folder
image_folder = 'testimages'  # Replace with the path to your folder of images
test_multiple_images(image_folder)