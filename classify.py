# Import necessary libraries
import streamlit as st
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import pickle
from streamlit_drawable_canvas import st_canvas
import cv2



class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(2304, 100) # 1600 = number channels * width * height
        self.fc2 = nn.Linear(100, 10)
        self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # flatten over channel, height and width = 1600
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


# Create a Streamlit app page for image classification
def image_classification_app():
    st.title("Image Classification Page")
    st.write("Upload an image or write a Digit for digit classification.")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the uploaded image
        model_path = 'best_model_20epochs.pth'  # Path to the best model from training
        model = SimpleCNN()  # Model
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode

        # Define transformations for inference on a single image
        transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize(32),
                    transforms.ToTensor(),

                ])

        image = Image.open(uploaded_image).convert("RGB")
        input_image = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(input_image)
            _, predicted_class = torch.max(output, 1)

        # Display the predicted class
        print(f"The predicted class is: {predicted_class.item()}")
        st.write(f"The predicted class is: {predicted_class.item()}")

    SIZE = 192
    mode = st.checkbox("Draw Digit", True)
    canvas_result1 = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw" if mode else "transform",
        key='canvas2')
    if st.button("Predict"):
        if canvas_result1.image_data is not None:
            cv2.imwrite(f"test.jpg",  canvas_result1.image_data)
            model_path = './best_model_20epochs.pth'  # Path to the best model from training
            model = SimpleCNN()  # Model
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set the model to evaluation mode

            # Define transformations for inference on a single image
            transform = transforms.Compose([
                        transforms.Grayscale(),
                        transforms.Resize(32),
                        transforms.ToTensor(),

                    ])
            # Load and preprocess the single image
            image_path = 'test.jpg'
            image = Image.open(image_path).convert("RGB")
            input_image = transform(image).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                output = model(input_image)
                _, predicted_class = torch.max(output, 1)

            # Display the predicted class
            print(f"The predicted class is: {predicted_class.item()}")
            st.write(f"The predicted class is: {predicted_class.item()}")
      
