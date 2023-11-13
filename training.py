import streamlit as st
from skorch import NeuralNetClassifier
import random
import torch
import os
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skorch.callbacks import EpochScoring
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from sklearn.datasets import fetch_openml
import pickle
from torchvision import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler

# Define model as a global variable to make it accessible for all pages
model = None


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





def training_and_evaluation_app():
    st.title("Training and Evaluation Page")
    st.write("On this page, you can set hyperparameters, train the model, and evaluate its performance.")

    # Schema of the Training Process
    st.subheader("Training Process")
    st.write("1. **Loading the MNIST Dataset**: The MNIST dataset is loaded.")
    st.write("2. **Creating the CNN Model**: A Convolutional Neural Network (CNN) is created with user-defined hyperparameters.")
    st.write("3. **Training the Model**: The model is trained with the specified hyperparameters, such as learning rate,number of epochs")

    # Hyperparameters input
    learning_rate = st.sidebar.selectbox("Learning Rate",[ 0.0001, 1.0, 0.001, 0.0001])
    optimizer = st.sidebar.selectbox("Optimizer", ["Adam"])
    epochs = st.sidebar.selectbox("Number of Epochs",[1,100,10,1])
    batch_size = st.sidebar.selectbox("Batch Size", [32, 512, 64, 1])


    # X_train, X_test, y_train, y_test  = load_mnist_data(batch_size, test_split)  # Pass the batch size to the data loader
    
    # optimizer = torch.optim.Adam if optimizer == "Adam" else torch.optim.SGD

    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(32),
            transforms.ToTensor(),

        ]),
        'test': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(32),
            transforms.ToTensor(),
        ]),
    }
    data_dir = './Dataset'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=2)
                  for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    st.write("Dataset size for training data: ", dataset_sizes['train'])
    st.write("Dataset size for validation data: ", dataset_sizes['test'])
    class_names = image_datasets['train'].classes
    #st.write("Class names: ", class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Train the model
    if st.button("Train Model"):
        st.write(f"Training the model with learning rate: {learning_rate}, batch size: {batch_size}")
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        ####
        best_accuracy = 0.0
        best_model_path = 'best_model.pth'
        ##
        # Training loop
        num_epochs = epochs
        for epoch in range(num_epochs):
            for inputs, labels in dataloaders['train']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
            st.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

            # Make predictions and calculate accuracy
            correct_predictions = 0
            total_samples = 0
            with torch.no_grad():
                for inputs, labels in dataloaders['test']:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

            # Calculate accuracy
            accuracy = correct_predictions / total_samples
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
            print(f'Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy}, Best Accuracy: {best_accuracy}')
            st.write(f'Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy}, Best Accuracy: {best_accuracy}')
                
            