import streamlit as st

# About page
def about():
    st.title("About")
    st.write("Welcome to the MNIST Neural Network Trainer!")
    st.write("This app is designed to teach you how to define and train a simple Neural Network using PyTorch. Below, we'll walk you through the process step by step.")

    st.subheader("Step 1: Loading the MNIST Dataset")
    st.write("We start by loading the MNIST dataset. The MNIST dataset contains images of handwritten digits (0-9), making it a great choice for learning computer vision and deep learning.")
    st.write("You can learn more about the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and the transformations used in the [PyTorch documentation](https://pytorch.org/vision/stable/transforms.html).")
    st.code("""
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
        data_dir = './Dataset/'

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'test']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                     shuffle=True, num_workers=2)
                      for x in ['train', 'test']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
        print("Dataset size for training data: ", dataset_sizes['train'])
        print("Dataset size for validation data: ", dataset_sizes['test'])

        class_names = image_datasets['train'].classes
        print("Class names: ", class_names)
    """)

    st.subheader("Step 2: Defining the Neural Network Model")
    st.write("Next, we define our Neural Network model using PyTorch. In this example, we'll create a simple Convolutional Neural Network (CNN).")
    st.write("The CNN architecture used in this example consists of two convolutional layers followed by two fully connected layers.")
    st.write("You can learn more about defining neural networks in PyTorch from the [PyTorch documentation](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html).")
    st.code("""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
    """)

    st.subheader("CNN Architecture Details")
    st.write("In the CNN model, we have:")
    st.write("- Two convolutional layers (conv1 and conv2) with 32 and 64 filters, respectively, both using ReLU activation.")
    st.write("- Max-pooling layers following each convolutional layer to downsample the feature maps.")
    st.write("- Two fully connected layers (fc1 and fc2) for classification.")
    st.write("The choice of this architecture is common for image classification tasks, and you can experiment with different architectures for your specific problem.")


    st.subheader("Step 3: Training the Neural Network")
    st.write("Now, it's time to train our Neural Network model.")
    st.write("Let's dive deeper into the training process:")

    st.subheader("3.1 Define Hyperparameters")
    st.write("Before training, we need to set hyperparameters such as learning rate and the number of epochs.")
    st.code("""
    # Define hyperparameters
    learning_rate = 0.001
    max_epochs = 10
    """)

    st.subheader("3.2 Create a NeuralNetClassifier")
    st.write("This step involves defining the model, [optimizer](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), [loss criterion](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#loss-function), and other training settings.")
    
    st.code("""
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        num_epochs = max_epochs
        best_accuracy = 0.0
        best_model_path = 'best_model.pth'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """)
    st.write("In this step, we call a `SimpleCNN` created earlier.")
    st.write("1. `criterion=nn.CrossEntropyLoss`: The criterion (or loss function) is used to calculate the difference between the model's predictions and the actual labels. For classification tasks, cross-entropy loss is commonly used. It's a measure of how well the model's predictions match the ground truth labels. Learn more about loss functions in the [PyTorch neural networks tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#loss-function).")
    st.write("2. `lr=learning_rate`: The learning rate determines the size of the steps taken during optimization. It's a crucial hyperparameter to adjust for efficient training. A small learning rate may result in slow convergence, while a large one may lead to overshooting the optimal solution.")
    st.write("3. `num_epochs=max_epochs`: This specifies the maximum number of training epochs (iterations). One epoch represents one pass through the entire training dataset. Training can stop earlier if the model converges before reaching the maximum epochs.")
    st.write("4. `device='cuda' if torch.cuda.is available() else 'cpu'`: This sets the device for training, either GPU ('cuda') or CPU ('cpu'). Using a GPU can significantly speed up training for deep learning models. Learn more about GPU support in PyTorch in the [PyTorch documentation](https://pytorch.org/docs/stable/notes/cuda.html).")
    st.write("5. best_accuracy, best_model were parameters used to store the best accuracy model after each epoch")
    st.subheader("3.3 Train the Model")
    st.write("The model is trained using the following code ")

    st.code("""
        # Train the model
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

    """)

    #st.write("Congratulations! You have now learned how to set hyperparameters, create a `NeuralNetClassifier`, and train a neural network using skorch.")

    st.write("You can experiment with different hyperparameters and see how they affect the training process and model performance on the 'Training and Evaluation' page.")

    st.write("Happy learning!")
