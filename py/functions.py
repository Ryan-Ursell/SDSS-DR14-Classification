import torch
import numpy as np
import pandas as pd
import seaborn as sns
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

class SDSSClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SDSSClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_features):
        input_features = self.fc1(input_features)
        input_features = self.relu(input_features)
        input_features = self.fc2(input_features)
        input_features = self.relu(input_features)
        input_features = self.fc3(input_features)
        return self.softmax(input_features)
        
def train_sdss_model(features, labels, 
                     test_ratio=0.2, val_ratio=0.2, 
                     hidden_size=64, num_epochs=20, 
                     batch_size=32, learning_rate=0.001,
                     use_class_weights=False):
    """
    Train and evaluate an SDSS Classifier neural network.

    Parameters:
        features (array-like): Input features.
        labels (array-like): Target labels.
        test_ratio (float): Proportion of the dataset to include in the test split.
        val_ratio (float): Proportion of the train/validation split to allocate to validation.
        hidden_size (int): Number of hidden units in each hidden layer.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        use_class_weights (bool): Whether to apply class weighting to address class imbalance.

    Returns:
        dict: A dictionary containing training and evaluation results, including the model.
    """
    # Split data
    features_train_val, features_test, label_train_val, label_test = train_test_split(
        features, labels, test_size=test_ratio, random_state=42)
    features_train, features_val, label_train, label_val = train_test_split(
        features_train_val, label_train_val, test_size=val_ratio, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    features_train_normalised = scaler.fit_transform(features_train)
    features_val_normalised = scaler.transform(features_val)
    features_test_normalised = scaler.transform(features_test)

    # Encode labels
    label_encoder = LabelEncoder()
    label_train_encoded = label_encoder.fit_transform(label_train)
    label_val_encoded = label_encoder.transform(label_val)
    label_test_encoded = label_encoder.transform(label_test)

    # Convert to PyTorch tensors
    features_train_tensor = torch.tensor(features_train_normalised, dtype=torch.float32)
    features_val_tensor = torch.tensor(features_val_normalised, dtype=torch.float32)
    features_test_tensor = torch.tensor(features_test_normalised, dtype=torch.float32)
    label_train_tensor = torch.tensor(label_train_encoded, dtype=torch.long)
    label_val_tensor = torch.tensor(label_val_encoded, dtype=torch.long)
    label_test_tensor = torch.tensor(label_test_encoded, dtype=torch.long)

    # Optionally compute class weights for imbalanced classes
    if use_class_weights:
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(label_train_encoded), 
            y=label_train_encoded
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # Prepare DataLoaders
    train_dataset = torch.utils.data.TensorDataset(features_train_tensor, label_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(features_val_tensor, label_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(features_test_tensor, label_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    input_size = features_train_tensor.shape[1]
    num_classes = len(np.unique(label_train_tensor))

    model = SDSSClassifier(input_size, hidden_size, num_classes)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_loss_history, val_loss_history = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)

        #print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Test the model
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # Classification Report
    classification_results = classification_report(all_labels, all_preds, 
                                                   target_names=label_encoder.classes_, 
                                                   digits=4)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return {
        'model': model,
        'classification_report': classification_results,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized
    }

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, marker='o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid()
    plt.show()

def plot_cm(cm, cm_normalized, labels):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
    sns.heatmap(cm_normalized, annot=True, fmt='.4f', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.show()