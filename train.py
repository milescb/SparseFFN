import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from ffnsparse.architecture import DenseNN
from ffnsparse.dataset import get_data_loaders
from ffnsparse.plotting import plot_loss

def train(batch_size=64, learning_rate=0.001, nEpochs=10):
    
    # Set default tensor type to float16
    torch.set_default_dtype(torch.float16)
    
    # Define model
    model = DenseNN(
        input_size=4096,
        hidden_sizes=[1024, 512, 1024],
        output_size=14336
    )
    model = model.float()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader, test_loader, split_indices = get_data_loaders(DATA, batch_size=batch_size, train_test_split=0.8)
    
    train_loss = []
    val_loss = []
    
  # Train and validate in the same loop
    for epoch in range(nEpochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_samples = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            # Ensure inputs and targets are float32
            inputs = inputs.float()
            targets = targets.float()
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Accumulate batch loss
            epoch_train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / train_samples
        train_loss.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Convert to float32
                inputs = inputs.float()
                targets = targets.float()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Accumulate batch loss
                epoch_val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
        
        # Calculate average validation loss for the epoch
        avg_val_loss = epoch_val_loss / val_samples
        val_loss.append(avg_val_loss)
        
        print(f"Epoch: {epoch}/{nEpochs-1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # Save model with split indices
    torch.save({
        'model_state_dict': model.state_dict(),
        'split_indices': split_indices
    }, "trained_models/model.pt")
    plot_loss(train_loss, val_loss, save='plots/loss.png')
    
def main():
    train()
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Train FFN model')
    parser.add_argument('--data', type=str, 
                        default="/mnt/storage/spffn/training_data/processed_data_layer2.pt", 
                        help='Path to dataset')
    args = parser.parse_args()
    
    DATA = args.data
    
    main()