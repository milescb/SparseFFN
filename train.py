import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from ffnsparse.architecture import DenseNN
from ffnsparse.dataset import get_data_loaders
from ffnsparse.plotting import plot_loss

def train(batch_size=32, learning_rate=0.0001, nEpochs=75):
    
    print(f"Training FFN model with batch size: {batch_size}, learning rate: {learning_rate}, epochs: {nEpochs}")
    
    input_size = 4_096
    hidden_sizes = [1_024, 512, 1_024, 4_096, 10_000]
    output_size = 14_336
    
    model = DenseNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size
    )
    model = model.float().to('cuda')
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader, test_loader, split_indices = get_data_loaders(DATA, batch_size=batch_size, train_test_split=0.8)
    
    train_loss = []
    val_loss = []
    
    for epoch in range(nEpochs):
        model.train()
        epoch_train_loss = 0.0
        train_samples = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float().to('cuda')
            targets = targets.float().to('cuda')
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
        
        avg_train_loss = epoch_train_loss / train_samples
        train_loss.append(avg_train_loss)
        
        model.eval()
        epoch_val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                
                inputs = inputs.float().to('cuda')
                targets = targets.float().to('cuda')
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                epoch_val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
        
        avg_val_loss = epoch_val_loss / val_samples
        val_loss.append(avg_val_loss)
        
        print(f"Epoch: {epoch}/{nEpochs-1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'split_indices': split_indices,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'nEpochs': nEpochs,
        'input_size': input_size,
        'hidden_sizes': hidden_sizes,
        'output_size': output_size
    }, os.path.join(args.outdir, args.out_name+'.pt'))
    plot_loss(train_loss, val_loss, 
              save=os.path.join(args.outdir, args.out_name+'_loss.png'))
    
def main():
    
    learning_rate = float(args.learning_rate)
    batch_size = int(args.batch_size)
    nEpochs = int(args.nEpochs)
    
    train(learning_rate=learning_rate, batch_size=batch_size, nEpochs=nEpochs)
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Train FFN model')
    parser.add_argument('--data', type=str, 
                        default="/mnt/storage/spffn/training_data/processed_data_layer2.pt", 
                        help='Path to dataset')
    parser.add_argument('-od', '--outdir', type=str, default="trained_models", help='Output directory')
    parser.add_argument('-o', '--out-name', type=str, default="model", help='Path to save model')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-e', '--nEpochs', type=int, default=20, help='Number of epochs')
    args = parser.parse_args()
    
    DATA = args.data
    
    main()