import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenseNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU):
        """
        Flexible MLP with variable hidden layer sizes
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes (e.g. [512, 256, 128])
            output_size: Size of output features
            activation: Activation function to use (default: ReLU)
        """
        super(DenseNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(activation())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(activation())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x