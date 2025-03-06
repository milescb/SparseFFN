import numpy as np
import argparse

import torch

from ffnsparse.architecture import DenseNN
from ffnsparse.dataset import get_data_loaders
from ffnsparse.plotting import compare_distributions, plot_multiple_histograms_with_ratio

def evaluate(model, test_loader):
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.float()
            outputs = model(inputs)
            all_outputs.append(outputs.numpy())
            all_targets.append(target.numpy())
    
    predictions = np.concatenate(all_outputs, axis=0) 
    targets = np.concatenate(all_targets, axis=0)    
    
    return predictions, targets

def main():
    checkpoint = torch.load(args.model_save_path)
    input_size = checkpoint['input_size']
    hidden_sizes = checkpoint['hidden_sizes']
    output_size = checkpoint['output_size']
    batch_size = checkpoint['batch_size']
    learning_rate = checkpoint['learning_rate']
    
    print("Hyperparameters and network shape: ")
    print(f"Input size: {input_size}")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Output size: {output_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    train_loader, test_loader, split_indices = get_data_loaders(args.data, 
                                                                batch_size=batch_size, 
                                                                train_test_split=0.8,
                                                                num_workers=4,
                                                                pin_memory=True)
    
    model = DenseNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.float()
    
    predictions, targets = evaluate(model, test_loader)
    
    compare_distributions(predictions[0], targets[0], save='plots/comparison_single_layer.png', xlim=(-0.25, 0.25))
    plot_multiple_histograms_with_ratio([predictions[0], targets[0]], bins=25, range=(-0.25, 0.25),
                                        xlabel='Value', ylabel='Frequency', labels=['Predictions', 'Targets'],
                                        save='plots/comparison_single_layer_ratio.png', histtype='step')
    
    # concat_predictions = np.concatenate(predictions, axis=0)
    # concat_targets = np.concatenate(targets, axis=0)
    # compare_distributions(concat_predictions, concat_targets, save='plots/comparison_all_layers.png', xlim=(-0.25, 0.25))
    # plot_multiple_histograms_with_ratio([concat_predictions, concat_targets], bins=25, range=(-0.25, 0.25),
    #                                     xlabel='Value', ylabel='Frequency', labels=['Predictions', 'Targets'],
    #                                     save='plots/comparison_all_layers_ratio.png', histtype='step')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mod', '--model-save-path', type=str, 
                        default="trained_models/model.pt", 
                        help='Path to saved model')
    parser.add_argument('--data', type=str, 
                        default="/mnt/storage/spffn/training_data/bylayer/layer_2.pt", 
                        help='Path to dataset')
    args = parser.parse_args()
    
    main()