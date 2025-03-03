import torch
from ffnsparse.architecture import DenseNN

def predict(input_vector, model_path):
    """
    Make predictions using a trained DenseNN model
    
    Args:
        input_vector: Input tensor of shape matching model's input_size
        model_path: Path to the saved model checkpoint
        
    Returns:
        outputs: Model predictions as numpy array
    """
    checkpoint = torch.load(model_path)
    
    model = DenseNN(
        input_size=checkpoint['input_size'],
        hidden_sizes=checkpoint['hidden_sizes'], 
        output_size=checkpoint['output_size']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.float()
    model.eval()
    
    # Convert input to tensor if needed
    if not isinstance(input_vector, torch.Tensor):
        input_vector = torch.tensor(input_vector)
    input_vector = input_vector.float()
    
    with torch.no_grad():
        outputs = model(input_vector)
        
    return outputs.numpy()