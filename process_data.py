import os
import numpy as np

import argparse

import torch

def load_torch_pickle(filepath):
    return torch.load(filepath)

def list_torch_files(directory, pattern="prompt_"):
    if os.path.exists(directory):
        return [f for f in os.listdir(directory) if f.startswith(pattern) and f.endswith('.pkl')]
    else:
        return []

def main():
    
    processed_data = []

    for f in torch_files:
        print(f"Loading: {f}")
        
        data = load_torch_pickle(os.path.join(datadir, f))
        model_data = data["data"]
        layer_i = model_data["layers"][2]
        
        for i in range(len(layer_i)):
            print(f"Processing pass {i}")
            p = layer_i[i]
            
            processed_data.append((p["input"], p["post silu"]))
            
    print("Saving processed data")
    torch.save(os.path.join(args.outdir, "processed_data.pt"), processed_data)
            
    return 0

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process PyTorch data files')
    parser.add_argument('--datadir', type=str, default='data', help='Directory containing PyTorch data files')
    parser.add_argument('--outdir', type=str, default='/mnt/storage/spffn/metrics/', help='Output file to save processed data')
    args = parser.parse_args()
    
    datadir = args.datadir
    torch_files = list_torch_files(datadir)
    
    main()