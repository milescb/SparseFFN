#!/usr/bin/bash
# filepath: /home/spffn/NNTraining/all_layers.sh

# Define output directory
OUTPUT_DIR="train_all_layers"
EPOCHS=50

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Loop through all layers (0-31)
for i in {0..31}
do
    echo "=========================================="
    echo "Training layer_${i} (${i}/31)"
    echo "=========================================="
    
    # Run the training script for this layer
    python train.py -od ${OUTPUT_DIR} -o layer_${i} -e ${EPOCHS} -d /mnt/storage/spffn/training_data/bylayer/layer_${i}.pt
    
    echo "Clearing GPU cache..."
    python -c "import torch; torch.cuda.empty_cache()"
    
    echo "Completed layer_${i}"
    echo ""
done

echo "All layers training complete!"