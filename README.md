# Sparse FFN

Train a small network to learn the output of feed-forward (FFN) blocks of an LLM. This distilation technique can then be used to predict the sparsity of the FFN blocks in LLMs. 

## Setup the environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

Training can be accomplished simply by running the `train.py` script:

```
python train.py -d <datapath>.pt -od <outdir> -o <outname> -bs <batch_size> -lr <learning rate> -e <number of epochs>
``` 

## Use trained model

To test and make plots, simply run:

```
python test.py --data <path to data> -mod <trained_model_name>.pt
```

To use trained models, run:

```python
from ffnsparse.query import predict

pred = predict(input_vector, "model_name.pt")
```

where `input_vector` is the embedded input to the FFN block, and `"model_name.pt"` is the path to the trained model. 