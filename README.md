# Topic Classification: Memory-Efficient Custom Deep Learning

This repository contains a custom-built PyTorch deep learning pipeline designed to classify text topics from a massive 4GB (10 million rows) Parquet dataset. 

Because loading 4GB of text into memory simultaneously crashes most environments, this project heavily leverages out-of-core learning and data streaming. The entire architecture was built strictly from scratch, strictly adhering to the constraint of avoiding any pre-trained models.

## a. Setup Instructions
**Environment Setup:**
This code is optimized for hardware-constrained environments and cloud setups like Google Colab (using a T4 GPU). It requires Python 3.8+.

**Dependencies Installation:**
Run the following to install the exact required libraries:
```bash
pip install -r requirements.txt
```

b. Training Instructions
To train the model from scratch, ensure dataset_10.parquet is in the root directory. Then execute:
python src/train.py

> Note: The script handles everything dynamically. It streams the 4GB file in chunks using PyArrow, builds a 100,000-word vocabulary using early stopping to prevent CPU bottlenecking, and automatically saves the trained weights to final_models/final_model.pt.
> 
c. Inference Instructions
To test the model on unseen text, run:
python src/inference.py

To run the rigorous batch-testing script (which calculates prediction confidence), execute:
python src/test_inference.py

d. Input/Output Schema
 * Input Data: Compressed Parquet file with two columns: DATA (raw text string) and TOPIC (target label string).
 * Model Input: Raw text string (dynamically tokenized, padded/mapped to <UNK>, and converted to tensor offsets during forward passes).
 * Model Output: Predicted topic string and softmax confidence score.
e. Reproducibility
 * Random Seeds: Torch seeds are fixed for weight initialization reproducibility.
 * Pipeline: The pipeline is completely end-to-end. Executing train.py handles the data streaming, dynamic label mapping, training loop, and artifact saving with zero manual intervention required.