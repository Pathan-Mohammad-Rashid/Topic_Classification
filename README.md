# Topic Classification: Memory-Efficient Custom Deep Learning

This repository contains a custom-built PyTorch deep learning pipeline designed to classify text topics from a massive **4GB (10 million rows)** Parquet dataset.

Because loading the entire dataset into memory simultaneously can exhaust system resources, this project leverages **out-of-core learning** and **data streaming**. The complete architecture is implemented from scratch, strictly adhering to the constraint of **not using any pre-trained models**.

---

## a. Setup Instructions

### Environment Setup

This project is optimized for hardware-constrained environments and cloud platforms such as **Google Colab (T4 GPU)**.

**Requirements:**
- Python 3.8+
- CUDA-enabled GPU (recommended but optional)

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## b. Training Instructions

Place the dataset file:

```
dataset_10.parquet
```

in the project's root directory.

Then start training by running:

```bash
python src/train.py
```

> **Note**
>
> The training pipeline is fully automated. It:
>
> - Streams the **4GB Parquet dataset** using **PyArrow**
> - Processes data in memory-efficient chunks
> - Builds a **100,000-word vocabulary**
> - Uses early stopping during vocabulary construction to reduce CPU bottlenecks
> - Dynamically creates label mappings
> - Trains the neural network from scratch
> - Automatically saves the trained model to:
>
> ```text
> final_models/final_model.pt
> ```

---

## c. Inference Instructions

### Predict on New Text

```bash
python src/inference.py
```

### Run Batch Evaluation

To evaluate multiple samples and compute prediction confidence scores:

```bash
python src/test_inference.py
```

---

## d. Input / Output Schema

### Input Dataset

A compressed **Parquet** file containing the following columns:

| Column | Description |
|--------|-------------|
| `DATA` | Raw input text |
| `TOPIC` | Target topic label |

### Model Input

- Raw text string
- Dynamic tokenization
- Unknown words mapped to `<UNK>`
- Sequence converted into tensor offsets during the forward pass

### Model Output

- Predicted topic label
- Softmax confidence score

---

## e. Reproducibility

### Random Seeds

PyTorch random seeds are fixed to ensure reproducible weight initialization.

### End-to-End Pipeline

The entire workflow is automated. Running:

```bash
python src/train.py
```

will perform:

1. Streaming the Parquet dataset
2. Vocabulary construction
3. Label encoding
4. Model training
5. Validation
6. Model checkpointing
7. Final model saving

No manual preprocessing or intermediate steps are required.