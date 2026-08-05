import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from model import FastTextTopicClassifier, count_parameters
from utils import build_vocab, stream_parquet_batches, collate_batch

def train_pipeline(data_path: str, output_dir: str = "final_models", epochs: int = 1):
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1: Building Vocabulary from streaming batches...")
    vocab = build_vocab(data_path, max_vocab_size=100000)
    print(f"Vocabulary size: {len(vocab)}")

    label_map = {}
    print("Step 2: Initializing Custom Model built from scratch...")
    model = None
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training hardware device: {device}")

    for epoch in range(epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{epochs} ---")
        batch_idx = 0
        all_preds = []
        all_targets = []

        for texts, labels in stream_parquet_batches(data_path, batch_size=10000):
            for l in labels:
                if l not in label_map:
                    label_map[l] = len(label_map)

            if model is None:
                model = FastTextTopicClassifier(len(vocab), embed_dim=64, num_classes=len(label_map)).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

            targets = torch.tensor([label_map[l] for l in labels], dtype=torch.long).to(device)
            text_tensor, offsets = collate_batch(texts, vocab)
            text_tensor, offsets = text_tensor.to(device), offsets.to(device)

            optimizer.zero_grad()
            outputs = model(text_tensor, offsets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())

            batch_idx += 1
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx} processed | Loss: {loss.item():.4f}")

        acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        print(f"\nEpoch {epoch+1} Metrics:")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'label_map': label_map
    }, os.path.join(output_dir, "final_model.pt"))
    print(f"Model saved successfully to {output_dir}/final_model.pt")

if __name__ == "__main__":
    train_pipeline("dataset_10.parquet")