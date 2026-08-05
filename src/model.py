import torch
import torch.nn as nn

class FastTextTopicClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, sparse: bool = False):
        super(FastTextTopicClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean', sparse=sparse)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text, offsets)
        x = self.fc1(embedded)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)