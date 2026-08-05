import re
import pyarrow.parquet as pq
from collections import Counter
from typing import List, Tuple, Dict, Generator
import torch

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()

def build_vocab(parquet_path: str, max_vocab_size: int = 100000, sample_rows: int = 100000) -> Dict[str, int]:
    counter = Counter()
    parquet_file = pq.ParquetFile(parquet_path)

    rows_processed = 0
    print(f"Sampling first {sample_rows} rows to build vocabulary...")

    for batch in parquet_file.iter_batches(batch_size=50000, columns=['DATA']):
        df = batch.to_pandas()
        for text in df['DATA']:
            counter.update(clean_text(text).split())
            rows_processed += 1
            if rows_processed >= sample_rows:
                break
        if rows_processed >= sample_rows:
            break

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    print(f"Vocabulary built! Total unique words: {len(vocab)}")
    return vocab

def stream_parquet_batches(parquet_path: str, batch_size: int = 10000) -> Generator[Tuple[List[str], List[str]], None, None]:
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['DATA', 'TOPIC']):
        df = batch.to_pandas()
        yield [clean_text(t) for t in df['DATA']], df['TOPIC'].tolist()

def collate_batch(batch_texts: List[str], vocab: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    text_list, offsets = [], [0]
    for text in batch_texts:
        tokens_idx = [vocab.get(t, vocab["<UNK>"]) for t in text.split()] or [vocab["<PAD>"]]
        text_list.append(torch.tensor(tokens_idx, dtype=torch.long))
        offsets.append(len(tokens_idx))
    return torch.cat(text_list), torch.tensor(offsets[:-1]).cumsum(dim=0)