"""
One-time data preparation for autoresearch experiments (CPU port).
Downloads a small dataset, trains a BPE tokenizer and pre-tokenizes the data.

Usage:
    python prepare.py                         # TinyStories (default, from HuggingFace)
    python prepare.py --dataset shakespeare   # TinyShakespeare (1MB, from GitHub)
    python prepare.py --max-mb 50             # download only the first 50MB of train text

Data and tokenizer are stored in ~/.cache/autoresearch-cpu/.
Running again with a different --dataset replaces the prepared data.
"""

import os
import sys
import time
import math
import json
import argparse
import pickle

import numpy as np
import requests
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 256        # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 2**17      # max number of tokens for val eval (capped by val split size)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-cpu")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
META_PATH = os.path.join(CACHE_DIR, "meta.json")
VOCAB_SIZE = 2048

TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
DATASETS = {
    # GPT-4 generated short stories, separated by <|endoftext|>
    "tinystories": {
        "train": f"{TINYSTORIES_URL}/TinyStoriesV2-GPT4-train.txt",
        "val": f"{TINYSTORIES_URL}/TinyStoriesV2-GPT4-valid.txt",
        "separator": "<|endoftext|>",
    },
    # ~1MB of Shakespeare; last 10% of paragraphs is the val split
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "separator": "\n\n",
        "val_frac": 0.1,
    },
}

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_text(url, max_bytes=None, max_attempts=5):
    """Download a text file (optionally only the first max_bytes) with retries."""
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            chunks, nbytes = [], 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                chunks.append(chunk)
                nbytes += len(chunk)
                if max_bytes is not None and nbytes >= max_bytes:
                    break
            response.close()
            return b"".join(chunks).decode("utf-8", errors="ignore")
        except requests.RequestException as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {url}: {e}")
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    print(f"Data: could not download {url}")
    sys.exit(1)


def split_docs(text, separator, truncated=False):
    docs = text.split(separator)
    if truncated:
        docs = docs[:-1]  # last doc may be cut off mid-way
    return [d.strip() for d in docs if d.strip()]


def download_data(dataset, max_mb):
    """Return (train_docs, val_docs) for the chosen dataset."""
    spec = DATASETS[dataset]
    sep = spec["separator"]
    if "url" in spec:
        print(f"Data: downloading {spec['url']}")
        docs = split_docs(download_text(spec["url"]), sep)
        n_val = max(1, int(len(docs) * spec["val_frac"]))
        return docs[:-n_val], docs[-n_val:]
    max_bytes = max_mb * 1024 * 1024 if max_mb > 0 else None
    print(f"Data: downloading train split (first {max_mb}MB)" if max_bytes else "Data: downloading full train split")
    train_docs = split_docs(download_text(spec["train"], max_bytes), sep, truncated=max_bytes is not None)
    print("Data: downloading val split")
    val_docs = split_docs(download_text(spec["val"]), sep)
    return train_docs, val_docs

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def train_tokenizer(train_docs, max_chars=200_000_000, doc_cap=10_000):
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle. Returns the encoding."""
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    def text_iterator():
        nchars = 0
        for doc in train_docs:
            doc = doc[:doc_cap]
            nchars += len(doc)
            yield doc
            if nchars >= max_chars:
                return

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    print(f"Tokenizer: trained in {time.time() - t0:.1f}s, saved to {tokenizer_pkl}")

    # Build token_bytes lookup for BPB evaluation
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), token_bytes_path)

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    decoded = enc.decode(enc.encode_ordinary(test))
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")
    return enc

# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------

def tokenize_split(enc, docs, path, chunk_size=10_000):
    """Tokenize docs into one flat uint16 stream, each doc prefixed with BOS."""
    bos = enc.encode_single_token(BOS_TOKEN)
    arrays = []
    for i in range(0, len(docs), chunk_size):
        for ids in enc.encode_ordinary_batch(docs[i:i + chunk_size], num_threads=os.cpu_count() or 1):
            arrays.append(np.array([bos] + ids, dtype=np.uint16))
    tokens = np.concatenate(arrays)
    tokens.tofile(path)
    return len(tokens)

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_dataset_name():
    with open(META_PATH) as f:
        return json.load(f)["dataset"]


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _load_tokens(split):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    assert os.path.exists(path), f"{path} not found. Run prepare.py first."
    return np.memmap(path, dtype=np.uint16, mode="r")


def make_dataloader(tokenizer, B, T, split, seed=1337):
    """
    Yields (inputs, targets, epoch) batches of shape (B, T) on CPU.
    train: random windows from the token stream (seeded, reproducible).
    val: consecutive non-overlapping windows from the start (deterministic).
    """
    assert split in ["train", "val"]
    data = _load_tokens(split)
    n = len(data) - T - 1
    assert n > 0, f"{split} split too small for T={T}"
    gen = torch.Generator().manual_seed(seed)
    tokens_seen = 0
    pos = 0
    while True:
        if split == "train":
            starts = torch.randint(0, n, (B,), generator=gen).tolist()
        else:
            starts = []
            for _ in range(B):
                if pos > n:
                    pos = 0
                starts.append(pos)
                pos += T
        rows = np.stack([data[s:s + T + 1] for s in starts]).astype(np.int64)
        rows = torch.from_numpy(rows)
        tokens_seen += B * T
        epoch = 1 + tokens_seen // len(data)
        yield rows[:, :-1].contiguous(), rows[:, 1:].contiguous(), epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN and a fixed slice of the val split so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cpu")
    num_val_tokens = len(_load_tokens("val")) - 1
    steps = min(EVAL_TOKENS, num_val_tokens) // (batch_size * MAX_SEQ_LEN)
    assert steps > 0, "val split too small for this batch size"
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch (CPU)")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="tinystories", help="Dataset to prepare")
    parser.add_argument("--max-mb", type=int, default=200, help="TinyStories only: MB of train text to download (-1 = all, ~2.2GB)")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Download data
    train_docs, val_docs = download_data(args.dataset, args.max_mb)
    print(f"Data: {len(train_docs):,} train docs, {len(val_docs):,} val docs")
    print()

    # Step 2: Train tokenizer
    enc = train_tokenizer(train_docs)
    print()

    # Step 3: Pre-tokenize
    print("Data: tokenizing...")
    n_train = tokenize_split(enc, train_docs, os.path.join(DATA_DIR, "train.bin"))
    n_val = tokenize_split(enc, val_docs, os.path.join(DATA_DIR, "val.bin"))
    with open(META_PATH, "w") as f:
        json.dump({"dataset": args.dataset, "train_tokens": n_train, "val_tokens": n_val}, f)
    print(f"Data: {n_train:,} train tokens, {n_val:,} val tokens at {DATA_DIR}")
    print()
    print("Done! Ready to train.")
