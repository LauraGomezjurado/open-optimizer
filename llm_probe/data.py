"""
Data + tokenizer for the probe. Two backends:

  - "tinystories" (pod): HF `datasets` + a GPT2 tokenizer, streamed, packed into
    fixed-length blocks. Reserves one extra id as the [MASK] token for the
    masked/mdlm objectives.
  - "bytes" (anywhere, incl. this CPU box): byte-level vocab (256 + MASK), reads
    a local text blob. Lets the pipeline smoke-test with no downloads.

A FIXED probe batch (same seed) is shared across arms so the factoredness proxy
is measured on identical tokens everywhere.

MASK_TOKEN is the last id (vocab_size-1) in both backends.
"""
import os
import torch

MASK_TOKEN = None   # set by get_data() once vocab is known (module-level for train.py)


class _Data:
    def __init__(self, ids, vocab_size, seq_len, batch_size, device, seed=0):
        self.ids = ids                  # 1D LongTensor of token ids
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.g = torch.Generator().manual_seed(seed)
        n = (len(ids) // seq_len) * seq_len
        self.blocks = ids[:n].view(-1, seq_len)
        # hold out a fixed probe (last 64 blocks) never used for training
        self.n_probe = min(64, self.blocks.shape[0] // 5)
        self.train_blocks = self.blocks[:-self.n_probe]
        self.probe_blocks = self.blocks[-self.n_probe:]
        self.mask_token = vocab_size - 1

    def train_batch(self):
        idx = torch.randint(0, self.train_blocks.shape[0], (self.batch_size,),
                            generator=self.g)
        return self.train_blocks[idx].to(self.device)

    def probe_batch(self):
        b = min(self.batch_size, self.probe_blocks.shape[0])
        return self.probe_blocks[:b].to(self.device)


def _load_bytes(seq_len, batch_size, device):
    # local fallback: read any text lying around, else synth repeating text.
    here = os.path.dirname(__file__)
    cand = [os.path.join(here, "sample.txt"),
            os.path.join(here, "..", "README.md")]
    text = ""
    for c in cand:
        if os.path.exists(c):
            text = open(c, encoding="utf-8", errors="ignore").read()
            break
    if len(text) < seq_len * 200:
        text = (text + " the quick brown fox. ") * 2000
    ids = torch.tensor(list(text.encode("utf-8", errors="ignore")), dtype=torch.long)
    vocab = 257            # 256 bytes + MASK
    return _Data(ids, vocab, seq_len, batch_size, device), vocab - 1


def _load_tinystories(seq_len, batch_size, device, max_tokens=5_000_000):
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    buf = []
    for ex in ds:
        buf.extend(tok.encode(ex["text"]))
        buf.append(tok.eos_token_id)
        if len(buf) >= max_tokens:
            break
    ids = torch.tensor(buf[:max_tokens], dtype=torch.long)
    vocab = tok.vocab_size + 1          # +1 for MASK
    return _Data(ids, vocab, seq_len, batch_size, device), vocab - 1


def get_data(corpus, seq_len, batch_size, device, seed=0):
    global MASK_TOKEN
    if corpus == "bytes":
        d, mask = _load_bytes(seq_len, batch_size, device)
    elif corpus == "tinystories":
        d, mask = _load_tinystories(seq_len, batch_size, device)
    else:
        raise ValueError(corpus)
    MASK_TOKEN = mask
    return d


# module import convenience: train.py imports MASK_TOKEN at top, but it's set by
# get_data(). We expose a getter so the value is read AFTER data load.
def _mask_token():
    return MASK_TOKEN
