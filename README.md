# NLP Sampling and Search

This coursework project decodes text from a provided, pretrained word-level LSTM language model. It compares vanilla, temperature, top-k, and top-p (nucleus) sampling with beam search. Every run starts from the prompt `"the night is dark and full of terrors"`. The decoders are the part implemented here; the model checkpoint and vocabulary were supplied with the assignment.

A reviewed write-up of the methods and recorded observations is available at [kapshaul.github.io/studies/sampling-search](https://kapshaul.github.io/studies/sampling-search/). According to that write-up, the checkpoint was trained on the first five *A Song of Ice and Fire* novels. The repository itself does not include training data or code.

## Files

| File | Role |
| --- | --- |
| [`decoder.py`](decoder.py) | Entry point. Loads the vocabulary and checkpoint, defines `sample()`, `beamsearch()`, and `reverseNumeralize()`, and runs a fixed demo in `main()`. |
| [`LanguageModel.py`](LanguageModel.py) | `LanguageModel(nn.Module)`: 100-d embedding → 3-layer LSTM (hidden size 512, sequence-first) → dropout → `Linear(512, 512)` + ReLU → `Linear(512, vocab_size)`. |
| [`got_language_model`](got_language_model) | Pretrained `state_dict` for `LanguageModel` (about 72 MB). |
| [`vocab.pkl`](vocab.pkl) | Pickled torchtext 0.6 `Field` providing the tokenizer (`text_field.tokenize`), numericalization (`text_field.process`), and vocabulary (`text_field.vocab.itos`). |

The checkpoint and vocabulary are preserved snapshots from the original work. They have not been retrained or regenerated. There is no `requirements.txt`, and no generated text is stored in the repository.

## Environment

The only recorded environment is a comment at the top of `decoder.py`:

```text
#!pip install torchtext==0.6.0 torch==1.13.1
```

The code also imports `numpy`. Treat this pair as the historical environment, not as a tested install recipe:

- `vocab.pkl` is a pickled torchtext 0.6 `Field`, so unpickling it requires a torchtext release that still provides that class at its original import path. Later torchtext releases moved and then removed the legacy `Field` API.
- `from torchtext.data import get_tokenizer` is imported but not used.
- Any Python version you choose must also have published wheels for `torch==1.13.1`.

No working environment was re-created or re-run for this README.

## Running

Run from the repository root, because the checkpoint and vocabulary are loaded by relative path:

```bash
python decoder.py
```

`main()` performs these steps:

1. Selects `cuda` if it is available, otherwise `cpu`, and logs the choice.
2. Loads `vocab.pkl`, builds `LanguageModel(vocab_size)`, loads `got_language_model` with `torch.load(chkpt)`, and calls `lm.eval()` so that dropout is disabled.
3. Enables `torch.use_deterministic_algorithms(True)` and sets `CUBLAS_WORKSPACE_CONFIG=:16:8` at import time. Before each configuration, it resets `torch.manual_seed(42)` and `np.random.seed(42)`.
4. Prints 11 decodes of `mlen = 150` tokens each, under these headers:

| Printed header | Call |
| --- | --- |
| Vanilla Sampling | `sample(..., max_len=150)` (`temp=1.0, k=0, p=1`) |
| Temp-Scaled Sampling 0.0001 / 100 | `sample(..., temp=0.0001)` and `temp=100` |
| Top-k Sampling 1 / 20 | `sample(..., k=1)` and `k=20` |
| Top-p Sampling 0.001 / 0.75 / 1 | `sample(..., p=0.001)`, `p=0.75`, `p=1` |
| Beam Search B=1 / 10 / 50 | `beamsearch(..., beams=1)`, `10`, `50` |

The script prints generated text and logs progress; it writes no result files. Each generated result is the prompt followed by 150 tokens, joined with spaces. There are **no command-line options**. `argparse` and `sys` are imported but never used. To change the prompt, seed, length, or configurations, edit `main()` or import `decoder` and call `sample()` / `beamsearch()` directly.

## Decoding implementation

Both decoders lowercase the prompt, tokenize and numericalize it with the pickled field, and feed the whole prompt through the LSTM once from zero hidden and cell states. After that, each step feeds one token together with the carried `(h, c)` state. `hidden_size = 512` and `num_layers = 3` are hard-coded in both functions and must match the `LanguageModel` defaults.

### `sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1)`

Each step divides the logits by `temp`, applies softmax, optionally filters, renormalizes, and draws one token with `torch.multinomial`.

- **Temperature:** `out / temp` is applied before softmax. Small positive values sharpen the distribution toward the arg-max token, and large values flatten it toward uniform.
- **Top-k (`k > 0`):** keeps the `k` largest probabilities and zeroes the rest. `k = 1` gives greedy decoding (up to ties).
- **Top-p (`p < 1`):** sorts the probabilities, finds the first index where the cumulative sum strictly exceeds `p`, and keeps every token up to and including that index. The threshold-crossing token is therefore retained. `torch.cumsum` runs with deterministic algorithms temporarily disabled.
- `assert (k==0 or p==1)` prevents combining top-k and top-p. When `k > 0`, the assertion requires `p == 1`.

### `beamsearch(model, text_field, beams=5, prompt="", max_len=50)`

Each hypothesis carries its token sequence, a cumulative log-probability score, its last input, and its own `(h, c)` state. At each step, every hypothesis is run through the model separately in a Python loop, so there is no batching across beams. Each hypothesis is expanded with its top `beams` tokens by `log_softmax`, the candidates are sorted by score, and the best `beams` are kept. The first expansion produces `beams` candidates; subsequent full-beam expansions produce `beams²`. After `max_len` steps, the highest-scoring sequence is returned. In the demo's evaluation mode, beam search draws no random samples; direct callers that leave dropout active can get stochastic results. `beams=1` is greedy decoding.

### Parameter ranges the code assumes

These are the ranges under which the source behaves as described. The functions do **not** validate them:

| Parameter | Assumed range | Outside that range |
| --- | --- | --- |
| `temp` | finite, `> 0` | `0` divides by zero, and negative values invert the ranking |
| `k` | integer, `0 ≤ k ≤ vocab size` | `torch.topk` raises an error if `k` exceeds the vocabulary |
| `p` | `0 < p ≤ 1` | `p ≤ 0` keeps only the top token. If `p` is so close to 1 that no cumulative value exceeds it, `.min()` on an empty tensor raises an error. |
| `beams` | integer, `1 ≤ beams ≤ vocab size` | Invalid widths can fail during top-k selection or leave no candidates to return. |

## Limitations of the code as written

- **Fixed length, no EOS handling.** Neither decoder stops at an end-of-sequence token. Both always generate exactly `max_len` tokens and include any special tokens in the output string.
- **No length normalization.** Beam search sums raw log-probabilities. Because all hypotheses have the same length, this does not bias the ranking here, while length normalization would be an optional scoring choice for comparisons between different-length hypotheses if EOS handling were added.
- **Autograd overhead.** `lm.eval()` disables dropout but does not disable gradient tracking. Decoding uses neither `no_grad` nor inference mode, so wide beam searches can retain unnecessary autograd graphs and consume additional memory.
- **Checkpoint device.** `torch.load(chkpt)` is called without `map_location`, even though the model is placed on the selected device. If the checkpoint stores CUDA tensors, loading fails on a CPU-only machine. The usual workaround is to pass `map_location=dev`, which is not applied here.
- **Limits of determinism.** The source comments note that deterministic settings do not guarantee identical output every time, especially on CUDA. Output can also differ across hardware and library versions.
- **Top-k versus beam 1.** `k=1` sampling and `beams=1` beam search should make the same greedy choices under matching settings. The reviewed write-up notes that the original report showed different continuations for the two, and the archived material does not identify the cause.
- **Qualitative results only.** Generated samples are not saved, and no quality metric is computed. Comparisons between settings are observations about individual samples.
