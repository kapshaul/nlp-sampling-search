# NLP Sampling and Search

This is a historical coursework-style project for decoding text from a pretrained
word-level LSTM language model. It compares
several sampling strategies and a beam search decoder, all starting from the prompt
`"the night is dark and full of terrors"`. A reviewed write-up of the methods and
results is at <https://kapshaul.github.io/studies/sampling-search/>.

## Files

| File | Description |
| --- | --- |
| [`LanguageModel.py`](LanguageModel.py) | `LanguageModel` module: embedding, 3-layer LSTM (hidden size 512), and two linear layers with dropout. |
| [`decoder.py`](decoder.py) | Loads the vocab and checkpoint, defines `sample()` and `beamsearch()`, and runs the demo in `main()`. |
| [`got_language_model`](got_language_model) | Pretrained model state dict (~72 MB), loaded with `torch.load`. |
| [`vocab.pkl`](vocab.pkl) | Pickled torchtext field holding the tokenizer and vocabulary. |

The model checkpoint and vocabulary are preserved snapshots from the original work.
They were not retrained or regenerated as part of this repository's cleanup.

## Decoding methods

All methods are implemented in `decoder.py` and generate a fixed `max_len` tokens
after the prompt. Neither `sample()` nor `beamsearch()` stops early on an
end-of-sequence token.

- **Vanilla sampling**: `sample(..., temp=1.0, k=0, p=1)` samples each next token
  from the softmax of the model's output logits.
- **Temperature scaling**: `temp` divides the logits before the softmax. Low values
  approach greedy decoding; high values flatten the distribution.
- **Top-k sampling**: `k > 0` keeps only the `k` most probable tokens, renormalizes,
  and samples from them.
- **Top-p (nucleus) sampling**: `p < 1` sorts probabilities, keeps the smallest
  prefix whose cumulative probability exceeds `p`, renormalizes, and samples.
  Top-k and top-p cannot be combined; an assertion enforces this.
- **Beam search**: `beamsearch(..., beams=B)` keeps `B` hypotheses scored by summed
  log-probabilities. At each step it runs the model once per hypothesis in a
  Python loop (not batched), expands each by its top `B` tokens, and keeps the
  best `B` candidates. The highest-scoring sequence is returned after `max_len` steps.

These descriptions follow the source as written. For known limitations of the
original implementation, see the reviewed write-up linked above. The algorithm code
has not been changed here.

## Running

Run from the repository root, because `decoder.py` loads `got_language_model` and
`vocab.pkl` by relative path:

```bash
python decoder.py
```

There are no command-line options. `argparse` is imported but never used. Settings
such as the prompt, `seed = 42`, and `mlen = 150` (passed as `max_len`) are set in `main()`. By default,
`main()` runs vanilla sampling, temperature 0.0001 and 100, top-k 1 and 20,
top-p 0.001, 0.75, and 1, and beam search with B = 1, 10, and 50. To try other
settings, change the calls in `main()` or call `sample()` / `beamsearch()` directly.
The code uses CUDA when available and falls back to CPU otherwise. Setting seeds and
turning on deterministic algorithms does not guarantee identical output every time,
as the source comments note.

## Historical requirements

A comment in `decoder.py` records the package versions the code was written for:
`torch==1.13.1` and `torchtext==0.6.0`. The code also imports `numpy`. These are
historical requirements, not a tested install recipe. `torchtext.data.get_tokenizer`
and the pickled torchtext field depend on that older torchtext API, and newer
environments may not load them.
