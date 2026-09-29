######################################################
# Use these package versions
#!pip install torchtext==0.6.0 torch==1.13.1
######################################################


import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8' #This is a command to reduce non-deterministic behavior in CUDA
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import get_tokenizer
import sys
import argparse
from LanguageModel import LanguageModel
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')



def main():
  chkpt = "got_language_model"

  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info('Using device: {}'.format(dev))

  logging.info("Loading tokenizer and vocab from vocab.pkl")  
  text_field = pickle.load(open("vocab.pkl", "rb"))
  vocab_size = len(text_field.vocab.itos)

  logging.info("Loading checkpoint {}".format(chkpt))
  lm = LanguageModel(vocab_size).to(dev)
  lm.load_state_dict(torch.load(chkpt))
  lm.eval()


  p = "the night is dark and full of terrors"

  # Torch is a bit frustrating at times and some things that ought to be deterministic are not.
  # This is an attempt to resolve that, but it doesn't work 100% of the time
  torch.use_deterministic_algorithms(True)
  seed = 42
  mlen = 150

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Vanilla Sampling -----------")
  print(sample(lm, text_field, prompt=p, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 0.0001 -------")
  print(sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 100 --------")
  print(sample(lm, text_field, prompt=p, temp=100, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 1 -----------")
  print(sample(lm, text_field, prompt=p, k=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 20 -----------")
  print(sample(lm, text_field, prompt=p, k=20, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.001 -----------")
  print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.75 -----------")
  print(sample(lm, text_field, prompt=p, p=0.75, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 1 -----------")
  print(sample(lm, text_field, prompt=p, p=1, max_len=mlen))


  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=1 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=10 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=10, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=50 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=50, max_len=mlen))

  print()

############################################################################################
# TASK 2.1
############################################################################################

def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
  decodedString = "Not implemented"

  # Model Setting
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  hidden_size = 512
  num_layers = 3
  batch_size = 1

  # Initialization
  p_tokens = text_field.process([text_field.tokenize(prompt.lower())]).to(dev)
  h_0 = torch.zeros(num_layers, batch_size, hidden_size).to(dev)  # initial hidden state
  c_0 = torch.zeros(num_layers, batch_size, hidden_size).to(dev)  # initial cell state

  # Process for Beam Search
  token = p_tokens.squeeze(-1)
  # Initialize the beam with the prompt
  beam = [(token, 0, token, h_0, c_0)]
  for _ in range(max_len):
    candidates = []
    for seq, score, x, h, c in beam:
      # Forward pass
      out, h_t, c_t = model(x.unsqueeze(-1), h, c)
      # Log-Softmax to get log probability distribution
      prob = F.log_softmax(out[-1], dim=-1).squeeze()
      # Get the top beam candidates
      topk_prob, indices = torch.topk(prob, beams, dim=-1)

      # Expansion stage to collect candidates
      for b in range(beams):
        next_token = indices[b]
        next_prob = topk_prob[b]
        new_seq = torch.cat((seq, next_token.unsqueeze(-1)), dim=0)
        new_score = score + next_prob
        candidates.append((new_seq, new_score, next_token.unsqueeze(-1), h_t, c_t))

    # Sort candidates by score
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    # Retain the top beams
    beam = candidates[:beams]

  # Return the sequence with the highest log probability
  best_seq, _, _, _, _ = max(beam, key=lambda x: x[1])
  decodedString = reverseNumeralize(best_seq, text_field)

  return decodedString

############################################################################################
# TASK 1.1
############################################################################################

def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1):
  assert (k==0 or p==1), "Cannot combine top-k and top-p sampling"
  decodedString = "Not implemented"

  # Model Setting
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  hidden_size = 512
  num_layers = 3
  batch_size = 1

  # Initialization
  p_tokens = text_field.process([text_field.tokenize(prompt.lower())]).to(dev)
  x = p_tokens
  h_t = torch.zeros(num_layers, batch_size, hidden_size).to(dev)  # initial hidden state
  c_t = torch.zeros(num_layers, batch_size, hidden_size).to(dev)  # initial cell state

  # Process to generate word tokens
  tokens = p_tokens.squeeze(-1).tolist()
  for _ in range(max_len):
    # Forward pass
    out, h_t, c_t = model(x, h_t, c_t)
    # Vanilla Sampling (temp = 1) & Temperature-scaled Sampling
    out = out / temp
    # Get the last output from sequence
    last_out = out[-1]
    # Softmax to get probability distribution
    prob = F.softmax(last_out, dim=-1).squeeze()

    # Top-k Sampling
    if k > 0:
      topk_prob, indices = torch.topk(prob, k=k)
      prob.zero_()
      prob[indices] = topk_prob

    # Nucleus (top-p) Sampling (Vanilla: p = 1)
    elif p < 1:
      # Sort probability by descending order
      sort_prob, sort_indices = torch.sort(prob, descending=True)
      torch.use_deterministic_algorithms(False)
      # Compute for CMF
      CMF = torch.cumsum(sort_prob, dim=-1)
      torch.use_deterministic_algorithms(True)

      # Find the index where the cumulative probability exceeds p
      cutoff_index = (CMF > p).nonzero().min()
      # Zero out all probabilities beyond this index
      if cutoff_index is not None:
        prob[sort_indices[cutoff_index+1:]] = 0

    # Normalize the probability
    prob = prob / torch.sum(prob, dim=-1)
    # Sampling based on the given probability
    next_token = torch.multinomial(prob, 1)

    # Adding the generated token into the token list
    tokens.append(next_token)
    x = torch.tensor([[next_token]]).to(dev)

  decodedString = reverseNumeralize(tokens, text_field)

  return decodedString

############################################################################################

def reverseNumeralize(numeralized_string, text_field):
  strings = [text_field.vocab.itos[i] for i in numeralized_string]
  return " ".join(strings)

if __name__ == "__main__":
  main()
