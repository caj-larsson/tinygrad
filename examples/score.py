#!/usr/bin/env python3
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
from tinygrad.helpers import Timing, GlobalCounters, getenv, DEBUG, dtypes, CI
from llama import LLaMa
from tinygrad.jit import TinyJit, JIT_SUPPORTED_DEVICE
import functools, sys, argparse, json, os
from pathlib import Path
from tqdm import tqdm
import json

JIT = getenv("JIT", 0 if CI else int(Device.DEFAULT in JIT_SUPPORTED_DEVICE))

class BasicScore:
  def __init__(self, llama: LLaMa, temperature=0.2):
    self.llama = llama
    self.temperature = temperature

  def score_file(self, path, ctx_len=1024 ):
    window_context = ctx_len // 20

    with open(path, 'r') as f:
      text = f.read()
    text_enc = self.llama.tokenizer.encode(text)

    progress_bar = tqdm(total=len(text_enc), unit_scale=True, desc=f"Scoring {path}:")
    start = 0
    token_probs = []
    context_enc = []
    while True:
      end = start + ctx_len - len(context_enc)
      pps = self.score_text(text_enc[start:end], progress_bar, context_enc)
      token_probs += pps
      start += len(pps)
      context_enc = text_enc[end-window_context:end]

      if start >= len(text_enc):
        break
    return token_probs


  def score_text(self, text_enc, progress_bar, context=None):
    pos = 0
    toks = [self.llama.tokenizer.bos_id()]
    if context:
      toks += context

    next_probs = self.llama.model(Tensor([toks]), pos, self.temperature)
    pos = len(toks)

    token_probs = []
    for tok in text_enc:
      next_probs_np = next_probs.numpy()
      tok_decode = self.llama.tokenizer.id_to_piece(tok).replace("▁", " ")
      token_probs.append((tok_decode, float(next_probs_np[tok])))
      progress_bar.update(1)
      toks.append(tok)
      next_probs = llama.model(Tensor([toks[pos:]]), pos, self.temperature).realize()
      pos = len(toks)

    return token_probs

if __name__ == "__main__":
  Tensor.no_grad = True
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description="BasicScore a file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--temperature", type=float, default=0.2, help="Temperature in the softmax")
  parser.add_argument("--size", type=str, default="7B", help="Size of model to use [7B, 13B, 30B, 65B] for Gen 1, [7B, 13B, 70B] for Gen 2, [7B, 13B, 34B] for Code LLaMA")
  parser.add_argument("--model", type=Path, default=None, help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file")
  parser.add_argument("files", type=Path, nargs="+", default=None, help="File to score")
  parser.add_argument("--token-output", type=Path, default=None, help="Output json data with all token probabilities")

  args = parser.parse_args()

  if args.files is None:
    print("Please add --files")
    exit(1)

  MODEL_PATH = args.model or Path(__file__).parents[1] / f"weights/LLaMA-code/{args.size}"
  TOKENIZER_PATH = (MODEL_PATH if MODEL_PATH.is_dir() else MODEL_PATH.parent) / "tokenizer.model"
  print(f"using LLaMA-code-{args.size} model")
  llama = LLaMa.build(MODEL_PATH, TOKENIZER_PATH, model_gen="code", model_size=args.size, quantize=False)

  bs = BasicScore(llama)

  file_scores = {}
  file_token_probs = {}

  for filename in args.files:
    p = bs.score_file(filename)
    s = sum(x[1] for x in p) / len(p)
    file_token_probs[str(filename)] = p
    file_scores[str(filename)] = s
    print(f"{filename}: {s * 100:0.2f}")
  print(file_scores)
  if args.token_output:
    with open(args.token_output, "w") as f:
      json.dump({"scores": file_scores, "tokens": file_token_probs}, f)
