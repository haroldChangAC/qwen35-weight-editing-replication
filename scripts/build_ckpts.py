"""Build text-only checkpoints: base, the authors' p10c/p25c/p50c edits (exact diffs), and p50c scaled by alpha."""
import os, shutil, subprocess, sys
import torch
from safetensors.torch import load_file, save_file
H = os.path.expanduser("~/wsp")
REL = f"{H}/qwen35_release"
TXT = f"{H}/logs/exp8_math/base_model/Qwen3.5-0.8B-text"
CODE = f"{H}/logs/exp8_math/code"
RUNS = f"{H}/logs/exp8_math/runs"
CK = f"{H}/ckpt"

def meta(out):
    os.makedirs(out, exist_ok=True)
    for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        shutil.copy(f"{REL}/{f}", out)
    for f in os.listdir(TXT):
        shutil.copy(f"{TXT}/{f}", out)

base = f"{CK}/base/model.safetensors"
if not os.path.exists(base):
    meta(f"{CK}/base")
    sd = load_file(f"{REL}/model.safetensors-00001-of-00001.safetensors")
    keep = {k: v for k, v in sd.items() if k.startswith("model.language_model.")}
    save_file(keep, base, metadata={"format": "pt"})
    print("base:", len(keep), "tensors")

for run in ["p10c", "p25c", "p50c"]:
    out = f"{CK}/{run}/model.safetensors"
    if not os.path.exists(out):
        meta(f"{CK}/{run}")
        subprocess.run([sys.executable, f"{CODE}/apply_edit.py", base, f"{RUNS}/{run}/edit", out], check=True)
        print(run, "built")

B = load_file(base)
E = load_file(f"{CK}/p50c/model.safetensors")
changed = [k for k in B if not torch.equal(B[k], E[k])]
print("p50c changed tensors:", changed)
for a in [0.5, 1.5]:
    name = f"p50c_a{int(a*100):03d}"
    out = f"{CK}/{name}/model.safetensors"
    if os.path.exists(out):
        continue
    meta(f"{CK}/{name}")
    sd = dict(B)
    for k in changed:
        b, e = B[k].float(), E[k].float()
        sd[k] = (b + a * (e - b)).to(torch.bfloat16).contiguous()
    save_file(sd, out, metadata={"format": "pt"})
    print(name, "built")
