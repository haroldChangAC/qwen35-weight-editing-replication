"""One condition on one benchmark: the authors' exp8 think35 protocol (their prompt, chat template, grader),
with the decoding penalties exposed. Resumable: one gzip per chunk of problems; every sample is kept."""
import argparse, gzip, json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/wsp/logs/exp8_math/code"))
sys.path.insert(0, os.path.expanduser("~/wsp/scripts/shim"))
import mathqa as Q

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--set", default="math500")
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--rep", type=float, default=1.0)
ap.add_argument("--presence", type=float, default=1.5)
ap.add_argument("--max_tokens", type=int, default=32768)
ap.add_argument("--chunk", type=int, default=100)
ap.add_argument("--mem", type=float, default=0.90)
a = ap.parse_args()

rows = [json.loads(l) for l in open(os.path.expanduser(f"~/wsp/bench/{a.set}.jsonl"))]
if a.limit:
    rows = rows[:a.limit]
os.makedirs(a.out, exist_ok=True)
sampling = dict(n=a.n, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=a.presence,
                repetition_penalty=a.rep, max_tokens=a.max_tokens, seed=0)
json.dump({"args": vars(a), "sampling": sampling, "instruction": Q.INSTRUCTION, "thinking": True},
          open(os.path.join(a.out, "config.json"), "w"), indent=1)
todo = [s for s in range(0, len(rows), a.chunk) if not os.path.exists(os.path.join(a.out, f"chunk_{s:04d}.jsonl.gz"))]
if not todo:
    print("ALREADY DONE", a.out); sys.exit(0)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt)
llm = LLM(model=a.ckpt, dtype="bfloat16", gpu_memory_utilization=a.mem, max_model_len=a.max_tokens + 2048, seed=0)
sp = SamplingParams(**sampling)
for start in todo:
    fn = os.path.join(a.out, f"chunk_{start:04d}.jsonl.gz")
    part = rows[start:start + a.chunk]
    prompts = [tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(r["problem"])}],
               tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in part]
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    ntok = 0
    with gzip.open(fn + ".tmp", "wt") as f:
        for r, o in zip(part, outs):
            for j, s in enumerate(o.outputs):
                pred, _ = Q.extract(s.text)
                f.write(json.dumps({"id": r["id"], "sample": j, "gold": r["answer"], "pred": pred,
                                    "correct": int(Q.is_correct(pred, r["answer"])),
                                    "finish": s.finish_reason, "ntok": len(s.token_ids), "text": s.text},
                                   ensure_ascii=False) + "\n")
                ntok += len(s.token_ids)
    os.replace(fn + ".tmp", fn)
    dt = time.time() - t0
    print(f"{a.out} chunk {start}: {dt:.0f}s {ntok / dt:.0f} tok/s", flush=True)
print("DONE", a.out, flush=True)
