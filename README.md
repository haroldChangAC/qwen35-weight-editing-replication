# Independent replication of LLM-guided weight editing on Qwen3.5-0.8B

Code, per-problem outputs and analysis for the paper
**"An Independent Replication of LLM-Guided Weight Editing on Qwen3.5-0.8B: Decoding Baselines, Dose–Response, and the Edit × Penalty Interaction"** (JingHao Chang, Chonnam National University; preprint forthcoming).

The paper replicates the Qwen3.5-0.8B / MATH-500 experiment of Takehi, Teoh, Zhang and Khattab,
[*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, November 2026),
using the authors' published weight differences ([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)),
and adds decoding baselines, an edit-strength dose–response, and a full edit-strength × repetition-penalty grid.

## Main results

MATH-500, all 500 problems, one sample per problem, the authors' prompt, grader and sampling settings
(thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, max 32,768 new tokens, seed 0).
95% CIs are percentile bootstrap intervals over problems (2,000 resamples, seed 0).

| Condition | Directory | α | RP | Accuracy (%) [95% CI] | Hit limit (%) | Loop (%) | No answer (%) |
|---|---|---|---|---|---|---|---|
| Original | `base` | 0 | 1.00 | 6.0 [4.2, 8.0] | 94.4 | 91.4 | 0.0 |
| Original + RP 1.05 | `base_rp105` | 0 | 1.05 | 16.6 [13.4, 20.0] | 83.6 | 79.4 | 0.0 |
| Original + RP 1.10 | `base_rp110` | 0 | 1.10 | 22.8 [19.2, 26.6] | 75.0 | 66.8 | 1.6 |
| Original + RP 1.20 | `base_rp120` | 0 | 1.20 | 1.8 [0.8, 3.0] | 70.4 | 42.4 | 17.6 |
| Edit p10c | `p10c` | – | 1.00 | 13.2 [10.2, 16.2] | 87.2 | 83.4 | 0.0 |
| Edit p25c | `p25c` | – | 1.00 | 13.2 [10.4, 16.2] | 86.8 | 84.6 | 0.4 |
| Edit p50c | `p50c` | 1.0 | 1.00 | 51.8 [47.6, 56.2] | 30.8 | 15.6 | 7.0 |
| p50c + RP 1.05 | `p50c_rp105` | 1.0 | 1.05 | 45.6 [41.2, 50.2] | 42.4 | 8.0 | 3.4 |
| p50c + RP 1.10 | `p50c_rp110` | 1.0 | 1.10 | 40.4 [36.2, 44.8] | 44.6 | 5.6 | 7.6 |
| ×0.5 | `p50c_a050` | 0.5 | 1.00 | 35.8 [31.6, 40.0] | 61.0 | 55.2 | 1.0 |
| ×0.5 + RP 1.05 | `p50c_a050_rp105` | 0.5 | 1.05 | 41.4 [37.2, 45.8] | 53.4 | 39.2 | 1.8 |
| ×0.5 + RP 1.10 | `p50c_a050_rp110` | 0.5 | 1.10 | 39.8 [35.6, 44.2] | 50.2 | 25.2 | 4.0 |
| ×1.5 | `p50c_a150` | 1.5 | 1.00 | 48.6 [44.2, 53.0] | 24.8 | 6.4 | 12.4 |
| ×1.5 + RP 1.05 | `p50c_a150_rp105` | 1.5 | 1.05 | 41.2 [37.0, 45.8] | 34.0 | 2.2 | 14.6 |
| ×1.5 + RP 1.10 | `p50c_a150_rp110` | 1.5 | 1.10 | 40.2 [36.0, 44.4] | 38.8 | 0.8 | 7.6 |

α = edit strength relative to the p50c weight difference (scaled weight = b + α(e − b) on the tensors p50c changes);
RP = repetition penalty. The authors reported 7.9 / 12.3 / 14.1 / 48.1% accuracy for Original / p10c / p25c / p50c
(mean of four samples); all four fall inside our intervals.

In short: the published edit replicates; the best repetition penalty on the original weights reaches only 22.8%;
full edit strength is optimal; and on edited weights every penalty lowers accuracy
(all six edit × penalty interaction contrasts are negative, down to −28.2 pp [−33.8, −23.0]).

## Repository layout

```
scripts/                 everything that ran on the GPU machine, unchanged
  prep_bench.py          writes bench/math500.jsonl (and gsm8k.jsonl, unused) from Hugging Face datasets
  build_ckpts.py         text-only base checkpoint, the authors' p10c/p25c/p50c edits, p50c scaled by α = 0.5, 1.5
  run_cond.py            one condition: vLLM generation + the authors' grader; resumable, one gzip per chunk
  run_all.sh, run_all2.sh, switch.sh, run_grid.sh   the run queues, in the order they were used
  shim/mcqa.py           our reimplementation of the authors' unreleased loop rule (diagnostics only)
  summarize.py, analyze.py                          per-condition summaries and McNemar tests (first ten conditions)
  export.py              export/per_problem.csv and outcome counts
  fig_data.py            level/subject accuracy, overlaps, outcome transitions, token quantiles
  grid_stats.py          all 15 conditions: accuracy + CI, McNemar tests, paired-bootstrap penalty effects and interactions
  grid_dump.py           compact per-problem outcome strings
  lenient.py             lenient re-grading of unanswered outputs (paper Appendix A)
  pick_samples.py        random representative outputs per outcome category (paper Table 4)
results/math500/<condition>/
  config.json            exact arguments and sampling parameters of the run
  chunk_0000.jsonl.gz, chunk_0250.jsonl.gz   one JSON line per problem (see below)
export/                  analysis outputs used for the paper's tables and figures
figures/src/             plotting scripts and the numbers they plot (with consistency asserts)
figures/out/             rendered figures (PNG 300 dpi and PDF)
logs/                    run log (summary and full vLLM log)
environment/             exact package versions of the generation environment
```

### Output format

Each line of `results/math500/<condition>/chunk_*.jsonl.gz` is one generated sample:

| Field | Meaning |
|---|---|
| `id` | MATH-500 `unique_id`, e.g. `test/algebra/1282.json` |
| `sample` | sample index (always 0: one sample per problem) |
| `gold` | reference answer |
| `pred` | content of the last `\boxed{}` in the output, or null |
| `correct` | 1 if the authors' grader (math-verify) accepts `pred` |
| `finish` | `stop` (finished) or `length` (hit 32,768 tokens) |
| `ntok` | number of generated tokens |
| `text` | full generated text, including the thinking part |

`export/per_problem.csv` has one row per condition and problem with `correct`, `finish`, `ntok`, `loop`
(either loop rule fired) and `outcome` (`correct`, `wrong`, `no_answer`, `cut_noloop`, `cut_loop`).

### Figures in the paper

| Paper | File |
|---|---|
| Figure 1 | `figures/out/fig1_pipeline` |
| Figure 2 | `figures/out/fig2_penalty_sweep` |
| Figure 3 | `figures/out/fig9_alpha_penalty_grid` |
| Figure 4 | `figures/out/fig4_outcomes` |
| Figure 5 | `figures/out/fig5_length_cdf` |
| Figure 6 | `figures/out/fig6_alluvial` |
| Figure 7 | `figures/out/fig7_cosolving` |
| Figure 8 | `figures/out/fig8_level_subject` |
| Figure B1 | `figures/out/figS1_alpha_penalty_surface` |

`make_figs.py` also draws an early dose–response plot (`fig3_dose_response`) that the paper no longer uses;
its content is part of Figure 3.

## Reproducing

The scripts were run as-is from a working directory `~/wsp` and use that path. Hardware: one NVIDIA L4 (24 GB),
Google Cloud `g2-standard-4`, Ubuntu 22.04, Python 3.12, vLLM 0.33.0, PyTorch 2.14.0, Transformers 5.20.2,
math-verify 0.9.0 (full list in `environment/`). Generation took 61 GPU-hours for the 15 conditions.

```bash
mkdir -p ~/wsp && cd ~/wsp          # put the contents of this repository here
python -m venv venv && . venv/bin/activate
pip install -r environment/requirements-lock.txt

# 1. the original weights, pinned revision
huggingface-cli download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir qwen35_release

# 2. the authors' weight differences, prompt, grader and text-only model config
git clone https://github.com/RikiyaT/lm_understand-logs logs        # provides logs/exp8_math/{code,runs,base_model}

# 3. benchmark, checkpoints (SHA-256 of the edits is checked by the authors' apply_edit.py)
python scripts/prep_bench.py
python scripts/build_ckpts.py

# 4. generation (long); each line of the queues is one condition
bash scripts/run_all.sh      # original model, the three edits, α = 0.5 / 1.5, penalties
bash scripts/run_grid.sh     # the remaining edit × penalty cells

# 5. analysis
python scripts/export.py
python scripts/grid_stats.py
python scripts/fig_data.py
python scripts/lenient.py
python scripts/pick_samples.py
```

To redraw the figures, run `python make_figs.py`, `make_figs2.py` and `make_figs3.py` inside `figures/src/`
(matplotlib, networkx; Liberation Serif font).

## Notes and limitations

- One sample per problem (the authors used four), seed 0. Generation with the same versions and hardware should be
  close but is not guaranteed to be bit-identical.
- Grading follows the authors exactly: only the last `\boxed{}` counts. `scripts/lenient.py` shows that accepting a
  final "Final Answer: …" line changes no conclusion (at most +2.2 points per condition).
- Loop detection combines the authors' verbatim rule with our reimplementation of their second rule
  (`scripts/shim/mcqa.py`). Both look only at the end of an output and miss long-period cycles, so loop shares are
  lower bounds. Loop labels are diagnostics and never affect grading.
- We tested the published edits; we did not rerun the editing agent.

## Citation

If you use this repository, please cite the original work and this replication:

```
R. Takehi, J. Teoh, A. Zhang, and O. Khattab. Sparks of Weight-Space Programming: LLMs can natively
read & write neural network weights. Blog post, MIT CSAIL, November 2026.
https://rikiyat.github.io/blog/weight-space-programming/

J. Chang. An Independent Replication of LLM-Guided Weight Editing on Qwen3.5-0.8B: Decoding Baselines,
Dose–Response, and the Edit × Penalty Interaction. Preprint, 2026 (link to follow).
```

## License

Code: MIT License (see `LICENSE`). Model outputs and analysis files: CC BY 4.0.
The original weights (Qwen3.5-0.8B) and the authors' weight differences are not redistributed here; obtain them from
their sources above under their own licenses.
