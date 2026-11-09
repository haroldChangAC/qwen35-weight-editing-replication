#!/bin/bash
# 2026-11-06: fill the alpha x repetition-penalty grid (5 missing cells) for the surface/heatmap figure
cd ~/wsp && . venv/bin/activate
R=results/math500
run() { echo "$(date -u +%FT%TZ) START $2" >> run_log.txt; python scripts/run_cond.py ckpt/$1 $R/$2 --chunk 250 "${@:3}" >> run_log.txt 2>&1; echo "$(date -u +%FT%TZ) END $2" >> run_log.txt; }
run p50c p50c_rp105 --rep 1.05
run p50c_a150 p50c_a150_rp105 --rep 1.05
run p50c_a150 p50c_a150_rp110 --rep 1.1
run p50c_a050 p50c_a050_rp105 --rep 1.05
run p50c_a050 p50c_a050_rp110 --rep 1.1
echo "$(date -u +%FT%TZ) GRID_DONE" >> run_log.txt
