#!/bin/bash
# overnight queue, most important first; each condition resumes from its last finished chunk
cd ~/wsp && . venv/bin/activate
R=results/math500
run() { echo "$(date -u +%FT%TZ) START $2" >> run_log.txt; python scripts/run_cond.py ckpt/$1 $R/$2 --chunk 250 "${@:3}" >> run_log.txt 2>&1; echo "$(date -u +%FT%TZ) END $2" >> run_log.txt; }
run p50c p50c
run base base_rp110 --rep 1.1
run base base
run p25c p25c
run p10c p10c
run p50c p50c_rp110 --rep 1.1
run p50c_a050 p50c_a050
run p50c_a150 p50c_a150
run base base_rp105 --rep 1.05
run base base_rp120 --rep 1.2
echo "$(date -u +%FT%TZ) ALL_DONE" >> run_log.txt
