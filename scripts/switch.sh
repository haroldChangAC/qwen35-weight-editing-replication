#!/bin/bash
# when the running queue logs "END base", stop it (it would start p25c next) and hand over to run_all2.sh
cd ~/wsp
while ! grep -q " END base$" run_log.txt; do sleep 20; done
tmux kill-session -t queue
sleep 5; pkill -f scripts/run_cond.py; sleep 10
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
echo "$(date -u +%FT%TZ) SWITCH to run_all2.sh" >> run_log.txt
bash scripts/run_all2.sh
