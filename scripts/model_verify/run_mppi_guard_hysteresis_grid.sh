#!/bin/bash
# Guard hysteresis margin/dwell grid on top of the 2026-08-18 A/B.
# Completes the 3x3 grid (margin 0.5/1.0/2.0 x dwell 3/5/8); the (1.0, 5)
# cell is the already-collected guard_hyst arm and is not rerun. Baseline
# and plain-guard arms are reused from the parent collection.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anycar
source /opt/ros/humble/setup.bash
source /home/plusai/anycar/install/setup.bash
source /home/plusai/anycar/set_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
cd /home/plusai/anycar

ROOT=/disk/collect_data_from_anycar/mppi_rl_closed_loop/guard_hysteresis_ab_20260818_v1
CKPT=/home/plusai/anycar/outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/direct_residual_online_ac_selected.pt

for pair in "0.5|3" "0.5|5" "0.5|8" "2.0|3" "2.0|5" "2.0|8"; do
  IFS='|' read -r margin dwell <<< "$pair"
  for seed in 3407 3411 3413; do
  for scen in "nominal|1.6|0,0,0,0,0,0" "high|2.8|0,0,0,0,0,0" "recovery|2.8|0,0.3,0.3,0,0,0"; do
    IFS='|' read -r name speed init <<< "$scen"
    id="ghist_m${margin}_d${dwell}_${name}_s${seed}"
    dir="$ROOT/$id"
    if [ -d "$dir" ] && [ -f "$dir/closed_loop_trace.jsonl" ]; then
      echo "skip existing $id"
      continue
    fi
    echo "=== running $id ==="
    timeout 900 ros2 launch car_ros2 car_sim.launch.py \
      mppi_backend:=dbm \
      mppi_seed:=$seed \
      mppi_reference_speed:=$speed \
      sim_initial_state:=$init \
      mppi_dataset_dir:=$ROOT \
      mppi_dataset_episode_id:=$id \
      mppi_dataset_start_step:=250 \
      mppi_dataset_stop_step:=-1 \
      mppi_dataset_stride:=10 \
      mppi_dataset_max_snapshots:=6 \
      mppi_dataset_shutdown_on_complete:=true \
      mppi_hard_guard_checkpoint:=$CKPT \
      mppi_hard_guard_switch_margin:=$margin \
      mppi_hard_guard_min_dwell:=$dwell \
      > "$ROOT/${id}.launch.log" 2>&1
    code=$?
    rows=0
    [ -f "$dir/closed_loop_trace.jsonl" ] && rows=$(wc -l < "$dir/closed_loop_trace.jsonl")
    echo "$id exit=$code rows=$rows"
  done
  done
done
echo "GRID DONE"
