#!/bin/bash
# Two-center guard closed-loop A/B: baseline / guard / guard+hysteresis
# across nominal/high-speed/recovery scenarios and three matched seeds.
# Hysteresis first-round parameters (pre-registered, untuned on this data):
# switch_margin=1.0 model-cost units (~3% of the mature mean selected cost,
# ~40% of the mean immediate gain in the 11.9 pilot), min_dwell=5 steps.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anycar
source /opt/ros/humble/setup.bash
source /home/plusai/anycar/install/setup.bash
source /home/plusai/anycar/set_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
cd /home/plusai/anycar

ROOT=/disk/collect_data_from_anycar/mppi_rl_closed_loop/guard_hysteresis_ab_20260818_v1
CKPT=/home/plusai/anycar/outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/direct_residual_online_ac_selected.pt
mkdir -p "$ROOT"

for seed in 3407 3411 3413; do
for pair in "nominal|1.6|0,0,0,0,0,0" "high|2.8|0,0,0,0,0,0" "recovery|2.8|0,0.3,0.3,0,0,0"; do
  IFS='|' read -r name speed init <<< "$pair"
  for arm in baseline guard guard_hyst; do
    id="${arm}_${name}_s${seed}"
    dir="$ROOT/$id"
    if [ -d "$dir" ] && [ -f "$dir/closed_loop_trace.jsonl" ]; then
      echo "skip existing $id"
      continue
    fi
    extra=""
    if [ "$arm" != "baseline" ]; then
      extra="mppi_hard_guard_checkpoint:=$CKPT"
    fi
    if [ "$arm" = "guard_hyst" ]; then
      extra="$extra mppi_hard_guard_switch_margin:=1.0 mppi_hard_guard_min_dwell:=5"
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
      mppi_dataset_max_snapshots:=6 \
      mppi_dataset_shutdown_on_complete:=true \
      $extra > "$ROOT/${id}.launch.log" 2>&1
    code=$?
    rows=0
    [ -f "$dir/closed_loop_trace.jsonl" ] && rows=$(wc -l < "$dir/closed_loop_trace.jsonl")
    echo "$id exit=$code rows=$rows"
  done
done
done
echo "ALL DONE"
