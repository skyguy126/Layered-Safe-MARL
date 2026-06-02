#!/bin/bash

# --- Experiment / model ---
experiment_name="double_integrator_safety_informed"
model_dir="trained_models/${experiment_name}"

# --- Dynamics & scenario ---
# "double_integrator" or "airtaxi"
dynamics_type="double_integrator"
# For custom scenario, use "navigation_graph_safe_eval" and check the last line of multiagent/config.py
scenario_name="navigation_graph_safe_eval"

# --- World layout ---
world_size=4
num_landmarks=3
num_obstacles=1
num_walls=0

# --- Evaluation run ---
num_eval_episodes=50
num_eval_agents=6
eval_episode_length=500
seed=0
horizon=1

# --- Agent / episode behavior ---
discrete_action="True"
use_masking="True"
use_dones="False"
collaborative="False"

# --- Safety filter ---
use_safety_filter="True"

# --- Packet uncertainty ---
enable_packet_uncertainty="False"
packet_loss_prob=0.1 # this is good
vmax_uncertainty=1.0
packet_loss_burst_len=20 # can also be 5 for more realistic results
warmup_steps=25
safety_filter_uncertainty_mode="nominal" # nominal | fixed_lcb | lipschitz_lcb
fixed_lcb_margin=0.12
lcb_lipschitz_const=1.0

# --- Rendering (set flag to empty string to disable) ---
save_gif_flag="--save_gifs"
use_render_flag="--use_render"
# Optional JSON report output (set empty to disable)
stats_json_output="baseline_0_0.json"

# Sanity check: last line of config.py should match the custom scenario in use
config_file="multiagent/config.py"
echo "config.py scenario line: $(tail -n 1 "${config_file}")"

echo "Running ${num_eval_episodes} episodes with ${num_eval_agents} agents and episode length ${eval_episode_length}."

python scripts/eval_mpe.py \
  --model_dir="${model_dir}" \
  --dynamics_type="${dynamics_type}" \
  --scenario_name="${scenario_name}" \
  --world_size="${world_size}" \
  --num_landmarks="${num_landmarks}" \
  --num_obstacles="${num_obstacles}" \
  --num_walls="${num_walls}" \
  --num_agents="${num_eval_agents}" \
  --render_episodes="${num_eval_episodes}" \
  --episode_length="${eval_episode_length}" \
  --seed="${seed}" \
  --horizon="${horizon}" \
  --discrete_action="${discrete_action}" \
  --use_masking="${use_masking}" \
  --use_dones="${use_dones}" \
  --collaborative="${collaborative}" \
  --use_safety_filter="${use_safety_filter}" \
  --enable_packet_uncertainty="${enable_packet_uncertainty}" \
  --packet_loss_prob="${packet_loss_prob}" \
  --vmax_uncertainty="${vmax_uncertainty}" \
  --packet_loss_burst_len="${packet_loss_burst_len}" \
  --warmup_steps="${warmup_steps}" \
  --safety_filter_uncertainty_mode="${safety_filter_uncertainty_mode}" \
  --fixed_lcb_margin="${fixed_lcb_margin}" \
  --lcb_lipschitz_const="${lcb_lipschitz_const}" \
  ${stats_json_output:+--stats_json_output="${stats_json_output}"} \
  ${save_gif_flag} \
  ${use_render_flag}
