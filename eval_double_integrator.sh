#!/bin/bash
experiment_name_str="double_integrator_safety_informed"

# "double_integrator" or "airtaxi"
dynamics_type="double_integrator"
# for custom scenario, use "navigation_graph_safe_eval" and check the last line of multiagent/config.py to see what custom scenario is used
scenario_name="navigation_graph_safe"
use_safety_filter="True"
world_size=4
# Read last line of config.py and echo
config_file="multiagent/config.py"
original_string=$(tail -n 1 $config_file)
model_dir_str="trained_models/${experiment_name_str}"

num_eval_episodes=3
num_eval_agents=4
eval_episode_length=250

# Optional overrides via CLI args:
# ./eval_double_integrator.sh [run1_rho run2_rho run3_rho run4_rho run5_rho]
run1_rho="${1:-0.00}"
run2_rho="${2:-0.02}"
run3_rho="${3:-0.05}"
run4_rho="${4:-0.10}"
run5_rho="${5:-0.20}"

echo "Running ${num_eval_episodes} episodes with ${num_eval_agents} agents and episode length ${eval_episode_length}."
echo "Run 1: SAFETY_STATE_UNCERTAINTY_RADIUS=${run1_rho}"
echo "Run 2: SAFETY_STATE_UNCERTAINTY_RADIUS=${run2_rho}"
echo "Run 3: SAFETY_STATE_UNCERTAINTY_RADIUS=${run3_rho}"
echo "Run 4: SAFETY_STATE_UNCERTAINTY_RADIUS=${run4_rho}"
echo "Run 5: SAFETY_STATE_UNCERTAINTY_RADIUS=${run5_rho}"

python scripts/eval_mpe.py \
--model_dir=${model_dir_str} \
--dynamics_type ${dynamics_type} --render_episodes=${num_eval_episodes} \
--world_size=${world_size} --num_landmarks=2 \
--num_agents=${num_eval_agents} \
--num_obstacles=0 \
--seed=0 \
--episode_length=${eval_episode_length} \
--use_dones=False --collaborative=False \
--scenario_name=${scenario_name} --horizon=1 --save_gifs --use_render --num_walls=0 \
--discrete_action=True \
--use_masking "True" \
--use_safety_filter ${use_safety_filter} \
--safety_state_uncertainty_radius ${run1_rho}

python scripts/eval_mpe.py \
--model_dir=${model_dir_str} \
--dynamics_type ${dynamics_type} --render_episodes=${num_eval_episodes} \
--world_size=${world_size} --num_landmarks=2 \
--num_agents=${num_eval_agents} \
--num_obstacles=0 \
--seed=0 \
--episode_length=${eval_episode_length} \
--use_dones=False --collaborative=False \
--scenario_name=${scenario_name} --horizon=1 --save_gifs --use_render --num_walls=0 \
--discrete_action=True \
--use_masking "True" \
--use_safety_filter ${use_safety_filter} \
--safety_state_uncertainty_radius ${run2_rho}

python scripts/eval_mpe.py \
--model_dir=${model_dir_str} \
--dynamics_type ${dynamics_type} --render_episodes=${num_eval_episodes} \
--world_size=${world_size} --num_landmarks=2 \
--num_agents=${num_eval_agents} \
--num_obstacles=0 \
--seed=0 \
--episode_length=${eval_episode_length} \
--use_dones=False --collaborative=False \
--scenario_name=${scenario_name} --horizon=1 --save_gifs --use_render --num_walls=0 \
--discrete_action=True \
--use_masking "True" \
--use_safety_filter ${use_safety_filter} \
--safety_state_uncertainty_radius ${run3_rho}

python scripts/eval_mpe.py \
--model_dir=${model_dir_str} \
--dynamics_type ${dynamics_type} --render_episodes=${num_eval_episodes} \
--world_size=${world_size} --num_landmarks=2 \
--num_agents=${num_eval_agents} \
--num_obstacles=0 \
--seed=0 \
--episode_length=${eval_episode_length} \
--use_dones=False --collaborative=False \
--scenario_name=${scenario_name} --horizon=1 --save_gifs --use_render --num_walls=0 \
--discrete_action=True \
--use_masking "True" \
--use_safety_filter ${use_safety_filter} \
--safety_state_uncertainty_radius ${run4_rho}

python scripts/eval_mpe.py \
--model_dir=${model_dir_str} \
--dynamics_type ${dynamics_type} --render_episodes=${num_eval_episodes} \
--world_size=${world_size} --num_landmarks=2 \
--num_agents=${num_eval_agents} \
--num_obstacles=0 \
--seed=0 \
--episode_length=${eval_episode_length} \
--use_dones=False --collaborative=False \
--scenario_name=${scenario_name} --horizon=1 --save_gifs --use_render --num_walls=0 \
--discrete_action=True \
--use_masking "True" \
--use_safety_filter ${use_safety_filter} \
--safety_state_uncertainty_radius ${run5_rho}