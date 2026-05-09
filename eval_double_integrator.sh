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
# ./eval_double_integrator.sh [run1_std run1_bias ... run5_std run5_bias run1_rho run2_rho run3_rho run4_rho run5_rho]
run1_std="${1:-0.00}"; run1_bias="${2:-0.00}"; run1_rho="${11:-0.0}"
run2_std="${3:-0.03}"; run2_bias="${4:-0.00}"; run2_rho="${12:-0.0}"
run3_std="${5:-0.03}"; run3_bias="${6:-0.01}"; run3_rho="${13:-0.0}"
run4_std="${7:-0.06}"; run4_bias="${8:-0.00}"; run4_rho="${14:-0.0}"
run5_std="${9:-0.06}"; run5_bias="${10:-0.01}"; run5_rho="${15:-0.0}"

echo "Running ${num_eval_episodes} episodes with ${num_eval_agents} agents and episode length ${eval_episode_length}."
echo "Run 1: SAFETY_VALUE_NOISE_STD=${run1_std}, SAFETY_VALUE_NOISE_BIAS=${run1_bias}, SAFETY_STATE_UNCERTAINTY_RADIUS=${run1_rho}"
echo "Run 2: SAFETY_VALUE_NOISE_STD=${run2_std}, SAFETY_VALUE_NOISE_BIAS=${run2_bias}, SAFETY_STATE_UNCERTAINTY_RADIUS=${run2_rho}"
echo "Run 3: SAFETY_VALUE_NOISE_STD=${run3_std}, SAFETY_VALUE_NOISE_BIAS=${run3_bias}, SAFETY_STATE_UNCERTAINTY_RADIUS=${run3_rho}"
echo "Run 4: SAFETY_VALUE_NOISE_STD=${run4_std}, SAFETY_VALUE_NOISE_BIAS=${run4_bias}, SAFETY_STATE_UNCERTAINTY_RADIUS=${run4_rho}"
echo "Run 5: SAFETY_VALUE_NOISE_STD=${run5_std}, SAFETY_VALUE_NOISE_BIAS=${run5_bias}, SAFETY_STATE_UNCERTAINTY_RADIUS=${run5_rho}"

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
--safety_value_noise_std ${run1_std} \
--safety_value_noise_bias ${run1_bias} \
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
--safety_value_noise_std ${run2_std} \
--safety_value_noise_bias ${run2_bias} \
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
--safety_value_noise_std ${run3_std} \
--safety_value_noise_bias ${run3_bias} \
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
--safety_value_noise_std ${run4_std} \
--safety_value_noise_bias ${run4_bias} \
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
--safety_value_noise_std ${run5_std} \
--safety_value_noise_bias ${run5_bias} \
--safety_state_uncertainty_radius ${run5_rho}