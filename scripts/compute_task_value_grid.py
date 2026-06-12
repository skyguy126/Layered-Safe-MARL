#!/usr/bin/env python
"""
Precompute offline single-agent task cost-to-go grids for navigation guidance.

The existing CBVF table is an offline pairwise safety value function.
This script builds a separate offline task value table T_goal(p) that
approximates cost-to-go from position p to each goal/landmark.
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.task_value_grid import (
    DEFAULT_TASK_VALUE_GRID_PATH,
    DEFAULT_TASK_VALUE_META_PATH,
    compute_task_value_grid,
    extract_eval_scenario_layout,
    save_task_value_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute offline task value grids.")
    parser.add_argument("--dynamics_type", type=str, default="double_integrator")
    parser.add_argument("--scenario_name", type=str, default="navigation_graph_safe_eval")
    parser.add_argument("--world_size", type=float, default=4.0)
    parser.add_argument("--num_agents", type=int, default=6)
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_obstacles", type=int, default=1)
    parser.add_argument("--grid_resolution", type=int, default=101)
    parser.add_argument(
        "--use_dijkstra",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
        help="Use Dijkstra shortest-path cost; otherwise Euclidean distance.",
    )
    parser.add_argument("--output_grid", type=str, default=DEFAULT_TASK_VALUE_GRID_PATH)
    parser.add_argument("--output_meta", type=str, default=DEFAULT_TASK_VALUE_META_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dynamics_type != "double_integrator":
        raise NotImplementedError("Only double_integrator is supported for task value grids.")

    goal_positions, obstacles = extract_eval_scenario_layout(
        scenario_name=args.scenario_name,
        world_size=args.world_size,
        num_agents=args.num_agents,
        num_landmarks=args.num_landmarks,
        num_obstacles=args.num_obstacles,
    )
    values, metadata = compute_task_value_grid(
        world_size=args.world_size,
        goal_positions=goal_positions,
        obstacles=obstacles,
        grid_resolution=args.grid_resolution,
        use_dijkstra=args.use_dijkstra,
    )
    metadata["dynamics_type"] = args.dynamics_type
    metadata["scenario_name"] = args.scenario_name
    metadata["num_agents"] = args.num_agents
    metadata["num_landmarks"] = args.num_landmarks
    save_task_value_grid(values, metadata, args.output_grid, args.output_meta)
    print(f"Saved task value grid to {args.output_grid}")
    print(f"Saved metadata to {args.output_meta}")
    print(f"Unique goals: {len(metadata['goal_positions'])}, grid shape: {values.shape}")


if __name__ == "__main__":
    main()
