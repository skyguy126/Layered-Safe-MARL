"""
Offline single-agent task cost-to-go grid for navigation guidance.

The existing CBVF table (HjDataHandle) is an offline pairwise safety value function.
This module provides a separate offline single-agent cost-to-go function T_goal(p)
that approximates distance or shortest-path cost from position p to a goal.
Online, CBVF/LCB enforces safety while T_goal biases safe action selection
toward task completion — separating strategic guidance from tactical safety filtering.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from multiagent.config import DoubleIntegratorConfig


DEFAULT_TASK_VALUE_GRID_PATH = "data/double_integrator_task_value_grid.npy"
DEFAULT_TASK_VALUE_META_PATH = "data/double_integrator_task_value_grid_meta.json"


def _grid_axes(world_size: float, grid_resolution: int) -> Tuple[np.ndarray, np.ndarray, float]:
    half = world_size / 2.0
    x = np.linspace(-half, half, grid_resolution)
    y = np.linspace(-half, half, grid_resolution)
    cell_size = world_size / max(grid_resolution - 1, 1)
    return x, y, cell_size


def _position_to_cell(
    position: np.ndarray,
    world_size: float,
    grid_resolution: int,
) -> Tuple[int, int]:
    half = world_size / 2.0
    cell_size = world_size / max(grid_resolution - 1, 1)
    ix = int(round((float(position[0]) + half) / cell_size))
    iy = int(round((float(position[1]) + half) / cell_size))
    ix = int(np.clip(ix, 0, grid_resolution - 1))
    iy = int(np.clip(iy, 0, grid_resolution - 1))
    return ix, iy


def _euclidean_cost_to_go(
    goal_position: np.ndarray,
    world_size: float,
    grid_resolution: int,
) -> np.ndarray:
    x, y, _ = _grid_axes(world_size, grid_resolution)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.sqrt((xx - goal_position[0]) ** 2 + (yy - goal_position[1]) ** 2)


def _mark_obstacle_cells(
    grid_resolution: int,
    world_size: float,
    obstacles: Optional[Sequence[Dict[str, Any]]],
) -> np.ndarray:
    blocked = np.zeros((grid_resolution, grid_resolution), dtype=bool)
    if not obstacles:
        return blocked
    for obstacle in obstacles:
        center = np.asarray(obstacle["position"], dtype=float)
        radius = float(obstacle.get("radius", 0.05))
        ix, iy = _position_to_cell(center, world_size, grid_resolution)
        cell_size = world_size / max(grid_resolution - 1, 1)
        radius_cells = int(np.ceil(radius / cell_size)) + 1
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cx = ix + dx
                cy = iy + dy
                if 0 <= cx < grid_resolution and 0 <= cy < grid_resolution:
                    x, y, _ = _grid_axes(world_size, grid_resolution)
                    cell_pos = np.array([x[cx], y[cy]])
                    if np.linalg.norm(cell_pos - center) <= radius:
                        blocked[cx, cy] = True
    return blocked


def _dijkstra_cost_to_go(
    goal_position: np.ndarray,
    world_size: float,
    grid_resolution: int,
    obstacles: Optional[Sequence[Dict[str, Any]]] = None,
) -> np.ndarray:
    x, y, cell_size = _grid_axes(world_size, grid_resolution)
    blocked = _mark_obstacle_cells(grid_resolution, world_size, obstacles)
    goal_ix, goal_iy = _position_to_cell(goal_position, world_size, grid_resolution)
    if blocked[goal_ix, goal_iy]:
        blocked[goal_ix, goal_iy] = False

    inf = np.finfo(np.float64).max / 4.0
    costs = np.full((grid_resolution, grid_resolution), inf, dtype=np.float64)
    costs[goal_ix, goal_iy] = 0.0

    neighbors = [
        (1, 0, cell_size),
        (-1, 0, cell_size),
        (0, 1, cell_size),
        (0, -1, cell_size),
        (1, 1, cell_size * np.sqrt(2.0)),
        (1, -1, cell_size * np.sqrt(2.0)),
        (-1, 1, cell_size * np.sqrt(2.0)),
        (-1, -1, cell_size * np.sqrt(2.0)),
    ]
    heap: List[Tuple[float, int, int]] = [(0.0, goal_ix, goal_iy)]
    while heap:
        current_cost, cx, cy = heapq.heappop(heap)
        if current_cost > costs[cx, cy]:
            continue
        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= grid_resolution or ny >= grid_resolution:
                continue
            if blocked[nx, ny]:
                continue
            next_cost = current_cost + step_cost
            if next_cost < costs[nx, ny]:
                costs[nx, ny] = next_cost
                heapq.heappush(heap, (next_cost, nx, ny))
    unreachable = costs >= inf / 2.0
    if np.any(unreachable):
        euclidean = _euclidean_cost_to_go(goal_position, world_size, grid_resolution)
        costs[unreachable] = euclidean[unreachable]
    return costs.astype(np.float32)


def compute_task_value_grid(
    world_size: float,
    goal_positions: Sequence[np.ndarray],
    obstacles: Optional[Sequence[Dict[str, Any]]] = None,
    grid_resolution: int = 101,
    use_dijkstra: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute a 2D task value grid for each goal position."""
    unique_goals: List[np.ndarray] = []
    for goal in goal_positions:
        goal_arr = np.asarray(goal, dtype=float)
        if not any(np.allclose(goal_arr, existing) for existing in unique_goals):
            unique_goals.append(goal_arr)

    grids = []
    for goal in unique_goals:
        if use_dijkstra and obstacles:
            grid = _dijkstra_cost_to_go(goal, world_size, grid_resolution, obstacles)
        elif use_dijkstra:
            grid = _dijkstra_cost_to_go(goal, world_size, grid_resolution, obstacles=None)
        else:
            grid = _euclidean_cost_to_go(goal, world_size, grid_resolution)
        grids.append(grid)

    values = np.stack(grids, axis=0).astype(np.float32)
    metadata = {
        "world_size": float(world_size),
        "grid_resolution": int(grid_resolution),
        "goal_positions": [g.tolist() for g in unique_goals],
        "obstacles": list(obstacles) if obstacles else [],
        "value_meaning": "approximate task cost-to-go",
        "method": "dijkstra" if use_dijkstra else "euclidean",
    }
    return values, metadata


def save_task_value_grid(
    values: np.ndarray,
    metadata: Dict[str, Any],
    grid_path: Union[str, Path] = DEFAULT_TASK_VALUE_GRID_PATH,
    meta_path: Union[str, Path] = DEFAULT_TASK_VALUE_META_PATH,
) -> None:
    grid_path = Path(grid_path)
    meta_path = Path(meta_path)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(grid_path, values)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


class TaskValueGridHandle:
    """Loads and queries offline single-agent task cost-to-go grids."""

    def __init__(
        self,
        grid_path: Union[str, Path] = DEFAULT_TASK_VALUE_GRID_PATH,
        meta_path: Optional[Union[str, Path]] = None,
    ):
        grid_path = Path(grid_path)
        if meta_path is None:
            meta_path = grid_path.with_name(grid_path.stem + "_meta.json")
        else:
            meta_path = Path(meta_path)

        self.values = np.load(grid_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.world_size = float(self.metadata["world_size"])
        self.grid_resolution = int(self.metadata["grid_resolution"])
        self.goal_positions = [np.asarray(g, dtype=float) for g in self.metadata["goal_positions"]]

    def goal_index(self, goal_position: np.ndarray) -> int:
        goal_position = np.asarray(goal_position, dtype=float)
        distances = [np.linalg.norm(goal_position - g) for g in self.goal_positions]
        return int(np.argmin(distances))

    def lookup(self, position: np.ndarray, goal_position: np.ndarray) -> float:
        position = np.asarray(position, dtype=float)
        goal_idx = self.goal_index(goal_position)
        ix, iy = _position_to_cell(position, self.world_size, self.grid_resolution)
        return float(self.values[goal_idx, ix, iy])


def predict_double_integrator_next_position(
    state: np.ndarray,
    action: np.ndarray,
    dt: float = DoubleIntegratorConfig.DT,
) -> np.ndarray:
    """One-step position prediction using double-integrator dynamics."""
    vx = state[2] + action[0] * dt
    vy = state[3] + action[1] * dt
    px = state[0] + state[2] * dt + 0.5 * action[0] * dt * dt
    py = state[1] + state[3] * dt + 0.5 * action[1] * dt * dt
    return np.array([px, py], dtype=float)


def get_double_integrator_discrete_actions() -> List[np.ndarray]:
    accel_x_options = np.linspace(
        DoubleIntegratorConfig.ACCELX_MIN,
        DoubleIntegratorConfig.ACCELX_MAX,
        DoubleIntegratorConfig.ACCELX_OPTIONS,
    )
    accel_y_options = np.linspace(
        DoubleIntegratorConfig.ACCELY_MIN,
        DoubleIntegratorConfig.ACCELY_MAX,
        DoubleIntegratorConfig.ACCELY_OPTIONS,
    )
    return [np.array([ax, ay], dtype=float) for ax in accel_x_options for ay in accel_y_options]


def _build_minimal_eval_world(num_agents: int, num_landmarks: int, num_obstacles: int):
    from multiagent.core import Agent, Landmark, EntityDynamicsType

    class _World:
        pass

    world = _World()
    world.dim_c = 0
    world.agents = []
    for i in range(num_agents):
        agent = Agent(EntityDynamicsType.DoubleIntegratorXY)
        agent.id = i
        agent.done = False
        world.agents.append(agent)
    world.landmarks = [Landmark() for _ in range(num_landmarks * num_agents)]
    world.obstacles = [Landmark() for _ in range(num_obstacles)]
    for obstacle in world.obstacles:
        obstacle.size = 0.050
    return world


def extract_eval_scenario_layout(
    scenario_name: str,
    world_size: float,
    num_agents: int,
    num_landmarks: int,
    num_obstacles: int = 0,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Extract landmark goal positions and obstacle info from an eval scenario layout."""
    from types import SimpleNamespace

    if scenario_name == "navigation_graph_safe_eval":
        from multiagent.custom_scenarios.navigation_graph_safe_eval import Scenario as EvalScenario

        scenario = EvalScenario()
    else:
        raise NotImplementedError(f"Scenario layout extraction not implemented for {scenario_name}")

    args = SimpleNamespace(
        world_size=world_size,
        num_agents=num_agents,
        num_scripted_agents=0,
        num_obstacles=num_obstacles,
        collaborative=False,
        use_dones=False,
        episode_length=100,
        num_env_steps=10000,
        n_rollout_threads=1,
        dynamics_type="double_integrator",
        graph_feat_type="global",
        num_walls=0,
        use_masking=True,
    )
    from multiagent.core import EntityDynamicsType

    scenario.world_size = world_size
    scenario.num_agents = num_agents
    scenario.num_obstacles = num_obstacles
    scenario.dynamics_type = EntityDynamicsType.DoubleIntegratorXY
    scenario.config_class = DoubleIntegratorConfig
    scenario.separation_distance = DoubleIntegratorConfig.SEPARATION_DISTANCE
    world = _build_minimal_eval_world(num_agents, num_landmarks, num_obstacles)
    scenario.init_landmarks(world, num_landmarks, 0)
    scenario.random_scenario(world)

    goal_positions = [landmark.state.p_pos.copy() for landmark in world.landmarks]
    obstacles = []
    for obstacle in world.obstacles:
        obstacles.append(
            {
                "position": obstacle.state.p_pos.tolist(),
                "radius": float(obstacle.size),
            }
        )
    return goal_positions, obstacles
