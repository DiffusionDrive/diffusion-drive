import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

from src.core.types import PlannerInput
from src.planner.diffusion_planner.inference import DiffusionPlannerNode
from src.planner.diffusion_planner.adapter import (
    extract_map_tensors, extract_neighbor_tensors, extract_navigation_route,
    extract_static_objects, extract_ego_state, normalize_to_ego_frame
)

def test_run():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    scenario_search_path = os.path.join(project_root, "data", "scenarios", "*.xml")
    scenario_files = glob.glob(scenario_search_path)
    if not scenario_files:
        raise FileNotFoundError("No .xml scenario files found in data/scenarios/")

    scenario_path = scenario_files[0]
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # Initialize your original node
    planner = DiffusionPlannerNode(
        checkpoint_path=os.path.join(project_root, "checkpoints", "diffusion_planner", "model.pth"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 1. Extract Raw Global Data
    polylines, speed, has_speed = extract_map_tensors(scenario, planning_problem)
    neighbors = extract_neighbor_tensors(scenario, current_time_step=0)
    static_objs = extract_static_objects(scenario)
    nav_route, route_speed, route_has_speed = extract_navigation_route(scenario, planning_problem)

    # Use the vehicle's exact ground-truth orientation from planning_problem
    ego_state = extract_ego_state(planning_problem)
    ego_pose = ego_state[:4]

    # 2. Transform spatial data to Ego-Centric frame
    local_poly, local_neigh, local_route, local_stat = normalize_to_ego_frame(
        ego_pose, polylines, neighbors, nav_route, static_objs
    )

    # 3. Formulate local ego state with forward velocity
    vel = np.linalg.norm(ego_state[4:6])
    accel = np.linalg.norm(ego_state[6:8])

    local_ego = np.array([
        0.0, 0.0, 1.0, 0.0,
        vel, 0.0,
        accel, 0.0,
        2.30, 5.18
    ], dtype=np.float32)

    live_input = PlannerInput(
        ego_current_state=local_ego,
        neighbor_agents_past=local_neigh,
        lanes=local_poly,
        lanes_speed_limit=speed,
        lanes_has_speed_limit=has_speed,
        static_objects=local_stat,
        route_lanes=local_route,
        route_lanes_speed_limit=route_speed,
        route_lanes_has_speed_limit=route_has_speed
    )

    output_trajectory = planner.plan(live_input)

    # 4. Transform Local Output Trajectory back to Global Map Coordinates
    ego_x, ego_y, cos_yaw, sin_yaw = ego_pose
    yaw = np.arctan2(sin_yaw, cos_yaw)

    rot_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    local_xy = output_trajectory.ego_states[:, :2]
    global_xy = (rot_matrix @ local_xy.T).T + np.array([ego_x, ego_y])

    # 5. Plotting
    plt.figure(figsize=(12, 12))
    rnd = MPRenderer()
    scenario.draw(rnd)
    planning_problem.draw(rnd)
    rnd.render()

    plt.plot(global_xy[:, 0], global_xy[:, 1],
             color='red', linewidth=2.5, marker='o', markersize=3,
             label='Diffusion-Planner Output', zorder=20)

    plt.legend()
    plt.title("Diffusion-Planner Trajectory")
    plt.show()

if __name__ == "__main__":
    test_run()