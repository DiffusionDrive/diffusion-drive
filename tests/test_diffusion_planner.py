from commonroad.common.file_reader import CommonRoadFileReader
import os
import numpy as np


def test_load_scenario():
    # 1. Load the XML file
    file_path = "../data/scenarios/DEU_Flensburg-41_1_T-1.xml"

    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}. Did you put it in the right folder?")
        return

    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    # 2. Extract the ego vehicle's planning problem
    ego_id = list(planning_problem_set.planning_problem_dict.keys())[0]
    planning_problem = planning_problem_set.planning_problem_dict[ego_id]

    # 3. Print the verification stats
    print(f"✅ Successfully loaded {scenario.scenario_id}!")
    print(f"🚗 Found {len(scenario.dynamic_obstacles)} dynamic obstacles (neighbors).")
    print(f"🛣️ Found {len(scenario.lanelet_network.lanelets)} lanelets in the map.")
    print(f"🎯 Ego vehicle must reach goal state: {planning_problem.goal}")

    print(f"🎯 Ego vehicle must reach goal state: {planning_problem.goal}")

    # --- NEW TEST CODE ---
    from src.planner.adapter import extract_map_tensors

    print("\nExtracting Map Tensors...")
    polylines, speed, has_speed = extract_map_tensors(scenario)

    print(f"Polylines shape: {polylines.shape} (Expected: 70, 20, 12)")
    print(f"Speed limit shape: {speed.shape} (Expected: 70, 1)")
    print(f"Has speed limit shape: {has_speed.shape} (Expected: 70, 1)")

    # Print the features of the first point of the first lane
    print(
        f"Lane 0, Point 0 Features (x, y, dx, dy, lx-x, ly-y, rx-x, ry-y, ...):\n{np.round(polylines[0, 0], 2)}"
    )

    print("\nExtracting Neighbor Tensors...")
    from src.planner.adapter import extract_neighbor_tensors

    # We use time step 0 for the test. In the real simulation, Student A will pass the live time step.
    neighbors = extract_neighbor_tensors(scenario, current_time_step=0)

    print(f"Neighbors shape: {neighbors.shape} (Expected: 32, 21, 11)")

    # Check the latest timestep (index 20) of the first vehicle (index 0)
    print(
        f"Vehicle 0, Latest State (x, y, cos, sin, vx, vy, w, l, type[3]):\n{np.round(neighbors[0, 20], 2)}"
    )

    print("\nExtracting Ego & Static Tensors...")
    from src.planner.adapter import extract_ego_state, extract_static_objects

    ego_state = extract_ego_state(planning_problem)
    static_objs = extract_static_objects(scenario)

    print(f"Ego state shape: {ego_state.shape} (Expected: (4,))")
    print(f"Ego state values: {np.round(ego_state, 2)}")
    print(f"Static objects shape: {static_objs.shape} (Expected: 5, 10)")

    print("\nExtracting Navigation Route Tensor...")
    from src.planner.adapter import extract_navigation_route

    nav_route = extract_navigation_route(scenario, planning_problem)

    print(f"Navigation Route shape: {nav_route.shape} (Expected: 25, 20, 4)")
    print(f"First point of the route (x, y, dx, dy): {np.round(nav_route[0, 0], 2)}")


if __name__ == "__main__":
    test_load_scenario()
