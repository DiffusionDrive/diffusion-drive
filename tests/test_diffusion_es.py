import torch
import numpy as np
from dataclasses import dataclass
from commonroad.common.file_reader import CommonRoadFileReader

# Import adapters and models
from src.planner.diffusion_es.adapter_diffusion_es import (
    extract_pgp_map_tensor,
    extract_pgp_agent_tensors,
)
from src.planner.diffusion_es.models.pgp.pgp_model import PGPModel
from src.planner.diffusion_es.models.pgp.pgp_encoder import PGPEncoder
from src.planner.diffusion_es.models.pgp.pgp_aggregator import PGP
from src.planner.diffusion_es.models.pgp.pgp_decoder import LVM


# --- Mock Dataclasses to replace nuPlan dependencies ---
@dataclass
class MockGraphMap:
    lane_node_feats: torch.Tensor
    lane_node_masks: torch.Tensor
    nodes_on_route_flag: torch.Tensor
    red_light_flag: torch.Tensor
    s_next: torch.Tensor
    edge_type: torch.Tensor
    edge_on_route_mask: torch.Tensor


@dataclass
class MockEgoAgents:
    ego_feats: torch.Tensor
    vehicle_agent_feats: torch.Tensor
    vehicle_agent_masks: torch.Tensor
    pedestrians_agent_feats: torch.Tensor
    pedestrians_agent_masks: torch.Tensor


@dataclass
class MockAgentNodeMasks:
    vehicle_node_masks: torch.Tensor
    pedestrian_node_masks: torch.Tensor


@dataclass
class MockPGPTraversals:
    visited_edges: torch.Tensor
    init_node: torch.Tensor
    node_seq_gt: torch.Tensor


@dataclass
class MockPGPFeatures:
    graph_map: MockGraphMap
    ego_agent_features: MockEgoAgents
    att_node_masks: MockAgentNodeMasks
    traversal_features: MockPGPTraversals
    has_init_node: bool = False


def test_smoke_run():
    print("1. Loading CommonRoad Scenario...")
    # Find any XML file in your project to test with
    xml_file = (
        "../data/scenarios/DEU_Flensburg-41_1_T-1.xml"  # Update this path if needed
    )
    scenario, planning_problem_set = CommonRoadFileReader(xml_file).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    print("2. Extracting Tensors via Diffusion-ES Adapter...")
    map_feats, map_masks = extract_pgp_map_tensor(scenario)
    ego_feats, veh_feats, ped_feats, veh_masks, ped_masks = extract_pgp_agent_tensors(
        scenario, planning_problem
    )

    # Create route flag (Mocking the route for now)
    # Create dummy graph adjacency matrices (must be int64!)
    max_edges = 15
    s_next = np.zeros((map_feats.shape[0], max_edges), dtype=np.float32)
    edge_type = np.zeros((map_feats.shape[0], max_edges), dtype=np.float32)
    route_flag = np.zeros((map_feats.shape[0], map_feats.shape[1], 1), dtype=np.float32)
    att_veh_masks = np.zeros((map_feats.shape[0], veh_feats.shape[0]), dtype=np.float32)
    att_ped_masks = np.zeros((map_feats.shape[0], ped_feats.shape[0]), dtype=np.float32)
    edge_on_route_mask = np.zeros((map_feats.shape[0], max_edges), dtype=np.float32)

    print("3. Packaging into PyTorch Dataclasses...")
    graph_map = MockGraphMap(
        lane_node_feats=torch.tensor(map_feats).unsqueeze(0),
        lane_node_masks=torch.tensor(map_masks).unsqueeze(0).unsqueeze(-1),
        nodes_on_route_flag=torch.tensor(route_flag).unsqueeze(0),
        red_light_flag=torch.tensor(route_flag).unsqueeze(0),
        s_next=torch.tensor(s_next).unsqueeze(0),
        edge_type=torch.tensor(edge_type).unsqueeze(0),
        edge_on_route_mask=torch.tensor(edge_on_route_mask).unsqueeze(0),
    )

    ego_agents = MockEgoAgents(
        ego_feats=torch.tensor(ego_feats).unsqueeze(0),
        vehicle_agent_feats=torch.tensor(veh_feats).unsqueeze(0),
        vehicle_agent_masks=torch.tensor(veh_masks).unsqueeze(0).unsqueeze(-1),
        pedestrians_agent_feats=torch.tensor(ped_feats).unsqueeze(0),
        pedestrians_agent_masks=torch.tensor(ped_masks).unsqueeze(0).unsqueeze(-1),
    )

    att_node_masks = MockAgentNodeMasks(
        vehicle_node_masks=torch.tensor(att_veh_masks).unsqueeze(0),
        pedestrian_node_masks=torch.tensor(att_ped_masks).unsqueeze(0),
    )

    # Initialize the missing traversal tensors
    visited_edges = np.zeros((map_feats.shape[0], 1), dtype=np.int64)

    # init_node must be a probability distribution over the map nodes [Max_Nodes]
    init_node = np.zeros((map_feats.shape[0],), dtype=np.float32)
    init_node[0] = 1.0  # 100% probability of starting at node 0

    node_seq_gt = np.zeros((1, 80), dtype=np.int64)  # Placeholder for pre-training

    traversals = MockPGPTraversals(
        visited_edges=torch.tensor(visited_edges).unsqueeze(0),
        init_node=torch.tensor(init_node).unsqueeze(0),  # <-- Now shape [1, 200]
        node_seq_gt=torch.tensor(node_seq_gt).unsqueeze(0),
    )

    pgp_features = MockPGPFeatures(
        graph_map=graph_map,
        ego_agent_features=ego_agents,
        att_node_masks=att_node_masks,
        traversal_features=traversals,
        has_init_node=True,  # <-- CRITICAL: This allows the encoder to pass the edges!
    )
    features_dict = {"pgp_features": pgp_features}

    print("4. Initializing Dummy Model...")
    encoder = PGPEncoder(
        target_agent_feat_size=7,
        target_agent_emb_size=32,
        target_agent_enc_size=64,
        node_feat_size=7,
        node_emb_size=32,
        node_enc_size=64,
        nbr_feat_size=7,
        nbr_emb_size=32,
        nbr_enc_size=32,
        num_gat_layers=2,
        use_route_feature=True,
        use_red_light_feature=True,
    )

    aggregator = PGP(
        pre_train=False,
        node_enc_size=64,
        target_agent_enc_size=64,
        pi_h1_size=64,
        pi_h2_size=64,
        emb_size=128,
        num_heads=4,
        use_route_mask=True,
        hard_masking=False,
        num_samples=1,
        num_traversals=1,
        horizon=80,
    )

    decoder = LVM(
        encoding_size=192,
        agg_type="sample_specific",
        lv_dim=5,
        hidden_size=128,
        num_clusters=1,
        use_ray=False,
        num_samples=1,
        op_len=80,
    )

    model = PGPModel(encoder=encoder, aggregator=aggregator, decoder=decoder)

    # Load your generated checkpoint
    model.load_state_dict(torch.load("../checkpoints/diffusion_es/model.ckpt"))
    model.eval()

    print("5. Running Forward Pass...")
    with torch.no_grad():
        output = model(features_dict)

    print(f"✅ Success! Trajectory generated with shape: {output['trajectory'].shape}")


if __name__ == "__main__":
    test_smoke_run()
