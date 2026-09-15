import random
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from src.planner.diffusion_es.models.pgp.pgp_aggregator import PGP
from src.planner.diffusion_es.models.pgp.pgp_decoder import LVM
from src.planner.diffusion_es.models.pgp.pgp_encoder import PGPEncoder


class PGPModel(nn.Module):
    """Decoupled PGP Model for CommonRoad Inference"""

    def __init__(
        self,
        encoder: PGPEncoder,
        aggregator: PGP,
        decoder: LVM,
        filter_trajectories_by_endpoint: bool = False,
        smooth_output_trajectory: bool = False,
        interpolate_yaw: bool = False,
        average_output_trajectories: bool = False,
        return_graph_map: bool = False,
        return_traversal_coordinates: bool = False,
    ):
        super(PGPModel, self).__init__()

        self.return_graph_map = return_graph_map
        self.return_traversal_coordinates = return_traversal_coordinates
        self.filter_trajectories_by_endpoint = filter_trajectories_by_endpoint
        self.smooth_output_trajectory = smooth_output_trajectory
        self.interpolate_yaw = interpolate_yaw
        self.average_output_trajectories = average_output_trajectories

        self.encoder = encoder
        self.aggregator = aggregator
        self.decoder = decoder

    def silently_seed_everything(self, seed) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_final_waypoint_on_route_mask(self, trajectories, graph_map, threshold):
        endpoints = trajectories[:, :, -1, :]
        lane_graph_poses = graph_map.lane_node_feats[:, :, :, :2].flatten(
            start_dim=1, end_dim=2
        )
        pairwise_distance = torch.cdist(endpoints, lane_graph_poses, p=2)
        on_route_poses_mask = (
            graph_map.nodes_on_route_flag[..., 0]
            .flatten(start_dim=1, end_dim=2)
            .unsqueeze(1)
        )
        pairwise_distance = on_route_poses_mask * pairwise_distance + torch.where(
            on_route_poses_mask.bool(), 0.0, np.inf
        )
        min_pairwise_distance = pairwise_distance.min(dim=-1).values
        endpoint_on_route_mask = (min_pairwise_distance < threshold).float()
        at_least_one_endpoint_one_route_mask = (
            endpoint_on_route_mask.sum(dim=-1) > 0
        ).float()
        smallest_distance_endpoint_mask = (
            min_pairwise_distance == min_pairwise_distance.min(dim=-1).values
        ).float()

        return (
            at_least_one_endpoint_one_route_mask * endpoint_on_route_mask
            + (1 - at_least_one_endpoint_one_route_mask)
            * smallest_distance_endpoint_mask
        )

    def forward(self, features: Dict[str, Any]) -> Dict[str, Any]:
        self.silently_seed_everything(0)
        encodings = self.encoder(features)
        agg_encoding = self.aggregator(encodings)
        outputs = self.decoder(agg_encoding)

        # (Remaining forward logic is untouched, just returns a raw dict now instead of nuPlan TargetsType)
        probs = outputs["probs"]
        most_likely_trajectory = (
            outputs["traj"]
            .take_along_dim(
                indices=probs.argmax(dim=1, keepdim=True)[..., None, None], dim=1
            )
            .squeeze(dim=1)
        )

        batch_size, num_poses = most_likely_trajectory.shape[:2]
        dummy_heading = torch.zeros(
            [batch_size, num_poses, 1], device=most_likely_trajectory.device
        )
        most_likely_trajectory = torch.cat(
            [most_likely_trajectory, dummy_heading], dim=-1
        )

        predictions = {
            "trajectory": most_likely_trajectory,  # Simplified for inference
            "probabilities": probs,
        }
        return predictions
