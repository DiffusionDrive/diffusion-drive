import torch
import torch.nn as nn


class StateNormalizer(nn.Module):
    """Denormalizes output trajectories from neural network latent space to physical meters."""

    def __init__(self, mean=None, std=None):
        super().__init__()
        # State normalization constants from args.json
        if mean is None:
            mean = [10.0, 0.0, 0.0, 0.0]
        if std is None:
            std = [20.0, 20.0, 1.0, 1.0]

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        # Maps normalized [-0.5, 0.5] predictions back to physical local meters
        return (x * self.std) + self.mean


class ObservationNormalizer(nn.Module):
    """Normalizes input tensors feature-by-feature while ignoring zero-padded rows."""

    def __init__(self):
        super().__init__()
        self.register_buffer("ego_mean", torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.register_buffer("ego_std", torch.tensor([20.0, 20.0, 1.0, 1.0, 20.0, 20.0, 20.0, 20.0, 1.0, 1.0]))

        self.register_buffer("neigh_mean", torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.register_buffer("neigh_std", torch.tensor([20.0, 20.0, 1.0, 1.0, 20.0, 20.0, 20.0, 20.0, 1.0, 1.0, 1.0]))

        self.register_buffer("lane_mean", torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.register_buffer("lane_std", torch.tensor([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0]))

        self.register_buffer("stat_mean", torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.register_buffer("stat_std", torch.tensor([20.0, 20.0, 1.0, 1.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0]))

    def _norm(self, tensor, mean, std):
        if tensor is None or tensor.numel() == 0:
            return tensor
        mask = tensor.abs().sum(dim=-1) > 1e-5
        if mask.any():
            tensor[mask] = (tensor[mask] - mean) / std
        return tensor

    def _inverse_norm(self, tensor, mean, std):
        if tensor is None or tensor.numel() == 0:
            return tensor
        mask = tensor.abs().sum(dim=-1) > 1e-5
        if mask.any():
            tensor[mask] = (tensor[mask] * std) + mean
        return tensor

    def forward(self, inputs):
        normalized = {}
        if "ego_current_state" in inputs:
            normalized["ego_current_state"] = (inputs["ego_current_state"] - self.ego_mean) / self.ego_std
        if "neighbor_agents_past" in inputs:
            normalized["neighbor_agents_past"] = self._norm(inputs["neighbor_agents_past"].clone(), self.neigh_mean, self.neigh_std)
        if "lanes" in inputs:
            normalized["lanes"] = self._norm(inputs["lanes"].clone(), self.lane_mean, self.lane_std)
        if "route_lanes" in inputs:
            normalized["route_lanes"] = self._norm(inputs["route_lanes"].clone(), self.lane_mean, self.lane_std)
        if "static_objects" in inputs:
            normalized["static_objects"] = self._norm(inputs["static_objects"].clone(), self.stat_mean, self.stat_std)


        # Normalize speed limits using checkpoint statistics.
        if "lanes_speed_limit" in inputs:
            normalized["lanes_speed_limit"] = (
                    inputs["lanes_speed_limit"] / 20.0
            )

        if "route_lanes_speed_limit" in inputs:
            normalized["route_lanes_speed_limit"] = (
                    inputs["route_lanes_speed_limit"] / 20.0
            )


        for k, v in inputs.items():
            if k not in normalized:
                normalized[k] = v

        return normalized

    def inverse(self, inputs):
        denormalized = {}
        if "ego_current_state" in inputs:
            denormalized["ego_current_state"] = (inputs["ego_current_state"] * self.ego_std) + self.ego_mean
        if "neighbor_agents_past" in inputs:
            denormalized["neighbor_agents_past"] = self._inverse_norm(inputs["neighbor_agents_past"].clone(), self.neigh_mean, self.neigh_std)
        if "lanes" in inputs:
            denormalized["lanes"] = self._inverse_norm(inputs["lanes"].clone(), self.lane_mean, self.lane_std)
        if "route_lanes" in inputs:
            denormalized["route_lanes"] = self._inverse_norm(inputs["route_lanes"].clone(), self.lane_mean, self.lane_std)
        if "static_objects" in inputs:
            denormalized["static_objects"] = self._inverse_norm(inputs["static_objects"].clone(), self.stat_mean, self.stat_std)


        if "lanes_speed_limit" in inputs:
            denormalized["lanes_speed_limit"] = (
                    inputs["lanes_speed_limit"] * 20.0
            )

        if "route_lanes_speed_limit" in inputs:
            denormalized["route_lanes_speed_limit"] = (
                    inputs["route_lanes_speed_limit"] * 20.0
            )


        for k, v in inputs.items():
            if k not in denormalized:
                denormalized[k] = v

        return denormalized