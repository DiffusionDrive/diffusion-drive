"""DTP pretrained inference with correct latent and trajectory denormalization."""
from pathlib import Path
import pickle
import numpy as np
import torch
import yaml
from .diffusion_policy.networks import ConditionalMLPDenoiser


def _schedule(config, device):
    d = config['diffusion']
    n = int(d['num_diffusion_steps'])

    if d['beta_schedule'] == 'cosine':
        x = torch.linspace(0, n, n + 1, device=device)
        a = torch.cos(((x / n + 0.008) / 1.008) * torch.pi / 2).square()
        a = a / a[0]
        beta = (1 - a[1:] / a[:-1]).clamp(0.0001, 0.9999)
    elif d['beta_schedule'] == 'linear':
        beta = torch.linspace(d['beta_start'], d['beta_end'], n, device=device)
    else:
        raise ValueError('Unsupported beta schedule')
    return torch.cumprod(1 - beta, dim=0)


class DTPPredictor:
    def __init__(self, root, device=None):
        root = Path(root)
        folder = root / 'checkpoints' / 'dtp'
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        config_path = root / 'src/planner/dtp/configs/main_config.yaml'

        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.config = config
        self.model = ConditionalMLPDenoiser(config).to(self.device)
        ckpt = torch.load(folder / 'model.pth', map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt)
        self.model.eval()
        self.traj_stats = torch.load(folder / 'normalization_stats.pt', map_location='cpu', weights_only=False)
        self.latent_stats = torch.load(folder / 'latent_stats.pt', map_location='cpu', weights_only=False)

        with open(folder / 'trajectory_pca.pkl', 'rb') as f:
            self.pca = pickle.load(f)

        self.alphas = _schedule(config, self.device)
        self.latent_dim = int(config['pca']['n_components'])

    @torch.no_grad()
    def predict(self, state, samples=12, steps=50, seed=42):
        if not 1 <= steps <= len(self.alphas):
            raise ValueError('Invalid DDIM step count')

        torch.manual_seed(seed)
        state = {k: v.to(self.device).repeat((samples,) + (1,) * (v.ndim - 1))
                 for k, v in state.items()}

        x = torch.randn(samples, self.latent_dim, device=self.device)
        times = torch.linspace(-1, len(self.alphas) - 1, steps + 1).long().flip(0).tolist()

        for t, next_t in zip(times[:-1], times[1:]):
            noise = self.model(x, torch.full((samples,), t, device=self.device, dtype=torch.long),
                               state, force_drop_goal=False)
            a = self.alphas[t]
            a_next = self.alphas[next_t] if next_t >= 0 else torch.ones_like(a)
            x0 = ((x - torch.sqrt(1 - a) * noise) / torch.sqrt(a)).clamp(-1, 1)
            x = torch.sqrt(a_next) * x0 + torch.sqrt(1 - a_next) * noise

        latent = x.cpu().numpy()
        low = np.asarray(self.latent_stats['min'])
        high = np.asarray(self.latent_stats['max'])
        latent = (latent + 1) / 2 * (high - low) + low
        flat = self.pca.inverse_transform(latent)

        if flat.shape[1] != 160:
            raise ValueError(f'Expected 160 PCA reconstruction coordinates, got {flat.shape}')

        xy = flat.reshape(samples, 80, 2)
        low = np.asarray(self.traj_stats['min'])
        high = np.asarray(self.traj_stats['max'])

        return ((xy + 1) / 2 * (high - low) + low).astype(np.float64)
