"""CommonRoad -> OptTrajDiff (QCNet/AV2) inference adapter.

Experimental domain transfer: CommonRoad lanelets are represented as AV2-style
map polygons. Model weights are never modified. The model's 10 Hz output is
interpreted as positions relative to the current ego pose.
"""
from __future__ import annotations

import sys
import time
import traceback
import platform
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import HeteroData, Batch


def _summary(value, name, max_unique=30):
    """Safe summary; never print full tensors or checkpoint contents."""
    try:
        if isinstance(value, torch.Tensor):
            x = value.detach()
            msg = f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
            if x.numel():
                if x.dtype == torch.bool or not x.is_floating_point():
                    unique = torch.unique(x.cpu())
                    msg += f", min={x.min().item()}, max={x.max().item()}, unique_count={unique.numel()}"
                    msg += f", unique_first={unique[:max_unique].tolist()}"
                else:
                    finite = torch.isfinite(x)
                    msg += f", finite={finite.sum().item()}/{x.numel()}"
                    if finite.any():
                        vals = x[finite]
                        msg += f", min={vals.min().item():.5g}, max={vals.max().item():.5g}, mean={vals.float().mean().item():.5g}"
            return msg
        if isinstance(value, np.ndarray):
            return _summary(torch.from_numpy(value), name, max_unique)
        return f"{name}: {type(value).__name__}, value={str(value)[:160]}"
    except Exception as exc:
        return f"{name}: summary failed: {type(exc).__name__}: {exc}"


def _diagnose_data(data, model):
    print("\n[DIAG] === INPUT FEATURE AND EMBEDDING AUDIT ===", flush=True)
    print(f"[DIAG] stores={list(data.node_types)} edges={list(data.edge_types)}", flush=True)
    for store_name in data.node_types:
        store = data[store_name]
        print(f"[DIAG] STORE {store_name} num_nodes={store.num_nodes}", flush=True)
        for key in store.keys():
            print('[DIAG] ' + _summary(store[key], f'{store_name}.{key}'), flush=True)
    for edge_type in data.edge_types:
        for key in data[edge_type].keys():
            print('[DIAG] ' + _summary(data[edge_type][key], f'{edge_type}.{key}'), flush=True)
    print('[DIAG] --- ALL EMBEDDING LAYERS ---', flush=True)
    for module_name, module in model.qcnet.named_modules():
        if not isinstance(module, torch.nn.Embedding):
            continue
        print(f'[DIAG] embedding={module_name}, num_embeddings={module.num_embeddings}, embedding_dim={module.embedding_dim}', flush=True)
    print('[DIAG] --- INPUT VS EMBEDDING CANDIDATE MATCHES ---', flush=True)
    checks = [
        ('map_point', 'type', 'encoder.map_encoder.type_pt_emb'),
        ('map_point', 'side', 'encoder.map_encoder.side_pt_emb'),
        ('map_polygon', 'type', 'encoder.map_encoder.type_pl_emb'),
        ('map_polygon', 'is_intersection', 'encoder.map_encoder.intersection_emb'),
        ('agent', 'type', 'encoder.agent_encoder.type_emb'),
    ]
    modules = dict(model.qcnet.named_modules())
    for store, key, module_name in checks:
        if key not in data[store]:
            print(f'[DIAG] {store}.{key}: absent', flush=True)
            continue
        x = data[store][key]
        layer = modules.get(module_name)
        if layer is None:
            print(f'[DIAG] {store}.{key}: embedding {module_name} not found (inspect model definitions)', flush=True)
        elif isinstance(layer, torch.nn.Embedding):
            invalid = ((x.long() < 0) | (x.long() >= layer.num_embeddings))
            print(f'[DIAG] {store}.{key} -> {module_name}: invalid={int(invalid.sum())}/{x.numel()}, bad_values={torch.unique(x[invalid]).cpu().tolist() if invalid.any() else []}', flush=True)
    print('[DIAG] === END AUDIT (NO INPUTS MODIFIED) ===\n', flush=True)


@dataclass
class Plan:
    points: np.ndarray
    latency: float
    candidate_count: int
    candidates: np.ndarray | None = None


def _xy(x):
    return np.asarray(x, dtype=float)[:2]


def _angle(v):
    return float(np.arctan2(v[1], v[0]))


def _resample(vertices, spacing=2.0):
    p = np.asarray(vertices, dtype=float)[:, :2]
    if len(p) < 2:
        return np.repeat(p[:1], 2, axis=0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0., np.cumsum(seg)]
    keep = np.r_[True, np.diff(s) > 1e-6]
    s, p = s[keep], p[keep]
    if len(s) < 2:
        return np.repeat(p[:1], 2, axis=0)
    t = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / spacing)) + 1))
    return np.stack([np.interp(t, s, p[:, k]) for k in range(2)], axis=-1)


def _history(scenario, obstacle, now, steps, dt, ego_state=None):
    """Return (positions, headings, velocities, valid) in 10Hz history."""
    positions = np.zeros((steps, 2), dtype=np.float32)
    headings = np.zeros(steps, dtype=np.float32)
    velocities = np.zeros((steps, 2), dtype=np.float32)
    valid = np.zeros(steps, dtype=bool)
    for j in range(steps):
        t = now - (steps - 1 - j)
        if obstacle is None:
            if t != now:
                continue
            state = ego_state
        else:
            try:
                state = obstacle.state_at_time(t)
            except (AttributeError, ValueError, IndexError):
                state = None
        if state is None:
            continue
        positions[j] = _xy(state.position)
        headings[j] = float(getattr(state, 'orientation', 0.0))
        speed = float(getattr(state, 'velocity', 0.0))
        velocities[j] = speed * np.array([np.cos(headings[j]), np.sin(headings[j])])
        valid[j] = True
    if obstacle is None:
        # Synthesized constant-velocity ego history before first closed-loop step.
        for j in range(steps - 1):
            positions[j] = positions[-1] - velocities[-1] * dt * (steps - 1 - j)
            headings[j] = headings[-1]
            velocities[j] = velocities[-1]
            valid[j] = True
    else:
        # Backfill unavailable history; never mark synthetic history as observed.
        if valid.any():
            first = int(np.flatnonzero(valid)[0])
            positions[:first] = positions[first]
            headings[:first] = headings[first]
            velocities[:first] = velocities[first]
    return positions, headings, velocities, valid


class OptTrajDiffAdapter:
    def __init__(self, project_root: Path, device='auto', samples=12):
        self.root = Path(project_root).resolve()
        print(f'[DIAG] Python={platform.python_version()} torch={torch.__version__} CUDA_runtime={torch.version.cuda} CUDA_available={torch.cuda.is_available()}', flush=True)
        print(f'[DIAG] project_root={self.root} device_request={device} samples={samples}', flush=True)
        self.repo = self.root / 'src/planner/opttrajdiff'
        if not (self.repo / 'predictors/diffnet.py').exists():
            raise FileNotFoundError(f'OptTrajDiff source not found: {self.repo}')
        for path in (self.repo, self.root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from predictors import DiffNet
        ckpt = self.root / 'checkpoints/OptTrajDiff/model.ckpt'
        qcnet = self.root / 'checkpoints/OptTrajDiff/QCNet_AV2.ckpt'
        for path in (ckpt, qcnet):
            if not path.is_file():
                raise FileNotFoundError(f'Missing checkpoint: {path}')
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device))
        # Lightning reconstructs DiffNet from its saved hyperparameters, including
        # the QCNet checkpoint path. Patch the saved args *before* construction.
        print(f'[DIAG] checkpoint={ckpt} bytes={ckpt.stat().st_size}; qcnet={qcnet} bytes={qcnet.stat().st_size}', flush=True)
        raw = torch.load(ckpt, map_location='cpu', weights_only=False)
        print(f'[DIAG] checkpoint top-level keys={list(raw.keys())}', flush=True)
        hp = raw.get('hyper_parameters', {})
        args = hp.get('args')
        if args is None:
            raise RuntimeError('model.ckpt lacks hyper_parameters["args"]. Inspect its training configuration.')
        if isinstance(args, dict):
            from argparse import Namespace
            args = Namespace(**args)
        print(f'[DIAG] saved args type={type(args).__name__} fields={sorted(vars(args).keys())}', flush=True)
        args.qcnet_ckpt_path = str(qcnet)
        for key, name in [('path_pca_s_mean', 's_mean_10.npy'), ('path_pca_VT_k', 'VT_k_10.npy'),
                          ('path_pca_latent_mean', 'latent_mean_10.npy'), ('path_pca_latent_std', 'latent_std_10.npy')]:
            path = self.repo / 'pca' / name
            if not path.is_file():
                raise FileNotFoundError(f'Missing PCA statistics: {path}')
            setattr(args, key, str(path))
        # Avoid Lightning's checkpoint loader re-instantiating with stale paths.
        print('[DIAG] constructing DiffNet...', flush=True)
        self.model = DiffNet(args=args)
        print('[DIAG] DiffNet constructed', flush=True)
        missing, unexpected = self.model.load_state_dict(raw['state_dict'], strict=False)
        if missing or unexpected:
            raise RuntimeError(f'Checkpoint keys incompatible: missing={missing[:8]}, unexpected={unexpected[:8]}')
        self.model.add_extra_param(__import__('argparse').Namespace(
            guid_sampling='no_guid', guid_task='none', guid_method='none', guid_plot='no_plot',
            std_reg=0.1, path_pca_V_k='none', cluster='traj', cluster_max_thre=2.5,
            cluster_mean_thre=2.5, cond_norm=0, cost_param_costl=0.0, cost_param_threl=0.0))
        self.model.sampling = 'ddim'
        self.model.sampling_stride = 20
        self.model.num_eval_samples = samples
        self.model.check_param()
        self.model.to(self.device).eval()
        print(f'[DIAG] model ready: device={self.device} history={self.model.num_historical_steps} future={self.model.num_future_steps} modes={self.model.num_modes}', flush=True)
        self.steps = int(self.model.num_historical_steps)
        self.future = int(self.model.num_future_steps)
        self.samples = samples

    def _make_data(self, scenario, ego, now, ego_history=None):
        h, f = self.steps, self.future
        dt = float(scenario.dt)
        if abs(dt - 0.1) > 1e-5:
            raise ValueError(f'Expected 0.1s CommonRoad dt (AV2 10Hz); got {dt}')
        histories = []
        if ego_history is None:
            histories.append(_history(scenario, None, now, h, dt, ego))
        else:
            arr = np.asarray(ego_history, dtype=float)[-h:]
            if len(arr) == 0:
                raise ValueError('Empty ego history')
            if len(arr) < h:
                first = arr[0].copy()
                synth = []
                for k in range(h - len(arr), 0, -1):
                    v = first[3] * np.array([np.cos(first[2]), np.sin(first[2])])
                    synth.append([first[0] - k * dt * v[0], first[1] - k * dt * v[1], first[2], first[3]])
                arr = np.vstack([synth, arr])
            p, head, vel = arr[:, :2].astype('float32'), arr[:, 2].astype('float32'), np.stack([arr[:, 3]*np.cos(arr[:, 2]), arr[:, 3]*np.sin(arr[:, 2])], axis=1).astype('float32')
            histories.append((p, head, vel, np.ones(h, dtype=bool)))
        for ob in scenario.obstacles:
            if getattr(ob, 'obstacle_role', None) is not None and 'STATIC' in str(ob.obstacle_role):
                continue
            hist = _history(scenario, ob, now, h, dt)
            if hist[3][-1] and np.linalg.norm(hist[0][-1] - _xy(ego.position)) < 90:
                histories.append(hist)
        n = len(histories)
        data = HeteroData()
        a = data['agent']
        a.num_nodes = n
        a.av_index = torch.tensor(0, dtype=torch.long)
        a.id = [f'agent_{i}' for i in range(n)]
        a.type = torch.zeros(n, dtype=torch.uint8)
        a.category = torch.tensor([3] + [1]*(n-1), dtype=torch.uint8)
        a.position = torch.zeros((n, h+f, 2), dtype=torch.float32)
        a.heading = torch.zeros((n, h+f), dtype=torch.float32)
        a.velocity = torch.zeros((n, h+f, 2), dtype=torch.float32)
        a.valid_mask = torch.zeros((n, h+f), dtype=torch.bool)
        a.predict_mask = torch.zeros((n, h+f), dtype=torch.bool)
        for i, (p, hd, vel, valid) in enumerate(histories):
            a.position[i, :h] = torch.from_numpy(p)
            a.heading[i, :h] = torch.from_numpy(hd)
            a.velocity[i, :h] = torch.from_numpy(vel)
            a.valid_mask[i, :h] = torch.from_numpy(valid)
            a.predict_mask[i, h:] = True
        # AV2 vector representation uses a valid consecutive pair.
        a.valid_mask[:, 1:h] &= a.valid_mask[:, :h-1].clone()
        a.valid_mask[:, 0] = False
        lanelets = [ll for ll in scenario.lanelet_network.lanelets if len(ll.center_vertices) >= 2]
        if not lanelets:
            raise ValueError('Scenario has no usable lanelets')
        # Limit map extent to keep attention manageable.
        pos = _xy(ego.position)
        lanelets = [ll for ll in lanelets if np.min(np.linalg.norm(np.asarray(ll.center_vertices)[:, :2] - pos, axis=1)) < 100]
        if not lanelets:
            raise ValueError('No lanelets within 100m of ego')
        ids = {ll.lanelet_id: i for i, ll in enumerate(lanelets)}
        poly_pos, poly_head, poly_intersection = [], [], []
        pt_pos, pt_head, pt_mag, pt_type, pt_side, edges = [], [], [], [], [], []
        for idx, ll in enumerate(lanelets):
            center = _resample(ll.center_vertices)
            poly_pos.append(center[0])
            poly_head.append(_angle(center[1]-center[0]))
            poly_intersection.append(0)
            # Match ArgoverseV2Dataset._point_types exactly:
            # UNKNOWN=14, CENTERLINE=16. CommonRoad boundary markings are
            # not necessarily known, so do not invent a specific marking.
            for side, verts, kind in [(0, ll.left_vertices, 14), (1, ll.right_vertices, 14), (2, ll.center_vertices, 16)]:
                p = _resample(verts)
                d = np.diff(p, axis=0)
                for j, delta in enumerate(d):
                    pt_pos.append(p[j]); pt_head.append(_angle(delta)); pt_mag.append(float(np.linalg.norm(delta)))
                    pt_type.append(kind); pt_side.append(side); edges.append((len(pt_pos)-1, idx))
        m = len(lanelets)
        mp = data['map_polygon']
        mp.num_nodes = m
        mp.position = torch.tensor(np.asarray(poly_pos), dtype=torch.float32)
        mp.orientation = torch.tensor(poly_head, dtype=torch.float32)
        mp.type = torch.zeros(m, dtype=torch.uint8)
        mp.is_intersection = torch.tensor(poly_intersection, dtype=torch.uint8)
        pts = data['map_point']
        pts.num_nodes = len(pt_pos)
        pts.position = torch.tensor(np.asarray(pt_pos), dtype=torch.float32)
        pts.orientation = torch.tensor(pt_head, dtype=torch.float32)
        pts.magnitude = torch.tensor(pt_mag, dtype=torch.float32)
        pts.type = torch.tensor(pt_type, dtype=torch.uint8)
        pts.side = torch.tensor(pt_side, dtype=torch.uint8)
        data['map_point', 'to', 'map_polygon'].edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
        links, types = [], []
        for ll in lanelets:
            j = ids[ll.lanelet_id]
            # ArgoverseV2Dataset relation vocabulary:
            # NONE=0, PRED=1, SUCC=2, LEFT=3, RIGHT=4.
            for rel, neighbors in [(2, ll.successor), (1, ll.predecessor)]:
                for neighbor in neighbors:
                    if neighbor in ids:
                        links.append((ids[neighbor], j)); types.append(rel)
            for rel, neighbor in [(3, ll.adj_left), (4, ll.adj_right)]:
                if neighbor in ids:
                    links.append((ids[neighbor], j)); types.append(rel)
        edge = data['map_polygon', 'to', 'map_polygon']
        edge.edge_index = torch.tensor(links, dtype=torch.long).T.contiguous() if links else torch.empty((2, 0), dtype=torch.long)
        edge.type = torch.tensor(types, dtype=torch.uint8)
        data['scenario_id'] = 'commonroad'
        data['city'] = 'CommonRoad'
        return Batch.from_data_list([data]).to(self.device)

    @torch.inference_mode()
    def plan(self, scenario, ego, now, target, ego_history=None):
        t0 = time.perf_counter()
        print(f'\n[DIAG] plan start now={now}, ego_pos={_xy(ego.position).tolist()}, ego_heading={ego.orientation}, ego_speed={ego.velocity}, target={_xy(target).tolist()}', flush=True)
        data = self._make_data(scenario, ego, now, ego_history)
        model = self.model
        _diagnose_data(data, model)
        print('[DIAG] STAGE qcnet forward BEGIN', flush=True)
        try:
            pred, scene_enc = model.qcnet(data)
        except Exception:
            print('[DIAG] STAGE qcnet forward FAILED', flush=True)
            traceback.print_exc()
            raise
        print('[DIAG] STAGE qcnet forward OK', flush=True)
        for key, val in pred.items():
            print('[DIAG] ' + _summary(val, f'pred.{key}'), flush=True)
        if isinstance(scene_enc, dict):
            for key, val in scene_enc.items():
                print('[DIAG] ' + _summary(val, f'scene_enc.{key}'), flush=True)
        traj = pred['loc_refine_pos'][..., :2]
        # The trained diffusion uses the QCNet marginal PCA latent modes.
        if model.s_mean is None:
            model.s_mean = torch.as_tensor(np.load(model.path_pca_s_mean), device=self.device)
            model.VT_k = torch.as_tensor(np.load(model.path_pca_VT_k), device=self.device)
            model.V_k = model.VT_k.T
            model.latent_mean = torch.as_tensor(np.load(model.path_pca_latent_mean), device=self.device)
            model.latent_std = torch.as_tensor(np.load(model.path_pca_latent_std)*2, device=self.device)
        print('[DIAG] PCA arrays: ' + ', '.join(_summary(getattr(model, key), key) for key in ('s_mean', 'VT_k', 'V_k', 'latent_mean', 'latent_std')), flush=True)
        mask = data['agent']['category'] >= 2
        print('[DIAG] ' + _summary(mask, 'eval_mask'), flush=True)
        # QCNet marginal trajectories: [selected_agents, modes, future*2].
        # Diffusion expects mm=[agents, modes, latent_dim] and
        # mean/std=[agents, latent_dim]. No extra scene dimension.
        marginal = traj[mask].reshape(-1, model.num_modes, self.future * 2)
        if marginal.shape[0] == 0:
            raise ValueError('No selected agents for diffusion sampling')
        latent = (marginal - model.s_mean) @ model.VT_k
        latent = model.normalize(latent, model.latent_mean, model.latent_std)
        if latent.ndim != 3 or latent.shape[-1] != model.latent_mean.numel():
            raise ValueError(f'Unexpected marginal latent shape: {tuple(latent.shape)}')
        mean = latent.mean(dim=1)
        std = latent.std(dim=1, unbiased=False) + model.std_reg
        if not torch.isfinite(latent).all() or not torch.isfinite(std).all():
            raise ValueError('Nonfinite diffusion conditioning tensors')
        print('[DIAG] ' + _summary(latent, 'latent') + '; ' + _summary(mean, 'mean') + '; ' + _summary(std, 'std'), flush=True)
        print(f'[DIAG] sampler contract mm={tuple(latent.shape)} mean={tuple(mean.shape)} std={tuple(std.shape)}', flush=True)
        # DDIM indexes alpha_bars[t-stride], so reverse_steps must be
        # divisible by stride to avoid negative indexing.
        schedule_steps = int(model.joint_diffusion.var_sched.num_steps)
        stride = int(model.sampling_stride)
        reverse_steps = (min(60, schedule_steps) // stride) * stride
        if reverse_steps < stride:
            raise ValueError(f'Invalid sampler schedule: steps={schedule_steps}, stride={stride}')
        print(f'[DIAG] sampler schedule steps={schedule_steps} reverse_steps={reverse_steps} stride={stride}', flush=True)
        print('[DIAG] STAGE diffusion sample BEGIN', flush=True)
        samples = model.joint_diffusion.sample(
            self.samples, data=data, scene_enc=scene_enc, mean=mean, std=std,
            mm=latent, mmscore=pred['pi'].exp()[mask], sampling=model.sampling,
            stride=stride, reverse_steps=reverse_steps, eval_mask=mask)
        print('[DIAG] STAGE diffusion sample OK; ' + _summary(samples, 'diffusion_samples'), flush=True)
        if samples.ndim != 3 or samples.shape[0] != latent.shape[0] or samples.shape[-1] != latent.shape[-1]:
            raise ValueError(f'Unexpected diffusion output shape: {tuple(samples.shape)}')
        if not torch.isfinite(samples).all():
            raise ValueError('Diffusion produced nonfinite samples')
        # latent [1, samples, 10] -> local [samples, future, 2]
        if model.cond_norm:
            samples = model.unnormalize(samples, mean, std)
        local = model.unnormalize(samples, model.latent_mean, model.latent_std)
        local = torch.matmul(local, model.V_k) + model.s_mean
        local = local.reshape(-1, self.future, 2).detach().cpu().numpy()
        theta = float(ego.orientation)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        global_paths = local @ rotation.T + _xy(ego.position)
        print('[DIAG] ' + _summary(global_paths, 'global_paths'), flush=True)
        target = _xy(target)
        start_distance = np.linalg.norm(_xy(ego.position) - target)
        # Reward the best intermediate approach; don't require the full 6s
        # horizon to terminate at the goal.
        scores = []
        for path in global_paths:
            distances = np.linalg.norm(path - target, axis=1)
            best = float(distances.min())
            progress = start_distance - best
            jumps = np.linalg.norm(np.diff(np.vstack([_xy(ego.position), path]), axis=0), axis=1)
            # Receding-horizon execution uses only the first five steps.
            # Score both immediate progress and progress over the full prediction,
            # rather than rewarding a distant point that will never be executed.
            near_index = min(4, len(path) - 1)
            near_distance = float(distances[near_index])
            immediate_progress = start_distance - near_distance
            terminal_progress = start_distance - float(distances[-1])
            scores.append(5.0 * immediate_progress + 1.5 * terminal_progress
                          + 0.5 * progress - best
                          - 0.5 * np.maximum(jumps - 3.5, 0).sum())
        print('[DIAG] ' + _summary(np.asarray(scores), 'candidate_scores'), flush=True)
        selected_index = int(np.argmax(scores))
        print(f'[DIAG] selected_candidate={selected_index}, score={scores[selected_index]:.5g}', flush=True)
        selected = global_paths[selected_index]
        return Plan(points=selected, latency=time.perf_counter()-t0, candidate_count=len(global_paths), candidates=global_paths)
