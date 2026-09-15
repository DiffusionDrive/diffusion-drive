import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.planner.diffusion_es.models.pgp.pgp_model import PGPModel
from src.planner.diffusion_es.models.pgp.pgp_encoder import PGPEncoder
from src.planner.diffusion_es.models.pgp.pgp_aggregator import PGP
from src.planner.diffusion_es.models.pgp.pgp_decoder import LVM


def create_dummy_weights():
    print("Initializing dummy Diffusion-ES architecture...")

    # We must define the exact feature dimensions we reverse-engineered (7)
    encoder = PGPEncoder(
        target_agent_feat_size=7,  # [x, y, yaw, vx, vy, width, length]
        target_agent_emb_size=32,
        target_agent_enc_size=64,
        node_feat_size=7,  # [x, y, yaw, boundary, stop, crosswalk, carpark]
        node_emb_size=32,
        node_enc_size=64,
        nbr_feat_size=7,  # [x, y, yaw, vx, vy, width, length]
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
        emb_size=128,  # The internal dimension for the attention mechanism
        num_heads=4,  # Number of attention heads
        use_route_mask=True,
        hard_masking=False,
        num_samples=1,
        num_traversals=1,
        horizon=80,
    )

    decoder = LVM(
        encoding_size=192,
        agg_type="combined",  # Changed from 'sample' to 'combine'
        lv_dim=5,
        hidden_size=128,
        num_clusters=1,
        use_ray=False,
        num_samples=1,
        op_len=80,
    )

    model = PGPModel(encoder=encoder, aggregator=aggregator, decoder=decoder)

    os.makedirs("../checkpoints/diffusion_es", exist_ok=True)
    save_path = "../checkpoints/diffusion_es/model.ckpt"
    torch.save(model.state_dict(), save_path)

    print(f"Successfully saved dummy checkpoint to {save_path}")


if __name__ == "__main__":
    create_dummy_weights()
