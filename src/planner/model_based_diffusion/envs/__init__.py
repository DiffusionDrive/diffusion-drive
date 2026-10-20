from brax import envs as brax_envs
from .car2d import Car2d


def get_env(env_name: str):
    if env_name == "car2d":
        return Car2d()
    else:
        raise ValueError(f"Unknown environment: {env_name}")
