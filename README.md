# Safe Interactive Artificial Intelligence for Safe Autonomous Driving

## Sub-Modules
* **Student A:** Simulation Environment (CommonRoad)
* **Student B:** Safe Trajectory Planner (Diffusion Models)
* **Student C:** Human-Machine Interface (LLMs)
* **Student D:** Vehicle Platooning

## Local Setup Guide
1. Clone this repository.
2. Create a virtual environment: `conda create -n safe_av python=3.10 -y`
3. Activate the environment: `conda activate safe_av`
4. Install the repository and dependencies in editable mode:
   ```bash
   pip install -e .[dev]

## Repository Structure

```text
EE4002D/
├── .github/
├── configs/                         # YAML config files for scenarios & models
├── docs/                            # Architecture diagrams & module specs
├── scenarios/                       # CommonRoad XML scenarios & NUS campus maps
├── src/
│   ├── __init__.py
│   ├── core/                        # Shared data contracts and transforms
│   │   ├── __init__.py
│   │   └── types.py                 # Trajectory, State, and Map dataclasses
│   ├── hmi/                         # Student C: LLM parser & visual GUI
│   │   └── __init__.py
│   ├── planner/                     # Student B: Diffusion model & safety tracker
│   │   ├── __init__.py
│   │   ├── models/                  # Diffusion architectures (e.g., U-Net, DiT)
│   │   └── guidance/                # Safety cost functions and gradient logic
│   ├── platooning/                  # Student D: Distributed platooning logic
│   │   └── __init__.py
│   └── simulator/                   # Student A: CommonRoad environment loaders
│       └── __init__.py
├── tests/                           # Pytest unit and integration tests
├── .gitignore                       # Ignored files (weights, IDE settings)
├── .pre-commit-config.yaml          # Git hooks for automated Ruff formatting
├── main.py                          # Master simulation orchestration runner
├── pyproject.toml                   # Project dependencies and package config
└── README.md                        # Project setup and documentation
