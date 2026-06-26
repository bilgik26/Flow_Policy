"""
Standalone evaluation script for a trained RoboCasa Flow_Policy checkpoint.

Usage:
    python -m maniflow.workspace.eval_maniflow_robocasa_workspace \
        --config-name maniflow_image_timm_policy_robocasa \
        task=robocasa_multitask \
        training.resume=True
"""

import pathlib
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent / "config"),
)
def main(cfg):
    from maniflow.workspace.train_maniflow_robocasa_workspace import (
        TrainManiFlowRoboCasaWorkspace,
    )

    workspace = TrainManiFlowRoboCasaWorkspace(cfg)
    workspace.eval(mode=cfg.get("eval_mode", "best"))


if __name__ == "__main__":
    main()
