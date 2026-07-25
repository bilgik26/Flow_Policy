"""
Standalone evaluation script for a trained LIBERO Flow_Policy checkpoint.

Usage:
    python -m maniflow.workspace.eval_maniflow_libero_workspace \
        --config-name maniflow_image_timm_policy_libero \
        task=libero_spatial \
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
    from maniflow.workspace.train_maniflow_libero_workspace import (
        TrainManiFlowLiberoWorkspace,
    )

    workspace = TrainManiFlowLiberoWorkspace(cfg)
    workspace.eval(mode=cfg.get("eval_mode", "best"), eval_dir_tag=cfg.get("eval_dir_tag", ""))


if __name__ == "__main__":
    main()
