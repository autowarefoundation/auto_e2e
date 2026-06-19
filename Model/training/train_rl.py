"""RL Post-Training for AutoE2E via PPO + CARLA.

Loads an IL-pretrained checkpoint and fine-tunes with PPO in CARLA.
Uses stable-baselines3 with a custom policy that wraps AutoE2E.

Usage:
    python train_rl.py \
        --checkpoint s3://checkpoints/il_best.pt \
        --carla-host carla-server \
        --total-timesteps 100000 \
        --save-dir /tmp/rl_ckpt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def parse_args():
    p = argparse.ArgumentParser(description="RL Post-Training (PPO + CARLA)")
    p.add_argument("--checkpoint", required=True, help="IL checkpoint (local path or s3://)")
    p.add_argument("--carla-host", default="carla-server")
    p.add_argument("--carla-port", type=int, default=2000)
    p.add_argument("--town", default="Town01")
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--n-steps", type=int, default=512, help="Rollout steps per update")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--save-dir", default="/tmp/rl_ckpt")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def download_checkpoint(checkpoint_path: str) -> str:
    """Download from S3 if needed, return local path."""
    if checkpoint_path.startswith("s3://"):
        import boto3
        bucket, key = checkpoint_path.replace("s3://", "").split("/", 1)
        local = "/tmp/il_checkpoint.pt"
        boto3.client("s3").download_file(bucket, key, local)
        return local
    return checkpoint_path


def run_rl_training(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from gymnasium import spaces

    from evaluation.carla_env import CarlaEnv
    from model_components.auto_e2e import AutoE2E

    # Load IL checkpoint
    local_ckpt = download_checkpoint(args.checkpoint)
    il_state = torch.load(local_ckpt, map_location=args.device, weights_only=False)

    # Custom feature extractor that uses AutoE2E encoder
    class AutoE2EExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: spaces.Dict):
            # Output features dim = 256 (embed_dim)
            super().__init__(observation_space, features_dim=256)
            self.model = AutoE2E(
                backbone="swin_v2_tiny", num_views=7, embed_dim=256,
                fusion_mode="concat", is_pretrained=False,
            )
            self.model.load_state_dict(il_state["model"], strict=False)
            # Freeze backbone (only fine-tune planner head)
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        def forward(self, observations):
            visual = observations["visual_tiles"]
            ego_hist = observations["egomotion_history"]
            vis_hist = observations["visual_history"]
            # Get hidden state from model (ego_hidden, 256-dim)
            _, ego_hidden, _ = self.model(visual, vis_hist, ego_hist, mode="eval")
            return ego_hidden

    # Create environment
    env = CarlaEnv(
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        town=args.town,
    )

    # Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=AutoE2EExtractor,
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    # PPO
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        verbose=1,
        device=args.device,
    )

    # MLflow logging
    mlflow_active = os.environ.get("MLFLOW_TRACKING_URI") and True
    if mlflow_active:
        import mlflow
        mlflow.set_experiment("auto_e2e/rl")
        mlflow.start_run()
        mlflow.set_tag("stage", "RL")
        mlflow.set_tag("rl_algo", "PPO")
        mlflow.set_tag("base_checkpoint", args.checkpoint)
        mlflow.log_params({
            "total_timesteps": args.total_timesteps,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "town": args.town,
        })

    # Train
    print(f"Starting PPO training: {args.total_timesteps} timesteps")
    model.learn(total_timesteps=args.total_timesteps)

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "rl_policy.zip")
    model.save(save_path)

    # Also save just the AutoE2E weights (for eval pipeline compatibility)
    extractor = model.policy.features_extractor
    ae2e_ckpt = os.path.join(args.save_dir, "rl_checkpoint.pt")
    torch.save({"model": extractor.model.state_dict()}, ae2e_ckpt)
    print(f"Saved: {save_path}, {ae2e_ckpt}")

    if mlflow_active:
        mlflow.log_artifact(ae2e_ckpt)
        mlflow.end_run()

    env.close()


def main():
    run_rl_training(parse_args())


if __name__ == "__main__":
    main()
