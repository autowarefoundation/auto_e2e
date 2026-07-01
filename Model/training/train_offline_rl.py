"""Offline RL post-training using Implicit Q-Learning (IQL).

Uses the same WebDataset shards from data ingest (no simulator needed).
Refines IL-pretrained policy by learning from expert demonstrations via
conservative Q-function estimation.

Usage:
  python train_offline_rl.py --shard-dir /data/shards --pretrained /ckpt/best.pt
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import webdataset as wds

# MLflow logging
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class ImplicitQLearning:
    """Simplified IQL for trajectory refinement."""

    def __init__(self, policy, lr=3e-4, tau=0.7, beta=3.0, gamma=0.99, device="cuda"):
        self.policy = policy.to(device)
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.device = device

        # Value and Q networks (simple MLPs on top of policy features)
        feat_dim = 512  # Matches policy backbone output
        self.vf = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)
        self.qf = nn.Sequential(nn.Linear(feat_dim + 4, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)

        self.policy_opt = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.vf_opt = torch.optim.AdamW(self.vf.parameters(), lr=lr)
        self.qf_opt = torch.optim.AdamW(self.qf.parameters(), lr=lr)

    def expectile_loss(self, diff, expectile):
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * diff.pow(2)).mean()

    def update(self, batch):
        states, actions, rewards, next_states = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        # Extract features from policy backbone
        with torch.no_grad():
            feats = self.policy.extract_features(states)
            next_feats = self.policy.extract_features(next_states)

        # Q-function update
        q_input = torch.cat([feats, actions], dim=-1)
        q_pred = self.qf(q_input).squeeze(-1)
        with torch.no_grad():
            next_v = self.vf(next_feats).squeeze(-1)
            q_target = rewards + self.gamma * next_v
        qf_loss = F.mse_loss(q_pred, q_target)

        self.qf_opt.zero_grad()
        qf_loss.backward()
        self.qf_opt.step()

        # Value function update (expectile regression)
        v_pred = self.vf(feats.detach()).squeeze(-1)
        with torch.no_grad():
            q_input2 = torch.cat([feats, actions], dim=-1)
            q_val = self.qf(q_input2).squeeze(-1)
        vf_loss = self.expectile_loss(q_val - v_pred, self.tau)

        self.vf_opt.zero_grad()
        vf_loss.backward()
        self.vf_opt.step()

        # Policy update (advantage-weighted regression)
        pred_actions = self.policy(states)
        with torch.no_grad():
            v_val = self.vf(feats).squeeze(-1)
            q_input3 = torch.cat([feats, actions], dim=-1)
            advantage = self.qf(q_input3).squeeze(-1) - v_val
            weights = torch.exp(self.beta * advantage).clamp(max=100.0)

        policy_loss = (weights * F.mse_loss(pred_actions, actions, reduction='none').mean(-1)).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        return {"qf_loss": qf_loss.item(), "vf_loss": vf_loss.item(), "policy_loss": policy_loss.item()}


def compute_rewards(actions, next_actions):
    """Compute proxy rewards from expert demonstrations (smoothness + progress)."""
    # Reward = low jerk (smooth) + forward progress (high acceleration)
    jerk = (next_actions - actions).pow(2).sum(-1)
    progress = actions[:, 0]  # acceleration component
    return progress - 0.1 * jerk


def build_dataloader(shard_dir, batch_size):
    """Build WebDataset DataLoader with (state, action, reward, next_state) tuples."""
    shards = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".tar")])
    if not shards:
        raise FileNotFoundError(f"No .tar shards in {shard_dir}")

    def decode(sample):
        import pickle
        import io
        import numpy as np
        meta = pickle.loads(sample["meta.pkl"])
        imgs = []
        for cam in ["front", "left", "right"]:
            key = f"{cam}.jpg"
            if key in sample:
                from PIL import Image
                img = Image.open(io.BytesIO(sample[key])).resize((224, 224))
                arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                imgs.append(torch.from_numpy(arr))
        if not imgs:
            imgs = [torch.zeros(3, 224, 224)]
        stacked = torch.cat(imgs, dim=0)  # (9, 224, 224) for 3 cameras
        action = torch.tensor(meta.get("action", [0.0, 0.0, 0.0, 0.0]), dtype=torch.float32)
        return stacked, action

    dataset = wds.WebDataset(shards).decode().map(decode)
    return DataLoader(dataset, batch_size=batch_size, num_workers=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--pretrained", default=None, help="IL pretrained checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", default="/tmp/ckpt-rl")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load IL model
    from Model.model import VisionPilot
    policy = VisionPilot(backbone="swin_v2_tiny", fusion_mode="concat", pretrained=False)
    if args.pretrained and os.path.exists(args.pretrained):
        policy.load_state_dict(torch.load(args.pretrained, map_location="cpu"))

    iql = ImplicitQLearning(policy, lr=args.lr, device=device)
    loader = build_dataloader(args.shard_dir, args.batch_size)

    if HAS_MLFLOW:
        mlflow.set_experiment("auto-e2e-offline-rl")
        mlflow.start_run(run_name="iql-refinement")
        mlflow.log_params({"epochs": args.epochs, "lr": args.lr, "tau": iql.tau, "beta": iql.beta})

    for epoch in range(args.epochs):
        metrics_accum = {"qf_loss": 0, "vf_loss": 0, "policy_loss": 0}
        steps = 0
        prev_actions = None
        for states, actions in loader:
            if prev_actions is None:
                prev_actions = actions
                continue
            rewards = compute_rewards(prev_actions, actions)
            batch = (states, actions, rewards, states)  # s'≈s for consecutive frames
            m = iql.update(batch)
            for k, v in m.items():
                metrics_accum[k] += v
            steps += 1
            prev_actions = actions

        if steps > 0:
            avg = {k: v / steps for k, v in metrics_accum.items()}
            print(f"Epoch {epoch+1}/{args.epochs} | " + " | ".join(f"{k}={v:.4f}" for k, v in avg.items()))
            if HAS_MLFLOW:
                mlflow.log_metrics(avg, step=epoch)

    # Save
    save_path = os.path.join(args.save_dir, "policy_rl.pt")
    torch.save(policy.state_dict(), save_path)
    print(f"Saved RL-refined policy to {save_path}")

    if HAS_MLFLOW:
        mlflow.log_artifact(save_path)
        mlflow.end_run()


if __name__ == "__main__":
    main()
