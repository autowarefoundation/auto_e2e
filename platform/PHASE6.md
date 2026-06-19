# AutoE2E Phase 6: Reinforcement Learning Post-Training

Status: RESEARCH COMPLETE — ready for planning review.

## Background

End-to-end AD models trained via imitation learning (IL) are brittle in
safety-critical long-tail scenarios. RL post-training addresses this by
letting the model learn from the consequences of its own actions in simulation.

## NVIDIA AlpaGym (Key Reference)

**AlpaGym** is NVIDIA's open-source closed-loop RL framework for AV models,
released as part of the Alpamayo platform (May 2026).

| Item | Details |
|------|---------|
| Repo | [NVlabs/alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes) |
| License | **Open source** (released at GTC 2026 as part of Alpamayo Open Platform) |
| Simulator | AlpaSim (NVIDIA, closed-loop replay with neural rendering) |
| RL Algorithm | **GRPO** (Group Relative Policy Optimization) via Cosmos-RL |
| Training framework | Cosmos-RL (distributed RL, async rollouts) |
| Requirements | Multi-GPU (rollout workers + trainer), NCCL, Redis |
| Data | Physical AI AV NuRec dataset (same dataset we already use!) |

### How AlpaGym Works

```
1. IL-trained model (e.g., Alpamayo, or AutoE2E)
2. AlpaSim runs closed-loop rollouts with current policy
3. Reward computed per episode (progress, collision, offroad, comfort)
4. GRPO updates policy (like PPO but for VLA/trajectory models)
5. Iterate → model improves on long-tail scenarios
```

### AlpaGym Reward Example

```yaml
terms:
  - kind: metric
    metric_name: progress
    scale: 1.0
  - kind: metric
    metric_name: collision_any
    scale: -10.0
  - kind: metric
    metric_name: offroad
    scale: -5.0
```

## RL Approaches for End-to-End AD

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **GRPO** (AlpaGym) | Group relative policy optimization | State-of-art for VLA models, stable training | Requires AlpaSim |
| **PPO** | Proximal policy optimization | Well-understood, widely used | Hyperparameter sensitive |
| **DAgger** | Dataset aggregation (online IL) | Simple, iterative | Not true RL (no reward signal) |
| **GAIL** | Generative adversarial IL | Learns reward implicitly | Unstable training |
| **Offline RL (CQL, IQL)** | RL from logged data | No simulator needed | Conservative, limited exploration |

**Recommendation for AutoE2E**: Start with **PPO + CARLA** (Phase 5 already
provides the closed-loop infrastructure), then migrate to **AlpaGym + AlpaSim**
when the model is mature enough for high-fidelity RL.

## Concrete Plan for AutoE2E

### Stage 1: PPO + CARLA (Builds on Phase 5)

Use Phase 5's CARLA infrastructure directly as the RL environment.

```python
# Gym-style wrapper around CARLA (existing pattern in RL AD research)
class CarlaEnv(gym.Env):
    """Wraps CARLA scenario into OpenAI Gym interface."""
    
    def __init__(self, carla_host, scenario_config):
        self.observation_space = spaces.Dict({
            "visual_tiles": spaces.Box(0, 1, shape=(7, 3, 256, 256)),
            "egomotion_history": spaces.Box(-inf, inf, shape=(256,)),
        })
        self.action_space = spaces.Box(-1, 1, shape=(2,))  # [accel, curvature]
    
    def step(self, action):
        # Apply action to CARLA, get next obs + reward
        accel, curvature = action
        # ... CARLA control loop (from closed_loop_runner.py) ...
        reward = self._compute_reward()
        return obs, reward, done, truncated, info
    
    def _compute_reward(self):
        # Progress + safety penalties
        reward = self.progress_delta * 1.0
        reward -= self.collision * 10.0
        reward -= self.offroad * 5.0
        reward -= self.jerk_penalty * 0.1
        return reward
```

Training loop:
```python
from stable_baselines3 import PPO

env = CarlaEnv(carla_host="carla-server", scenario_config=config)
model = PPO("MultiInputPolicy", env, verbose=1,
            n_steps=512, batch_size=64, learning_rate=1e-5)
model.learn(total_timesteps=100_000)
```

**Infrastructure requirement**: Same as Phase 5.
- CARLA server (g5.xlarge simulation node)
- Training pod (CPU for 10Hz inference during rollout)
- Additional: RL trainer pod (GPU for PPO updates, can share training node)

### Stage 2: AlpaGym + AlpaSim (Future)

When AutoE2E model is mature:
1. Export checkpoint in Alpamayo-compatible format
2. Configure AlpaSim scenes from Physical AI NuRec dataset
3. Define reward in YAML
4. Run `alpagym_host.cli` with AutoE2E as the policy

Requirements:
- Multi-GPU (8x A100/H100 recommended for large-scale RL)
- AlpaSim license (open source, but runtime needs NVIDIA GPU)
- Model adaptation to Alpamayo driver interface

### Stage 3: RLHF / Reward Model (Long-term)

Train a reward model from human driving preferences:
- Collect pairs of rollout clips
- Human annotators rank which driving is better
- Train reward model
- Use as RL reward signal (replaces hand-crafted reward)

## Reward Design for AutoE2E

| Term | Signal | Weight | Rationale |
|------|--------|--------|-----------|
| Progress | distance_along_route / dt | +1.0 | Encourage forward motion |
| Collision | binary per step | -10.0 | Hard safety constraint |
| Offroad | wheels outside drivable area | -5.0 | Stay on road |
| Comfort (jerk) | |d(accel)/dt| > threshold | -0.5 | Smooth driving |
| Comfort (lat) | |curvature * v²| > threshold | -0.3 | Avoid harsh steering |
| Red light | binary | -10.0 | Traffic rule compliance |
| Speed limit | v > limit | -1.0 | Legal speed |

## Implementation Timeline

| Step | Effort | Dependency |
|------|--------|-----------|
| CarlaEnv gym wrapper | 2 days | Phase 5 CARLA infra |
| Reward function | 1 day | Metrics from closed_loop_runner.py |
| PPO training loop | 2 days | stable-baselines3 or cleanrl |
| RL training on EKS | 1 day | Same Kueue/GPU infra |
| Evaluate RL policy | 1 day | Phase 4 eval pipeline |
| AlpaGym migration | 1 week | Multi-GPU, AlpaSim setup |

## Key Papers

1. **Alpamayo-R1** (NVIDIA, 2025): "Bridging Reasoning and Action Prediction
   for Generalizable Autonomous Driving in the Long Tail" — RL post-training
   improves reasoning quality by 45%.
   https://arxiv.org/html/2511.00088

2. **ThinkDrive** (2024): "RL enables reasoning-then-planning for end-to-end
   driving." Uses PPO with CARLA simulator.

3. **DriveVLM** (2024): "Vision-Language-Action models for driving."
   Post-trained with DPO (Direct Preference Optimization).

4. **Hydra-MDP** (NVIDIA, 2024): "End-to-end driving at scale with
   multi-decoder policy." Shows RL fine-tuning improves closed-loop metrics.

## Open Questions

1. **GPU budget for RL**: PPO + CARLA can run on single g5.xlarge (rollout) +
   g6e (gradient updates). AlpaGym needs 8+ GPUs. Start small.
2. **When to start RL**: After IL model reaches reasonable open-loop metrics
   (ADE@3s < 1.5m). RL before this wastes compute on a bad starting policy.
3. **Sim-to-real gap**: CARLA rendering ≠ real cameras. Domain randomization
   (weather, lighting, camera noise) helps. AlpaSim uses neural rendering
   from real data (NuRec), reducing this gap.
4. **Reward hacking**: Model may find exploits (e.g., driving very slowly to
   avoid collisions). Need minimum speed reward term.
