# AutoE2E Phase 5 Platform Design: Closed-Loop Simulation (CARLA)

Status: DRAFT — ready for review.

## Goal

Models promoted to `staging` (Phase 4 gate) are tested in closed-loop driving
scenarios in CARLA before final promotion to `champion`. This catches failures
that open-loop metrics miss: reaction to dynamic agents, recovery from
perturbations, and multi-step planning coherence.

## Why Closed-Loop

Open-loop (Phase 4) measures prediction accuracy on recorded data.
Closed-loop measures **driving ability** — the model's predictions are executed
in a simulator, and the resulting vehicle state becomes the next input.
Compounding errors, oscillation, and failure to react to other agents only
surface in closed-loop.

## Architecture

```
MLflow Registry (alias: staging)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Flyte Simulation Pipeline                                            │
│                                                                       │
│  1. provision_carla     CARLA server Pod (GPU, headless, g5.xlarge)    │
│  2. load_model          Download staging checkpoint → client Pod       │
│  3. run_scenarios       N scenarios in parallel (ScenarioRunner)       │
│  4. collect_results     Route completion, collisions, comfort, time    │
│  5. aggregate_report    Summary → MLflow + optional Grafana            │
│  6. teardown            Kill CARLA server Pod                          │
│                                                                       │
│  Compute: Simulation NodePool (g5.xlarge, scale-to-zero, Karpenter)   │
│  Model Pod: CPU-only (lightweight inference at 10Hz is fine on CPU)    │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
Pass all scenarios → promote to "champion" (manual review + alias set)
```

## CARLA Deployment on K8s

### Server Pod (GPU)

CARLA requires GPU for rendering (even headless uses Vulkan offscreen).

```yaml
# k8s/carla-server.yaml
apiVersion: v1
kind: Pod
metadata:
  name: carla-server
  namespace: auto-e2e-training
spec:
  nodeSelector:
    workload-type: simulation
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  containers:
    - name: carla
      image: carlasim/carla:0.9.15
      command: ["/bin/bash", "-c"]
      args:
        - ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000
      ports:
        - containerPort: 2000  # RPC
        - containerPort: 2001  # streaming
      resources:
        requests:
          cpu: "4"
          memory: "16Gi"
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
      env:
        - name: SDL_VIDEODRIVER
          value: "offscreen"
```

### Client Pod (CPU — model inference + scenario control)

AutoE2E at 10Hz inference is lightweight (~40ms on CPU for single-batch).
No GPU needed for the client — keeps simulation costs down.

```yaml
# Runs alongside CARLA server, connects via RPC
containers:
  - name: client
    image: <ECR>/auto-e2e/training:latest
    command: ["python", "Model/evaluation/closed_loop_runner.py"]
    args:
      - "--carla-host=carla-server"
      - "--checkpoint=s3://checkpoints/staging.pt"
      - "--scenarios=town01_straight,town03_intersection"
    resources:
      requests:
        cpu: "4"
        memory: "8Gi"
```

### Simulation NodePool (Karpenter)

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: simulation
spec:
  template:
    metadata:
      labels:
        workload-type: simulation
    spec:
      nodeClassRef:
        group: eks.amazonaws.com
        kind: NodeClass
        name: default
      requirements:
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["g5.xlarge"]  # A10G 24GB — sufficient for CARLA headless
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot"]  # Spot OK for simulation (can retry)
      taints:
        - key: nvidia.com/gpu
          effect: NoSchedule
  limits:
    nvidia.com/gpu: "2"  # Max 2 concurrent sim servers
  disruption:
    consolidateAfter: 5m  # Scale to zero when idle
```

Key difference from training NodePool:
- **Spot instances** OK (simulation is idempotent, can retry on interruption)
- **Scale-to-zero** (no warm node — sim runs infrequently)
- **g5.xlarge** (A10G, cheaper than g6e L40S, sufficient for CARLA rendering)

## Closed-Loop Metrics

| Metric | Definition | Pass Threshold |
|--------|-----------|----------------|
| Route Completion | % of route successfully driven | ≥ 90% |
| Collision Rate | Collisions per km | ≤ 0.5 |
| Red Light Violations | Per scenario | 0 |
| Comfort (jerk) | Mean absolute jerk | ≤ 2.5 m/s³ |
| Comfort (lat accel) | Max lateral acceleration | ≤ 3.0 m/s² |
| Timeout | Scenario completed in time | Yes |

## Scenario Suite

Initial scenario set (expand over time):

| ID | Town | Description | Difficulty |
|----|------|-------------|------------|
| S01 | Town01 | Straight road, no traffic | Easy |
| S02 | Town01 | Follow lead vehicle | Easy |
| S03 | Town03 | Unprotected left turn | Medium |
| S04 | Town03 | Intersection with cross-traffic | Medium |
| S05 | Town05 | Lane change on highway | Medium |
| S06 | Town03 | Pedestrian crossing | Hard |
| S07 | Town05 | Cut-in vehicle | Hard |

Each scenario: 60s max duration, deterministic seed for reproducibility.

## Model ↔ CARLA Interface

```python
# Closed-loop control at 10Hz:
while not done:
    # 1. Get observations from CARLA
    cameras = [get_camera(cam) for cam in camera_sensors]  # 7 RGB images
    ego_state = get_ego_vehicle_state()  # speed, heading, etc.

    # 2. Preprocess → model input format
    visual_tiles = preprocess_cameras(cameras)  # (1, 7, 3, 256, 256)
    egomotion_history = update_ego_buffer(ego_state)  # rolling (1, 256)

    # 3. Model inference (CPU, ~40ms)
    trajectory = model(visual_tiles, visual_history, egomotion_history)
    accel, curvature = trajectory[0, 0], trajectory[0, 1]  # next-step only

    # 4. Convert to CARLA control
    throttle, brake = accel_to_throttle_brake(accel)
    steer = curvature_to_steer(curvature, speed)
    carla_vehicle.apply_control(carla.VehicleControl(
        throttle=throttle, brake=brake, steer=steer
    ))

    # 5. Tick simulator
    world.tick()
    time.sleep(0.1)  # 10Hz
```

## Flyte Workflow

```python
@task(requests=Resources(cpu="4", mem="16Gi", gpu="1"),
      labels={"kueue.x-k8s.io/queue-name": "gpu-queue"})
def run_carla_scenarios(checkpoint_s3: str, scenarios: List[str]) -> dict:
    """Start CARLA server, run scenarios, collect results."""
    ...

@workflow
def closed_loop_eval(checkpoint_s3: str) -> bool:
    results = run_carla_scenarios(checkpoint_s3=checkpoint_s3, scenarios=DEFAULT_SCENARIOS)
    return promote_to_champion(results=results)
```

## Implementation Plan

1. **Simulation NodePool** (Terraform): g5.xlarge spot, scale-to-zero, new taint.
2. **CARLA Docker**: Use official `carlasim/carla:0.9.15` (no custom build needed).
3. **closed_loop_runner.py** (`Model/evaluation/`): CARLA client, model inference
   loop, metric collection.
4. **Scenario definitions** (`Model/evaluation/scenarios/`): YAML per scenario
   (town, spawn point, traffic config, success criteria).
5. **Flyte workflow** (`platform/pipelines/simulation/workflow.py`).
6. **E2E verify**: staging model → CARLA → scenario S01 (straight road) passes.

## Cost Considerations

- g5.xlarge spot: ~$0.40/hr (vs $1.01 on-demand). 70% savings.
- Scale-to-zero: node only exists during simulation (~10 min per eval cycle).
- 7 scenarios × ~60s each = ~7 min CARLA time. Total cost per eval: ~$0.05.
- No warm node — acceptable since sim runs after training (which takes hours).

## Open Questions

1. **CARLA version**: 0.9.15 is latest stable. 0.9.16 may have better headless
   support. Pin to 0.9.15 initially.
2. **g5 availability**: us-west-2 has good g5 spot supply. If capacity issues,
   fall back to g4dn.xlarge (T4, slower but cheaper).
3. **Camera calibration**: CARLA cameras can be configured to match L2D extrinsics.
   Initial impl uses default positions; calibration matching is future work.
4. **Multi-scenario parallelism**: Single CARLA server runs scenarios sequentially.
   For parallelism, spawn multiple server pods (up to NodePool limit of 2 GPUs).
5. **When to trigger**: After Phase 4 gate passes (automatic) or manual from Flyte UI.
   Start with manual; automate in Phase 6.
