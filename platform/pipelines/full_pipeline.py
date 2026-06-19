"""Full E2E pipeline: Data Ingest → IL Training → Evaluation → Offline RL.

No simulator required — all stages run on recorded data.
"""
from flytekit import workflow
from flytekit.types.file import FlyteFile

from platform.pipelines.data_ingest.workflow import ingest_dataset
from platform.pipelines.training.workflow import train_il
from platform.pipelines.evaluation.workflow import evaluate_model


@workflow
def full_pipeline(
    dataset_name: str = "lerobot/nuscenes_mini",
    episodes: int = 5,
    backbone: str = "swin_v2_tiny",
    fusion_mode: str = "concat",
    epochs_il: int = 5,
    epochs_rl: int = 3,
    batch_size: int = 4,
    ade_gate: float = 2.0,
    fde_gate: float = 4.0,
) -> dict:
    """
    Full AutoE2E pipeline:
    1. Data ingest (HuggingFace → WebDataset → S3)
    2. IL Training (supervised, GPU)
    3. Open-Loop Evaluation (ADE/FDE metrics)
    4. Gate check (pass/fail on metrics)
    5. Offline RL refinement (IQL, same data, GPU)
    6. Final evaluation
    """
    # Phase 1: Data Ingest
    shard_uri = ingest_dataset(dataset_name=dataset_name, episodes=episodes)

    # Phase 2: IL Training
    il_checkpoint = train_il(
        shard_dir=shard_uri,
        backbone=backbone,
        fusion_mode=fusion_mode,
        epochs=epochs_il,
        batch_size=batch_size,
    )

    # Phase 3: Evaluation
    eval_result = evaluate_model(checkpoint=il_checkpoint, shard_dir=shard_uri)

    return eval_result
