"""Process-parallel labeling worker correctness (#98 labeling audit).

Guards two silent-failure classes in the ProcessPool labeler:
  * init_worker's positional args must map exactly to what workflows.py passes
    (a swap would build a different dataset / cache and mislabel silently);
  * an abstained record must NOT be cached (so a re-run retries the bad sample
    instead of reusing the failure).
No network / GPU: dependencies are monkeypatched.
"""

from __future__ import annotations

import data_processing.reasoning_label_generation.parallel_label as pl
from data_processing.reasoning_label_generation.schema import ReasoningLabelRecord


def test_init_worker_positional_arg_mapping(monkeypatch):
    captured = {}

    class _FakeDS:
        def __init__(self, repo_id, episodes, include_world_model_windows):
            captured["repo_id"] = repo_id
            captured["episodes"] = episodes
            captured["wm"] = include_world_model_windows

    def _fake_build_teacher(teacher, **kw):
        captured["teacher"] = teacher
        captured["teacher_kwargs"] = kw
        return object()

    class _FakeCache:
        def __init__(self, bucket, dataset, teacher, prompt_version):
            captured["cache_bucket"] = bucket
            captured["cache_dataset"] = dataset
            captured["cache_teacher"] = teacher
            captured["cache_prompt"] = prompt_version

    monkeypatch.setattr("data_parsing.l2d.L2DDataset", _FakeDS, raising=False)
    monkeypatch.setattr(
        "data_processing.reasoning_label_generation.teacher_client.build_teacher",
        _fake_build_teacher)
    monkeypatch.setattr(
        "data_processing.reasoning_label_generation.label_cache.LabelCache",
        _FakeCache)

    # EXACT tuple workflows.py builds: (repo_id, episodes, dataset_name, teacher,
    # teacher_kwargs, cache_bucket, prompt_version).
    pl.init_worker("yaak-ai/L2D", [0, 1, 2], "yaak-ai/L2D", "openai_compatible",
                   {"base_url": "u", "strict": False}, "bkt", "promptX")

    assert captured["repo_id"] == "yaak-ai/L2D"
    assert captured["episodes"] == [0, 1, 2]
    assert captured["wm"] is True
    assert captured["teacher"] == "openai_compatible"
    assert captured["cache_bucket"] == "bkt"
    assert captured["cache_dataset"] == "yaak-ai/L2D"   # dataset_name slot
    assert captured["cache_teacher"] == "openai_compatible"
    assert captured["cache_prompt"] == "promptX"


def test_abstained_record_is_not_cached(monkeypatch):
    puts = []

    class _Cache:
        def get(self, k):
            return None

        def put(self, k, r):
            puts.append(k)

    class _Client:
        def label(self, req):
            return ReasoningLabelRecord.abstain(
                sample_id=req.sample_id, dataset_name="l2d",
                teacher_provider="openai_compatible", teacher_model="m",
                prompt_version="v", request_mode="clip_horizons",
                teacher_error="malformed")

    class _DS:
        def __getitem__(self, i):
            import torch
            f = torch.zeros(4, 6, 3, 8, 8)
            return {"history_frames": f, "future_frames": f}

    monkeypatch.setattr(pl, "_CACHE", _Cache())
    monkeypatch.setattr(pl, "_CLIENT", _Client())
    monkeypatch.setattr(pl, "_DS", _DS())
    monkeypatch.setattr(pl, "_DATASET_NAME", "l2d")

    si, rec_json, status = pl.label_sample(7)
    assert si == 7
    assert status == "abstained"
    assert rec_json["abstained"] is True
    assert puts == []          # abstentions never cached -> re-run retries them
