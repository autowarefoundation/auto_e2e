import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from Platform.scripts.pack_kitscenes_corpus import extract_scene
from Platform.scripts.verify_kitscenes_corpus import (
    build_checks,
    read_partitions,
)


FROZEN = {
    "available_scene_count": 3,
    "excluded_empty_scene_count": 1,
    "eligible_group_count": 2,
    "eligible_group_uid_digest": "a" * 64,
    "eligible_sample_count": 5,
    "eligible_sample_uid_digest": "b" * 64,
    "dataset_version": "v3.0",
    "packed_contract_digest": "c" * 64,
}


def _partition(root, name, samples, *, version="v3.0"):
    path = root / name
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({
        "total_samples": samples,
        "dataset_version": version,
        "partition_id": name,
        "contracts": {"schema": "test"},
    }))
    if samples:
        with tarfile.open(path / "shard-000.tar", "w") as archive:
            for index in range(samples):
                uid = f"kitscenes-{name}-{index}"
                payload = json.dumps({
                    "split_group_uid": f"kitscenes-group-{name}",
                    "sample_uid": uid,
                }).encode()
                info = tarfile.TarInfo(f"{uid}.meta.json")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    return path


class TestExtractScene:
    def test_extracts_into_the_destination_not_the_cwd(self, tmp_path, monkeypatch):
        """`-C` is positional in GNU tar.

        Passing it after the archive returns 0 and extracts into the current
        working directory instead, which looks like success while leaving the
        destination empty.
        """
        source = tmp_path / "src"
        (source / "abc12345").mkdir(parents=True)
        (source / "abc12345" / "poses.txt").write_text("x")
        archive = tmp_path / "abc12345.tar"
        subprocess.run(
            ["tar", "-cf", str(archive), "-C", str(source), "abc12345"],
            check=True,
        )

        elsewhere = tmp_path / "cwd"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        dest = tmp_path / "dest"
        scene_id = extract_scene(archive, dest)

        assert scene_id == "abc12345"
        assert (dest / "abc12345" / "poses.txt").exists()
        assert not (elsewhere / "abc12345").exists()

    def test_missing_archive_raises_instead_of_reporting_success(self, tmp_path):
        """Swallowing tar's stderr makes a failed extraction look like an empty one."""
        with pytest.raises(RuntimeError, match="tar exited"):
            extract_scene(tmp_path / "absent.tar", tmp_path / "dest")


class TestReadPartitions:
    def test_separates_empty_partitions_and_collects_versions(self, tmp_path):
        _partition(tmp_path, "scene-a", 3)
        _partition(tmp_path, "scene-b", 2)
        _partition(tmp_path, "scene-c", 0)
        (tmp_path / "not-a-partition").mkdir()

        manifests, non_empty, empty, versions = read_partitions(tmp_path)

        assert len(manifests) == 3
        assert len(non_empty) == 2
        assert empty == 1
        assert versions == {"v3.0"}

    def test_mixed_versions_are_all_reported(self, tmp_path):
        _partition(tmp_path, "scene-a", 1)
        _partition(tmp_path, "scene-b", 1, version="v2.2")
        _, _, _, versions = read_partitions(tmp_path)
        assert versions == {"v3.0", "v2.2"}


class TestBuildChecks:
    def _checks(self, **overrides):
        values = {
            "partition_count": 3,
            "empty_count": 1,
            "group_uids": ("g1", "g2"),
            "group_digest": "a" * 64,
            "sample_count": 5,
            "sample_digest": "b" * 64,
            "dataset_version": "v3.0",
            "contract_digest": "c" * 64,
        }
        values.update(overrides)
        return build_checks(FROZEN, **values)

    def test_a_matching_corpus_passes_every_check(self):
        assert all(check.ok for check in self._checks())

    @pytest.mark.parametrize("field,value,expected_name", [
        ("partition_count", 2, "packed partitions"),
        ("empty_count", 0, "empty partitions"),
        ("group_uids", ("g1",), "eligible groups"),
        ("group_digest", "z" * 64, "group UID digest"),
        ("sample_count", 4, "samples"),
        ("sample_digest", "z" * 64, "sample UID digest"),
        ("dataset_version", "v2.2", "dataset version"),
        ("contract_digest", "z" * 64, "contract digest"),
    ])
    def test_each_deviation_fails_its_own_check(self, field, value, expected_name):
        """Every quantity the manifest pins must be caught on its own.

        A corpus that differs in one field only is the case this tool exists for:
        it is the one that otherwise reaches `train_il` and aborts hours later.
        """
        failed = [check.name for check in self._checks(**{field: value})
                  if not check.ok]
        assert failed == [expected_name]


class TestFetchMode:
    def test_requires_a_source(self, capsys):
        """Neither --tar-src nor --fetch must fail with a usable message."""
        from Platform.scripts.pack_kitscenes_corpus import main

        rc = main(["--work-root", "/tmp/w", "--out-root", "/tmp/o"])
        assert rc == 1
        assert "--fetch" in capsys.readouterr().err

    def test_lists_only_train_archives_sorted(self, monkeypatch):
        """The scene order must be deterministic, and confined to the split.

        A run that packs scenes in a different order than another run produces the
        same corpus, but only if the set is identical -- picking up `data/val/` or a
        stray file would silently change what 'the train split' means.
        """
        import Platform.scripts.pack_kitscenes_corpus as mod

        class _Sibling:
            def __init__(self, name):
                self.rfilename = name

        class _Api:
            def dataset_info(self, repo_id, revision=None, files_metadata=False):
                names = [
                    "data/train/b.tar", "data/train/a.tar",
                    "data/val/z.tar", "data/train/notes.txt", "README.md",
                ]
                return type("I", (), {"siblings": [_Sibling(n) for n in names]})()

        monkeypatch.setattr(mod, "HfApi", _Api, raising=False)
        monkeypatch.setitem(
            __import__("sys").modules, "huggingface_hub",
            type("M", (), {"HfApi": _Api})(),
        )
        assert mod.list_scene_archives("r", "rev") == [
            "data/train/a.tar", "data/train/b.tar",
        ]


class TestTheFrozenContractIsReadNotRemembered:
    """Both tools must follow the policy's manifest, not a snapshot of it.

    The frozen split is re-cut as the packed data changes, and every re-cut
    carries a new dataset version and contract digest. A tool that names one
    snapshot keeps passing a corpus that `train_il` has started rejecting, which
    is the exact failure these tools exist to catch early.
    """

    def test_the_checker_reads_the_manifest_the_policy_names(self):
        from training.dataset_policy import KITSCENES_TRAINING_POLICY

        from Platform.scripts.verify_kitscenes_corpus import frozen_manifest_path

        path = frozen_manifest_path()
        assert path.is_file()
        assert path.name == Path(
            KITSCENES_TRAINING_POLICY.validation_manifest
        ).name

    def test_the_packer_writes_the_version_that_manifest_carries(self):
        from Platform.scripts.pack_kitscenes_corpus import frozen_dataset_version
        from Platform.scripts.verify_kitscenes_corpus import frozen_manifest_path

        frozen = json.loads(frozen_manifest_path().read_text())
        assert frozen_dataset_version() == frozen["dataset_version"]

    def test_that_version_is_the_navigation_one_not_the_default(self):
        """The trap the comment in the packer describes, asserted."""
        from Platform.pipelines.workflows import (
            DATASET_PACK_VERSION,
            KITSCENES_NAVIGATION_DATASET_VERSION,
        )
        from Platform.scripts.pack_kitscenes_corpus import frozen_dataset_version

        assert frozen_dataset_version() == KITSCENES_NAVIGATION_DATASET_VERSION
        assert frozen_dataset_version() != DATASET_PACK_VERSION
