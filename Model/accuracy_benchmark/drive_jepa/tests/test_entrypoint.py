import numpy as np
import pytest

from accuracy_benchmark.drive_jepa.eval_kitscenes import parse_args
from accuracy_benchmark.drive_jepa.prepare_inputs import prepare


def test_default_and_missing_command_options():
    args = parse_args([])
    assert args.variant == "perception-based-v2"
    assert args.commands == "20m" and args.precision == "fp32"
    assert "perception_adapter" in args.adapter
    assert args.stream and args.command == [0.0] * 4
    args = parse_args(["--variant", "perception-free-v1", "--commands", "missing"])
    assert args.precision == "bf16" and args.command_file is None
    with pytest.raises(SystemExit):
        parse_args(["--precision", "bf16"])
    with pytest.raises(SystemExit):
        parse_args(["--commands", "missing", "--command-file", "route.json"])


def test_bundled_inputs_have_exact_population_and_refuse_overwrite(tmp_path):
    baseline = prepare(tmp_path)
    assert len(list(baseline.glob("*.npz"))) == 117
    with np.load(next(baseline.glob("*.npz")), allow_pickle=False) as data:
        assert set(data.files) == {"sample_uid", "timestamp_ns"}
    prepare(tmp_path)
    commands = tmp_path / "route_inputs_20m/commands.json"
    commands.write_text("{}")
    with pytest.raises(ValueError, match="Existing file differs"):
        prepare(tmp_path)
