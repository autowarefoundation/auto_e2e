# Drive-JEPA evaluation

Evaluate the official perception-based ViT-L NAVSIM v2 checkpoint with 20 m
route commands on AutoE2E v3.5's 117 validation scenes / 11,035 samples.

## Setup

Tested on Windows with Python 3.11 and an RTX 3070 (8 GB).
Accept the [KITScenes access terms](https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal),
then run these commands from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv-drive-jepa
.\.venv-drive-jepa\Scripts\Activate.ps1
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r Model/accuracy_benchmark/drive_jepa/requirements.txt
$env:PYTHONPATH = "$PWD\Model"
$env:PYTHONUTF8 = "1"
hf auth login
python -m accuracy_benchmark.drive_jepa.eval_kitscenes --setup
```

Setup downloads the model checkpoint and required source repositories into `.benchmark/`.

## Run

First obtain AutoE2E v3.5 navigation exports at
`<navigation-root>/<scene_uuid>/scene_navigation.json` for all validation scenes.
These require the separate [Lanelet2 preprocessing step](../../data_parsing/kit_scenes/navigation.py);
raw KITScenes and the Hugging Face mirror do not include these files.

```powershell
python -m accuracy_benchmark.drive_jepa.eval_kitscenes --generate-commands --navigation-root D:/v3.5_navigation
python -m accuracy_benchmark.drive_jepa.eval_kitscenes
```

The first command creates 20 m labels from the offline reconstructed routes.
Use `--command-file path/to/commands.json` to reuse an existing generated file.

Evaluation streams raw scenes one at a time: about **473 GiB downloaded**, with
roughly **35 GiB free space** needed in addition to weights and caches.
For existing data, use `--raw-root D:/data/kitscenes` with layout `data/val/<scene_uuid>/`.
Use `--scene <scene_uuid>` for a subset run.

Outputs are saved in `.benchmark/results/drive_jepa_v2_20m/`.
Run the same command again to resume, or use `--output` to choose another directory.
