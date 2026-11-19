# Evaluate Qwen-drive-1.0 model

## Prepare KITScenes evaluation data for val split
[Prepare KITScenes evaluation data for val split](../kitscenes/README.md)

## Evaluate Qwen-drive-1.0 model
### Clone  Qwen-Drive-1.0 github repository
```
$ clone https://github.com/QwenLM/Qwen-Drive-1.0.git qwen-drive
```
### Create virtual environment 
```
$ cd qwen-drive
$ python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
$ pip install -r requirements.txt --no-build-isolation
$ pip install -e . --no-deps
```
### Download model weights
```
$ hf download Qwen/Qwen-Drive-1.0-4B --local-dir ./Qwen-Drive-1.0-4B
```
### Generate predictions data
```
$ export PYTHONPATH=src
```
```
$nohup python scripts/run_planning.py \
    --model Qwen-Drive-1.0-4B --planner Qwen-Drive-1.0-4B/planner-rl \
    --scenes <KITSCENES_VAL_EVAL_DIR>/kitscenes_val.jsonl \
    --image-root /tmp/$USER/ratarmount/kitscenes/data  \
    --output <OUTPUT_DIR>/qwendrive/predictions.jsonl \
    --mode reasoning_planning --num-samples 6 > <OUTPUT_DIR>/qwendrive/run.log 2>&1 & 
```

### Generate evaluation results
Navigate to the  Model/accuracy_benchmark/qwen_drive_1_0 directory and execute
```
$ $ python eval_kitscenes.py \
    --predictions <OUTPUT_DIR>/qwendrive/predictions.jsonl \
    --scenes <KITSCENES_VAL_EVAL_DIR>/kitscenes_val.jsonl \
    --output <OUTPUT_DIR>/qwendrive/metrics
```



