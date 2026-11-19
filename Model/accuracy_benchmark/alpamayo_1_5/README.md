# Evaluate Alpamayo_1.5 model

## Prepare KITScenes evaluation data for val split

[Prepare KITScenes evaluation data for val split](../kitscenes/README.md)

## Evaluate Alpamayo_1.5 model
### Clone  Alpamayo_1.5 github repository
```
$ git clone https://github.com/NVlabs/alpamayo1.5
```
### Create virtual environment 
```
$ cd aalpamayo1.5
$ python3 -m venv .venv && source .venv/bin/activate
$ python -m pip install -U pip wheel setuptools
$ pip install -e . --no-deps
$ python - <<'EOF' \
    import tomllib \
    deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"] \
    open("requirements.txt", "w").write("\n".join(d for d in deps if not d.startswith("flash-attn")) + "\n") \
    print(open("requirements.txt").read()) \
EOF
$ pip install "torch>=2.8" torchvision
$ pip install -r requirements.txt
$ pip install ninja packaging
$ MAX_JOBS=8 pip install "flash-attn>=2.8.3" --no-build-isolation
```
### Generate predictions data and evaluation results
Navigate to the  Model/accuracy_benchmark/alpamayo_1_5 directory and execute
```
$ nohup python eval_kitscenes.py \
    --scenes <KITSCENES_VAL_EVAL_DIR>/kitscenes_val.jsonl \
    --out --out <OUTPUT_DIR>/alpamayo15/metrics \
    > <OUTPUT_DIR>/alpamayo15/run.log 2>&1 &
```
To generate only statistics if the metrics directory already exists
```
$ python eval_kitscenes.py \
    --scenes <KITSCENES_VAL_EVAL_DIR>/kitscenes_val.jsonl \
    --out --out <OUTPUT_DIR>/alpamayo15/metrics \
    --report-only
```
