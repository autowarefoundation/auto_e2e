# Prepare KITScenes val split for evaluation

## Download KITScenes-Multimodal val subset 

### Clone KITScenes github repository

```
$ git clone https://github.com/KIT-MRT/kitscenes.git
```

### Create virtual environment

```
$ cd kitscenes
$ python3 -m venv .venv && source .venv/bin/activate
```
### Install requirements
```
$ python -m pip install -U pip
$ pip install -U huggingface_hub hf_xet
$ pip install ratarmount
$ pip install -e .
```
### Download KITScenes validation split
```
$ export KITSCENES_ROOT=<KITSCENES_ROOT_DIR>
$ hf download KIT-MRT/KITScenes-Multimodal \
    --repo-type dataset \
    --include "data/val/*" \
    --local-dir "$KITSCENES_ROOT"
```
### Mount the root directory
```
$ bash scripts/mount_kitscenes_tars.sh <KITSCENES_ROOT_DIR>
$ export KITSCENES_ROOT=/tmp/$USER/ratarmount/kitscenes
```
If needed umount the directory
```
$ for d in /tmp/$USER/ratarmount/kitscenes/data/*/; do
  ratarmount -u "$d" || fusermount -u "$d"
done
```

### Prepare KITScenes evaluation data for val split
```
$ python prepare.py --out-dir <KITSCENES_VAL_EVAL_DATA>
```