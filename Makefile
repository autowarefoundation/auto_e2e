.PHONY: setup test benchmark e2e_test

setup:
	pip install torch timm pytest

test:
	cd Model/tests && python -m pytest test_auto_e2e.py -v

benchmark:
	cd Model/speed_benchmark && python speed_benchmark.py

e2e_test:
	@if [ -n "$$HF_TOKEN" ] && [ ! -d data/nvidia_av/camera ]; then \
		echo "Downloading NVIDIA PhysicalAI dataset (1 clip)..."; \
		cd Model/data_parsing/nvidia_physical_ai && \
		python download_dataset.py --out ../../../data/nvidia_av --clips 1; \
	fi
	cd Model/tests && NVIDIA_AV_ROOT=../../data/nvidia_av \
		python -m pytest e2e_test.py -v -m e2e_data -s
