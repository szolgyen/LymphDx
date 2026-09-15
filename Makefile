SHELL := /bin/bash

UV ?= uv
VENV_ROOT ?= .venvs

HF_ENV := $(VENV_ROOT)/heme-llm-hf
VLLM_ENV := $(VENV_ROOT)/heme-llm-vllm
SGLANG_ENV := $(VENV_ROOT)/heme-llm-sglang
OLLAMA_ENV := $(VENV_ROOT)/heme-llm-ollama

HF_PY := $(abspath $(HF_ENV))/bin/python

# Default CUDA 12.4 stack for HF runtime on this project.
HF_TORCH_VERSION ?= 2.6.0+cu124
HF_TORCHVISION_VERSION ?= 0.21.0+cu124
HF_TORCHAUDIO_VERSION ?= 2.6.0+cu124
HF_TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu124

.PHONY: help bootstrap python312 envs env-hf env-vllm env-sglang env-ollama lockfiles lock-hf lock-vllm lock-sglang lock-ollama test test-hf clean-envs

help:
	@echo "Available targets:"
	@echo "  make bootstrap        # lockfiles + all backend venvs"
	@echo "  make envs             # create all backend venvs from lock files"
	@echo "  make env-hf           # create HF backend venv"
	@echo "  make env-vllm         # create vLLM backend venv"
	@echo "  make env-sglang       # create SGLang backend venv"
	@echo "  make env-ollama       # create Ollama backend venv"
	@echo "  make lockfiles        # regenerate backend lock files"

bootstrap: python312 lockfiles envs
	@echo "Bootstrap complete."

python312:
	$(UV) python install 3.12

envs: env-hf env-vllm env-sglang env-ollama

env-hf: python312
	$(UV) venv $(HF_ENV) --python 3.12 --managed-python
	. $(HF_ENV)/bin/activate && $(UV) pip install -r requirements/hf/lock.txt
	. $(HF_ENV)/bin/activate && $(UV) pip install --reinstall --index-url $(HF_TORCH_INDEX_URL) \
		torch==$(HF_TORCH_VERSION) torchvision==$(HF_TORCHVISION_VERSION) torchaudio==$(HF_TORCHAUDIO_VERSION)
	. $(HF_ENV)/bin/activate && $(UV) pip install -e .

env-vllm: python312
	$(UV) venv $(VLLM_ENV) --python 3.12 --managed-python
	. $(VLLM_ENV)/bin/activate && $(UV) pip install -r requirements/vllm/lock.txt
	. $(VLLM_ENV)/bin/activate && $(UV) pip install -e .

env-sglang: python312
	$(UV) venv $(SGLANG_ENV) --python 3.12 --managed-python
	. $(SGLANG_ENV)/bin/activate && $(UV) pip install -r requirements/sglang/lock.txt
	. $(SGLANG_ENV)/bin/activate && $(UV) pip install -e .

env-ollama: python312
	$(UV) venv $(OLLAMA_ENV) --python 3.12 --managed-python
	. $(OLLAMA_ENV)/bin/activate && $(UV) pip install -r requirements/ollama/lock.txt
	. $(OLLAMA_ENV)/bin/activate && $(UV) pip install -e .

lockfiles: lock-hf lock-vllm lock-sglang lock-ollama

lock-hf:
	$(UV) pip compile --python 3.12 requirements/hf/requirements.txt -o requirements/hf/lock.txt

lock-vllm:
	$(UV) pip compile --python 3.12 requirements/vllm/requirements.txt -o requirements/vllm/lock.txt

lock-sglang:
	$(UV) pip compile --python 3.12 requirements/sglang/requirements.txt -o requirements/sglang/lock.txt

lock-ollama:
	$(UV) pip compile --python 3.12 requirements/ollama/requirements.txt -o requirements/ollama/lock.txt

clean-envs:
	rm -rf $(VENV_ROOT)