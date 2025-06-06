PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT_NAME := sceneflow

SRC_DIR := ./$(PROJECT_NAME)
TEST_DIR := ./tests
DEMOS_DIR := ./demos

.PHONY: install dev clean lint format check

NUM_JOBS := 8

install:
	MAX_JOBS=$(NUM_JOBS) $(PIP) install .

dev:
	MAX_JOBS=$(NUM_JOBS) $(PIP) install -e .[dev,docs,test,extra]

torch:
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --upgrade


requirements:
	$(PIP) install -r requirements.txt

requirements-upgrade:
	$(PIP) install -r requirements.txt --upgrade

clean:
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	$(PIP) uninstall -y $(PROJECT_NAME)

lint:
	ruff format $(SRC_DIR) 

format:
	ruff format $(SRC_DIR)  

sort:
	ruff check $(SRC_DIR)  --fix

test:
	pytest -v tests