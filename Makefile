PY := uv run python -m soundwatch
export SDL_VIDEODRIVER ?=

.PHONY: install test lint doctor data data-quick train quantize eval demo demo-file demo-ui all

install:
	uv sync

test:
	SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run pytest

lint:
	uv run ruff check src tests

doctor:
	$(PY) doctor

data:
	$(PY) download --with-fsd50k
	$(PY) build-dataset --with-fsd50k

data-quick:
	$(PY) download --quick
	$(PY) build-dataset --quick

train:
	$(PY) train-teacher
	$(PY) train-student --distill
	$(PY) train-student --no-distill --tag nodistill

quantize:
	$(PY) quantize
	$(PY) export
	$(PY) export-icons
	$(PY) export-golden

eval:
	$(PY) make-scenarios
	$(PY) eval
	$(PY) eval-streaming
	$(PY) benchmark
	$(PY) report

demo:
	$(PY) run --source mic

demo-file:
	$(PY) run --source assets/demo/demo_scenario.wav

demo-ui:
	uv run python -m soundwatch.ui.app --demo

all: data train quantize eval
