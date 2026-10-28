PY := uv run python -m soundwatch
export SDL_VIDEODRIVER ?=

VAL_SCEN := artifacts/data/scenarios_val

.PHONY: install test lint doctor data data-quick train quantize scenarios-val tune eval demo demo-file \
	demo-headless demo-gif demo-ui record all

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

# WP8: val-split scenarios (tuning only; the test scenarios stay untouched by tuning)
scenarios-val:
	$(PY) make-scenarios --split val --out $(VAL_SCEN) --no-demo --n-events 30 --n-background 20

# grid-search post-processing on val -> artifacts/reports/tuning.md (copy the choice into the config)
tune: scenarios-val
	$(PY) tune --scenarios $(VAL_SCEN)

eval:
	$(PY) make-scenarios
	$(PY) eval
	$(PY) eval-streaming
	$(PY) fa-analysis
	$(PY) benchmark
	SDL_VIDEODRIVER=dummy $(PY) report --measure-ui-fps

demo:
	$(PY) run --source mic

demo-file:
	$(PY) run --source assets/demo/demo_scenario.wav

demo-headless:
	$(PY) run --source assets/demo/demo_scenario.wav --headless --fast --no-log

# record the simulator window while playing the demo scenario (headless OK) -> GIF (~7 MB)
demo-gif:
	SDL_VIDEODRIVER=dummy $(PY) run --source assets/demo/demo_scenario.wav --max-seconds 70 --no-log \
		--record-gif artifacts/reports/demo.gif --ui-stats artifacts/reports/ui_stats.json

# usage: make record CLASS=door_knock SECONDS=5 COUNT=3
record:
	$(PY) record --class $(CLASS) --seconds $(or $(SECONDS),5) --count $(or $(COUNT),1)

demo-ui:
	uv run python -m soundwatch.ui.app --demo

all: data train quantize eval
