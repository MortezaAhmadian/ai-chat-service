.PHONY: install lint format type test cov check run run-vllm vllm-check docker-build docker-test docker-run

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt && pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

type:
	mypy

test:
	pytest -q

cov:
	pytest --cov --cov-report=term-missing

check: lint type cov

run:
	uvicorn api.main:create_app --factory --reload --port 8080

# API backed by the vLLM server (start vLLM first; see README "Local GPU with vLLM")
run-vllm:
	APP_LLM_PROVIDER=vllm uvicorn api.main:create_app --factory --reload --port 8080

# Is vLLM up, and which model id does it serve?
vllm-check:
	curl -s $${APP_VLLM_BASE_URL:-http://localhost:8000/v1}/models | python -m json.tool

docker-build:
	docker build --target runtime -t ai-chat-service:local .

docker-test:
	docker build --target test .

docker-run:
	docker compose up --build
