.PHONY: install lint format type test cov check run docker-build docker-test docker-run

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
	uvicorn api.main:create_app --factory --reload --port 8000

docker-build:
	docker build --target runtime -t ai-chat-service:local .

docker-test:
	docker build --target test .

docker-run:
	docker compose up --build
