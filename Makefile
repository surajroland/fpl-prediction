.PHONY: help build up down logs shell shell-env test format lint jupyter tensorboard clean

help:
	@echo "FPL XGBoost Development Commands:"
	@echo "  build       - Build containers"
	@echo "  up          - Start services"
	@echo "  down        - Stop services"
	@echo "  shell       - Enter dev container"
	@echo "  shell-env   - Enter container with API keys"
	@echo "  test        - Run tests"
	@echo "  format      - Format code"
	@echo "  lint        - Run linting"
	@echo "  jupyter     - Start Jupyter Lab"
	@echo "  tensorboard - Start TensorBoard"
	@echo "  clean       - Clean project resources"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec fpl-dev bash

shell-env:
	docker-compose exec \
		-e GITHUB_TOKEN="$$GITHUB_TOKEN" \
		-e HF_TOKEN="$$HF_TOKEN" \
		-e WANDB_API_KEY="$$WANDB_API_KEY" \
		fpl-dev bash

test:
	docker-compose exec fpl-dev pytest tests/ -v --cov=src

format:
	docker-compose exec fpl-dev black src/ tests/
	docker-compose exec fpl-dev isort src/ tests/

lint:
	docker-compose exec fpl-dev flake8 src/ tests/
	docker-compose exec fpl-dev mypy src/

jupyter:
	docker-compose exec fpl-dev jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

tensorboard:
	docker-compose exec fpl-dev tensorboard --logdir=logs --host=0.0.0.0 --port=6006

clean:
	docker-compose down --rmi local
	docker-compose rm -f
