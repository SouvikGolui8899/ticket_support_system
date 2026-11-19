up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build

bash:
	docker compose exec app bash

test:
	docker compose exec app pytest --maxfail=1

test-cov:
	docker compose exec app python -m pytest --cov=src

eval:
	docker compose exec app python -m src.scripts.evaluation.harness --input data/validation_tickets.json --output evaluation/results/; docker cp app-container:/app/evaluation .

dev-eval:
	docker compose exec app python -m src.scripts.evaluation.harness --input data/development_tickets.json --output evaluation/results/; docker cp app-container:/app/evaluation .

# Pass args="./app/data/test_tickets.json" to report the test metrics
test-eval:
	docker cp $(args) app-container:/app/data; docker compose exec app python -m src.scripts.evaluation.harness --input data/$(shell basename ${args}) --output evaluation/results/; docker cp app-container:/app/evaluation .
