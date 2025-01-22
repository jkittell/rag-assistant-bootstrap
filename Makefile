.PHONY: setup dev deploy clean test

# Development
setup:
	cp .env.example .env
	pip install -r requirements.txt

dev:
	devspace dev

# Deployment
deploy:
	devspace deploy

deploy-prod:
	helm install langchain-app ./chart

# Testing and Linting
test:
	python -m pytest

lint:
	flake8 .
	black --check .

format:
	black .

# Cleanup
clean:
	devspace purge
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
