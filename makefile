.PHONY: venv # create virtual environment
venv:
	pip install pip --upgrade
	pip install pipreqs
	rm -rf requirements.txt requirements.in
	pipreqs . --mode no-pin --encoding utf-8 --ignore .venv
	mv requirements.txt requirements.in

	pip install pip-tools
	pip-compile requirements.in -o requirements.txt -vvv

	rm -rf .venv
	python3.11 -m venv .venv
	./.venv/bin/python3 -m pip install -r requirements.txt
	@echo "activate venv with: \033[1;33msource .venv/bin/activate\033[0m"

.PHONY: lock # freeze dependencies
lock:
	./.venv/bin/python3 -m pip freeze > requirements.in
	pip-compile requirements.in -o requirements.txt -vvv

.PHONY: docker # run or rebuild docker container
docker:
	@if docker compose ps --services --filter "status=running" | grep -q .; then \
		echo "rebuilding..."; \
		docker compose build; \
	else \
		echo "starting container..."; \
		docker compose up --detach; \
	fi

.PHONY: clean # wipe venv and all containers
clean:
	rm -rf ./.venv
	docker compose down --rmi all --volumes --remove-orphans
	docker system prune -a -f

.PHONY: fmt # format code
fmt:
	./.venv/bin/python3 -m pip install isort
	./.venv/bin/python3 -m pip install ruff
	./.venv/bin/python3 -m pip install autoflake
	./.venv/bin/python3 -m pip install mypy

	./.venv/bin/python3 -m isort .
	./.venv/bin/python3 -m autoflake --remove-all-unused-imports --recursive --in-place .
	./.venv/bin/python3 -m ruff format --config line-length=130 .
	-./.venv/bin/python3 -m mypy .

.PHONY: test # run all tests
test:
	./.venv/bin/python3 -m pip install pytest
	./.venv/bin/python3 -m pytest ./src/test_sync.py

.PHONY: help # generate help message
help:
	@echo "Usage: make [target]\n"
	@grep '^.PHONY: .* #' makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1	\2/' | expand -t20
