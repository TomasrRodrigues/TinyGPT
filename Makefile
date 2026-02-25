install:
	pip install -r requirements.txt

train:
	python experiments/run_experiment.py --config configs/base.yaml

lint:
	black .
	flake8 .

test:
	pytest