#################################################################################
# GLOBALS                                                                       #
#################################################################################

THIS_FILE := $(lastword $(MAKEFILE_LIST))
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3
CACHE_DIR := /var/tmp/

export IR_DATASETS_HOME := $(CACHE_DIR)/.ir-datasets
export HF_DATASETS_CACHE := $(CACHE_DIR)/.cache
export TOKENIZERS_PARALLELISM := true

#################################################################################
# GENERAL COMMANDS                                                              #
#################################################################################

.PHONY: requirements
## Install Python Dependencies
requirements:
	pip install -r requirements.txt

.PHONY: clean
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# DATA COMMANDS                                                                 #
#################################################################################
data/processed/trec-dl-2019.parquet:
	python3 scripts/eval-data.py \
		--dataset msmarco-passage/trec-dl-2019/judged \
		--output data/processed/trec-dl-2019.parquet

data/processed/trec-dl-2020.parquet:
	python3 scripts/eval-data.py \
		--dataset msmarco-passage/trec-dl-2020/judged \
		--output data/processed/trec-dl-2020.parquet

data/processed/msmarco-train.parquet:
	python3 scripts/train-data.py \
		--dataset msmarco-passage/train \
		--output data/processed/msmarco-train.parquet \
		--batch_size 128

.PHONY: data-test
## Download and tokenize eval dataset
data-eval: data/processed/trec-dl-2019.parquet data/processed/trec-dl-2020.parquet

.PHONY: data-train
## Download and tokenize train dataset
data-train: data/processed/msmarco-train.parquet

#################################################################################
# PROJECT COMMANDS                                                              #
#################################################################################
.PHONY: fit
## Run the training process
fit: data-train data-test
	python3 main.py fit -c configs/run-config.dev.yaml

.PHONY: eval
## Run eval process
eval: data-eval
	python3 main.py test -c configs/run-config.dev.yaml

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
