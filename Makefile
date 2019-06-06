.PHONY: clean v3_results v3_test_results v3_train_results csv_export manual_measurments data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = burke_leaf_veins
PYTHON_INTERPRETER = pipenv run python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
setup_environment:
	pipenv install

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

reports/data/:
	mkdir -p reports/data/

## Make Datasets
clean_targets:
	$(PYTHON_INTERPRETER) src/data/clean_targets.py

dataset:
	$(PYTHON_INTERPRETER) src/data/generate_dataset.py

clean_data:
	rm -rf data/interim/veins
	rm -rf data/processed/veins

###
# Use the v3 trained model to segment images using wegiths from 800 epochs
###
models/v3/results/dilations/test/:
	mkdir -p models/v3/results/dilations/800/test/
	$(PYTHON_INTERPRETER) src/eval/segment_images.py \
	                       models/v3/dilations	\
	                       models/v3/dilations/saves/noAug_dilations_epoch800 \
	                       models/v3/results/dilations/800/test/ \
	                       data/processed/veins/test/real/*.j*
models/v3/results/dilations/train/:
	mkdir -p models/v3/results/dilations/800/train/
	$(PYTHON_INTERPRETER) src/eval/segment_images.py \
 	                       models/v3/dilations	\
 	                       models/v3/dilations/saves/noAug_dilations_epoch800 \
 	                       models/v3/results/dilations/800/train/ \
 	                       data/processed/veins/train/real/*.j*
# Use the v3 trained model to segment the images from other species
models/v3/results/dilations/800/generalization/:
	mkdir -p models/v3/results/800/generalization/
	$(PYTHON_INTERPRETER) src/eval/segment_images.py \
	                       models/v3/dilations	\
	                       models/v3/dilations/saves/noAug_dilations_epoch800 \
	                       models/v3/results/dilations/800/generalization \
	                       data/raw/Veins\ machine\ learning\ with\ Chris/Assorted\ species\ veins/*.jpg
# Extract vein length from the v3 segmented images
v3_test_results: models/v3/results/dilations/800/test/ reports/data/
	$(PYTHON_INTERPRETER) src/eval/extract_vein_length.py \
													sqlite:///reports/data/results.sqlite \
													v3_test_vein_length \
													models/v3/results/dilations/800/test/*_probs.png
v3_train_results:  models/v3/results/dilations/800/train/ reports/data/
	$(PYTHON_INTERPRETER) src/eval/extract_vein_length.py \
													sqlite:///reports/data/results.sqlite \
													v3_train_vein_length \
													models/v3/results/dilations/800/train/*_probs.png
v3_results: v3_test_results v3_train_results

###############################################################################
# Use the v3 trained model to segment test images using weights from 1500 epochs
###
models/v3/results/dilations/1500/test/:
	mkdir -p models/v3/results/dilations/1500/test/
	$(PYTHON_INTERPRETER) src/eval/segment_images.py \
	                       models/v3/dilations	\
	                       models/v3/dilations/saves/noAug_dilations_epoch1500 \
	                       models/v3/results/dilations/1500/test/ \
	                       data/processed/veins/test/real/*.j*
models/v3/results/dilations/1500/train/:
	mkdir -p models/v3/results/dilations/1500/train/
	$(PYTHON_INTERPRETER) src/eval/segment_images.py \
	                       models/v3/dilations	\
	                       models/v3/dilations/saves/noAug_dilations_epoch1500 \
	                       models/v3/results/dilations/1500/train/ \
	                       data/processed/veins/train/real/*.j*

v3_1500_test_results: models/v3/results/dilations/1500/test/ reports/data/
	$(PYTHON_INTERPRETER) src/eval/extract_vein_length.py \
 													sqlite:///reports/data/results.sqlite \
 													v3_1500_test_vein_length \
 													models/v3/results/dilations/1500/test/*_probs.png

################################################################################
# V3 Model trained with trianing images randomally resized
# -75% to +25%
###
models/v3/results/dilations/Scaled_075_125/test/:
	mkdir -p models/v3/results/dilations/Scaled_075_125/test/
	$(PYTHON_INTERPRETER) src/eval/segment_images.py \
 	                       models/v3/dilations	\
 	                       models/v3/dilations/saves/V2_Scaled_075_125_epoch2200 \
 	                       models/v3/results/dilations/Scaled_075_125/test/ \
 	                       data/processed/veins/test/real/*.j*
# Use the v3 trained model to segment the images from other species
models/v3/results/dilations/Scaled_075_125/generalization/:
	mkdir -p models/v3/results/dilations/Scaled_075_125/generalization/
	$(PYTHON_INTERPRETER) src/eval/segment_images.py \
	                       models/v3/dilations	\
	                       models/v3/dilations/saves/V2_Scaled_075_125_epoch2200 \
	                       models/v3/results/dilations/Scaled_075_125/generalization/ \
	                       data/raw/Veins\ machine\ learning\ with\ Chris/Assorted\ species\ veins/*.jpg


#################################################################################
# V3 Model trained with trianing images randomally resized
# -20% to +05%
###
models/v3/results/dilations/Scaled_020_105/test/:
	mkdir -p models/v3/results/dilations/Scaled_020_105/test/
	$(PYTHON_INTERPRETER) src/eval/segment_images.py \
  	                       models/v3/dilations	\
  	                       models/v3/dilations/saves/V2_Scaled_020_105_epoch7100 \
  	                       models/v3/results/dilations/Scaled_020_105/test/ \
  	                       data/processed/veins/test/real/*.j*
# Use the v3 trained model to segment the images from other species
models/v3/results/dilations/Scaled_020_105/generalization/:
	mkdir -p models/v3/results/dilations/Scaled_020_105/generalization/
	$(PYTHON_INTERPRETER) src/eval/segment_images.py \
	                       models/v3/dilations	\
	                       models/v3/dilations/saves/V2_Scaled_020_105_epoch7100 \
	                       models/v3/results/dilations/Scaled_020_105/generalization/ \
	                       data/raw/Veins\ machine\ learning\ with\ Chris/Assorted\ species\ veins/*.jpg

#################################################################################
# Add the manual measruments to the results sqlite databse.
####
manual_measurments: reports/data/
	$(PYTHON_INTERPRETER) src/eval/manual_measurments_to_db.py \
													sqlite:///reports/data/results.sqlite \
													manually_measured \
													data/raw/VeinLengths.csv

################################################################################
# Reports exported to csv
#####
# Export vein measurment data as csv files
reports/csv/:
	mkdir -p reports/csv/

reports/csv/v3_train_vein_length.csv: reports/csv/
	$(PYTHON_INTERPRETER) src/eval/db_to_csv.py \
													sqlite:///reports/data/results.sqlite \
													v3_train_vein_length \
													'csv' \
													reports/csv/v3_train_vein_length.csv

reports/csv/v3_test_vein_length.csv: reports/csv/
	$(PYTHON_INTERPRETER) src/eval/db_to_csv.py \
													sqlite:///reports/data/results.sqlite \
													v3_test_vein_length \
													'csv' \
													reports/csv/v3_test_vein_length.csv

reports/csv/manually_measured_vein_length.csv:
	$(PYTHON_INTERPRETER) src/eval/db_to_csv.py \
													sqlite:///reports/data/results.sqlite \
													manually_measured \
													'csv' \
													reports/csv/manually_measured_vein_length.csv

csv_export: reports/csv/v3_train_vein_length.csv reports/csv/v3_test_vein_length.csv reports/csv/manually_measured_vein_length.csv

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
