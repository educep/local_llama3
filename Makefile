#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = local_llama3
PYTHON_VERSION = 3.9
PYTHON_INTERPRETER = python
VENV = .venv
AWS_PROFILE = eb-cli-ank
AWS_BUCKET = ank-public-bkt-1

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	$(info PROJECT_NAME=$(PROJECT_NAME))
	$(info PYTHON_INTERPRETER=$(PYTHON_INTERPRETER))
	$(info VENV=$(VENV))
ifeq ($(OS),Windows_NT)
	$(info Running on Windows_NT)
	@if not exist "$(VENV)\Scripts\activate" ( \
		cmd /c "mkvirtualenv.bat $(VENV) $(PYTHON_INTERPRETER)" \
	) else ( \
		echo Virtual environment $(VENV) already exists. \
	)
else
	$(info Running on Unix-like OS)
	@if [ ! -d "$(VENV)/bin" ]; then \
		$(PYTHON_INTERPRETER) -m venv $(VENV); \
	else \
		echo Virtual environment $(VENV) already exists.; \
	fi
endif
	@echo ">>> New virtualenv created. Activate with:\nsource $(VENV)/bin/activate (Unix-like systems) or .\\$(VENV)\\Scripts\\activate (Windows)"



## Install Python Dependencies
.PHONY: requirements
requirements:
ifeq ($(OS),Windows_NT)
	@cmd /c "call .\\$(VENV)\\Scripts\\activate && $(PYTHON_INTERPRETER) -m pip install -U pip"
	$(info Trying to install pytorch with cuda on Windows_NT)
	@cmd /c "call .\\$(VENV)\\Scripts\\activate && $(PYTHON_INTERPRETER) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
	@cmd /c "call .\\$(VENV)\\Scripts\\activate && $(PYTHON_INTERPRETER) -m pip install -r requirements.txt"
else
	@bash -c "source $(VENV)/bin/activate && $(PYTHON_INTERPRETER) -m pip install -U pip && $(PYTHON_INTERPRETER) -m pip install -r requirements.txt"
endif


## Delete all compiled Python files
.PHONY: clean
clean:
ifeq ($(OS),Windows_NT)
	@for /r %%p in (*.pyc *.pyo) do if not "%%~dp$VENV\.."==".\\$(VENV)\\" del "%%p"
	@for /d %%p in (.) do if "%%p" neq ".\\$(VENV)" if exist "%%p\\__pycache__" rmdir /S /Q "%%p\\__pycache__"
else
	@find . -type f -name "*.py[co]" -not -path "./$(VENV)/*" -delete
	@find . -type d -name "__pycache__" -not -path "./$(VENV)/*" -exec rm -r {} +
endif


## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 $(PROJECT_NAME)
	isort --check --diff --profile black $(PROJECT_NAME)
	black --check --config pyproject.toml $(PROJECT_NAME)


## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml $(PROJECT_NAME)


## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	aws s3 sync s3://$(AWS_BUCKET)/llama3/ data/ --profile $(AWS_PROFILE)


## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	aws s3 sync data/ s3://$(AWS_BUCKET)/llama3/ --profile $(AWS_PROFILE)



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
.PHONY: fb_data
fb_data: # requirements
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data/fb_make_dataset.py


## Make Dataset
.PHONY: data
data: # requirements
	$(PYTHON_INTERPRETER) $(PROJECT_NAME)/data/make_dataset.py


## Update Project Structure
.PHONY: project_structure
project_structure:
	$(PYTHON_INTERPRETER) tools/project_tree.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
