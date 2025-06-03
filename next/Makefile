# Makefile for setting up Python project using Conda + Poetry + requirements.txt

# Lists minimal Python dependencies without versions
PIP_REQUIREMENTS_FILE = ./requirements.txt

# Makes this more reproducible by pinning the versions of Python and Java
# and adding Poetry for further package management and dependency locking
CONDA_ENV_FILE = ./environment.yml
CONDA_ENV_NAME = gbad-next   # Will be used to create Conda environment

# Poetry takes its config from ./pyproject.toml

.PHONY: install conda poetry uninstall  # these are interpreted as make commands

install: conda poetry  # this defines the install command

# Specific definitions follow

conda:
	@conda env create -f $(CONDA_ENV_FILE) -n $(CONDA_ENV_NAME)

poetry:
	@conda run -n $(CONDA_ENV_NAME) poetry config virtualenvs.create false
	@conda run -n $(CONDA_ENV_NAME) poetry add $(shell cat $(PIP_REQUIREMENTS_FILE))
	@conda run -n $(CONDA_ENV_NAME) poetry install

uninstall:  # to uninstall, just remove the entire conda env
	@conda env remove -n $(CONDA_ENV_NAME)
