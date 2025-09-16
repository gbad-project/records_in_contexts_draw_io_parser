# Makefile for setting up Python project using Conda + Poetry + requirements.txt

# Addon for testing
SHELL := /bin/bash
OS := $(shell uname)

ESC    := $(shell printf '\033')
BLUE   := $(ESC)[0;34m
GREEN  := $(ESC)[0;32m
YELLOW := $(ESC)[0;33m
RED    := $(ESC)[0;31m
RESET  := $(ESC)[0m

# Lists minimal Python dependencies without versions
PIP_REQUIREMENTS_FILE = next/requirements.txt

# Makes this more reproducible by pinning the versions of Python and Java
# and adding Poetry for further package management and dependency locking
CONDA_ENV_FILE = next/environment.yml
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

test:
ifeq ($(OS),Darwin)
	@printf "%b>>> Running macOS tests...%b\n" "$(BLUE)" "$(RESET)"
	@bash tests/scripts/macos/test_rg_1-429.sh
else ifeq ($(OS),Linux)
	@printf "%b>>> Running Linux tests...%b\n" "$(BLUE)" "$(RESET)"
	@bash tests/scripts/linux/test_rg_1-429.sh
	@bash tests/scripts/linux/test_add.sh
	@bash tests/scripts/linux/test_auth.sh
	@bash tests/scripts/linux/test_run.sh
else
	@printf "%b>>> Unsupported OS: %s%b\n" "$(YELLOW)" "$(OS)" "$(RESET)"
endif
	@printf "%b>>> Running pytest...%b\n" "$(BLUE)" "$(RESET)"
	@conda run -n $(CONDA_ENV_NAME) python -m pytest tests
