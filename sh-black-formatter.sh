#!/bin/bash 
set -e 
#black --config ./pyproject.toml --no-cache --verbose .
black --config ./pyproject.toml --no-cache .
