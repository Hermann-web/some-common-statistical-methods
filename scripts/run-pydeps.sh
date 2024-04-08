#!/bin/bash

# This script generates dependency graphs using [pydeps](https://github.com/thebjorn/pydeps) for Python files in specified folders.
# It creates both regular and minimized versions of the graphs for main and test files.

# Define paths
DOCS_FOLDER="./docs/pydeps"
MAIN_FOLDER=./statanalysis
TEST_FOLDER=./tests
TEMP_INIT=$TEST_FOLDER/__init__.py

# Log the start of the script
mkdir -p $DOCS_FOLDER
echo "Starting dependency graph generation..."

# Generate dependency graph for main files
echo "Generating dependency graph for main files..."
# pydeps --noshow $MAIN_FOLDER --cluster --rankdir LR -o $DOCS_FOLDER/pydeps.svg
echo "created $DOCS_FOLDER/pydeps.svg"
pydeps --noshow $MAIN_FOLDER --cluster --rankdir LR -o $DOCS_FOLDER/pydeps.cycles.svg --show-cycles
echo "created $DOCS_FOLDER/pydeps.cycles.svg"
pydeps --noshow $MAIN_FOLDER --cluster --rankdir LR -o $DOCS_FOLDER/pydeps.min.svg --max-module-depth 2
echo "created $DOCS_FOLDER/pydeps.min.svg"

# Generate dependency graph for test files
echo "Generating dependency graph for test files..."
# Ensure __init__.py exists temporarily in the test folder
touch $TEMP_INIT
pydeps --noshow $TEST_FOLDER --cluster --rankdir LR -o $DOCS_FOLDER/pydeps.test.svg
echo "created $DOCS_FOLDER/pydeps.test.svg"
pydeps --noshow $TEST_FOLDER --cluster --rankdir LR -o $DOCS_FOLDER/pydeps.test.min.svg --exclude numpy --max-module-depth 2
echo "created $DOCS_FOLDER/pydeps.test.min.svg"
# Clean up temporary __init__.py file
rm $TEMP_INIT

# Log the completion of the script
echo "Dependency graph generation completed."
