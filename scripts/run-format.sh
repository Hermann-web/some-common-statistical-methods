#!/bin/bash

poetry run isort .
poetry run autopep8 . -r -i
