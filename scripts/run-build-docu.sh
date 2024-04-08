#!/bin/bash

sphinx-apidoc ./statanalysis/ -o ./docs/source/statanalysis/ -f -E

cd docs
make html
cd ..
open docs/build/html/index.html
