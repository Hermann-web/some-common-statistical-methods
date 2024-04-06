
poetry export -f requirements.txt --output ./docs/requirements.txt --without-hashes --only visu --only pandas --only sklearn
poetry export -f requirements.txt --output ./requirements.txt --without-hashes

