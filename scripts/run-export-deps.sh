
poetry export -f requirements.txt --output ./docs/requirements.txt --without-hashes --with visu --with pandas --with sklearn --with buildthedocs
poetry export -f requirements.txt --output ./requirements.txt --without-hashes
