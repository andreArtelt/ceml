# Developing stuff
## Run tests
`python -m pytest --cov=ceml --cov-report=html tests/`

## Build documentation
`make clean && make html`

## Build PyPI package
`python setup_pip.py bdist_wheel`