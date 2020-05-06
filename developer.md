# Developing stuff
## Run tests
`python -m pytest --cov=ceml --cov-report=html tests/`

## Build documentation
`make clean && make html`

## Build PyPI package
`python setup_pip.py bdist_wheel`

Upload to PyPI:
`twine upload dist/ceml-VERSION-py3-none-any.whl`

Be aware of possible issues with *keyring* and *keyrings.alt*