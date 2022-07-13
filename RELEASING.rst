To be able to do a release, you will need an environment with the `twine` and `build` packages installed:

```
python -m pip install build twine
```

In that environment, and having the credentials for the PyPI user, you can then do a test release by following these steps:

0. Clean the dist directory, if present: `rm dist/*`
1. Edit the version in `version.py`, and commit it.
2. Build the package with `python -m build`
3. Upload the package to the test PyPI instance with `twine upload dist/* --repository-url https://test.pypi.org/legacy/`

You should then test if everything went right, by installing that package in a new virtual environment using pip (pointing to that test index).

If everything went right, you can then do the actual release to the real PyPI index:

```
twine upload dist/*
```
