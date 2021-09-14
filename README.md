# calorimetry-likelihood
Calorimetry based particle identiification tools for LArTPC

## Conda environment
Running calorimetry-likelihood requires well defined python package.
You can create the necessary conda environment by:
```conda env create -f environment.yml```
and activate it at any session with:
```conda activate calorimetry_likelihood```

## Install the Python package
This will make the `calorimetry_likelihood` package accessible with `import calorimetry_likelihood`
```python setup.py bdist_wheel```
```pip install dist/calorimetry_likelihood-0.1.0-py3-none-any.whl```

## Setting up the python path
First go in the directory of the repository and run `. setup.sh`
This will set the correct `PYTHONPATH` so that you can load the libraries in `lib` in the notebooks.

