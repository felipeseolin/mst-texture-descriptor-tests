# Validation and Tests - Minimum Spanning Trees texture descriptor

This repository contains the tests' source code of a new proposed 
texture descriptor for images, which was created as the
final paper of the student Felipe Seolin Bento to obtain
the degree of bachelor in Software Engineering.

## Results
The results are in the `results` folder. They are split in the
descriptors, and then in datasets.

## Important

The file `TestBase.py` represents the main class that is used
for all tests and validations, so that are some configuration 
methods, as `set_is_arff(Bool)`, `set_NAME_classifier()`, 
`set_NAME_descriptor()`, `set_NAME_dataset()`.

The file `classifier_constants.py`, `dataset_constants.py`, 
`descriptor_constants.py` defines the used classifiers, datasets, and
descriptors respectively.

## Requirements

These are the main requirements to run the repository.
Your operating system might impact other packages that you will
need to install, so consider vising the libraries that are described
at the Pipfile, to check if your OS needs some configuration.

- [Python 3.8](https://www.python.org)
- [Pip](https://www.python.org)
- [Pipenv](https://pipenv.pypa.io)

## Steps to run

### Build pipenv
First, you need to run pipenv, so open up a terminal window
and run at the project root

```bash
pipenv install
```

To activate the environment run:

```bash
pipenv shell
```

### Data
All the data should be in a `.arff` or `.csv` extension, 
and the name of the file should be the same as the descriptor name.
Also, it needs to be located at the `datasets` folder, inside the
`name-of-the-dataset` folder.

### Run

Change the `base_path_results` value in `TestBase.py`.

Choose one `main.py` file localed inside the 
`descriptor-name > dataset-name > supervised-learning-algorithm`.
Then run, inside the folder:

```bash
pipenv run python ./main.py
```

## Other repositories related

- [Paper](https://github.com/felipeseolin/TCC)
- [Descriptor code](https://github.com/felipeseolin/mst-texture-descriptor)
