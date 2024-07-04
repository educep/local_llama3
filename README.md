# local_llama3

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Fine-tune Llama 3 on a dataset of patient-doctor conversations.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for local_llama3
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── local_llama3                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes local_llama3 a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

``
pipreqs . --ignore "C:\path\to\anaconda3\lib\"
``
# Getting started
1. Create the environment
    ```sh
    make create_environment
    ```
2. Install packages. If you are on Windows, the following command will try to install CUDA. Comment out the corresponding line if you are not equipped with a GPU.
    ```sh
    make requirements
    ```

# Model
We'll be using HuggingFace's `meta-llama/Meta-Llama-3-8B`, for this
you need to ask for access.

# Generating medical dataset
We will work with the medical dataset [here](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot).

## Downloading from remote repository
If you have previously loaded the file, you can directly sync the local folder with S3.

```sh
make sync_data_down
```

## Downloading from HuggingFace and store in S3

```sh
make data
make sync_data_up
```

## Treating the dataset
To load and pre-process our dataset, we load the
`data/raw/medical` dataset, shuffle it, and select
only the top 1000 rows. And it will be stored in
`data/processed/medical`.

This will significantly reduce the training time.

This treatment is performed also in the `make data` command.
