# When the Sun Becomes an Attacker: Glare-Induced Adversarial Vulnerabilities in Traffic Sign Classification

# Prerequisites
## This code was primarily run on Ubuntu 22.04.5. However, nothing is depending on specific version of Ubuntu. You are required to have [virtual environment](https://docs.python.org/3/library/venv.html) in your system.

### How to installed virtual environment in your system?

``` bash
python -m venv .venv          # create
source .venv/bin/activate     # activate (Linux/macOS)
.venv\Scripts\activate        # activate (Windows)
pip install -r requirements.txt

```



## Repository Structure

This repository is containes several folders, each serving a specific purpose in our study. Below is a table detailing each folder and it's contents.

| Folder Name              | Description                                                                                                  | README Link                                   |
|--------------------------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `1_Datasets`             | Contains three datasets, each with images, human-annotated JSON files, and model-generated JSON files.       | [README](./1_Datasets/README.md)              |
| `2_GenerateDescriptions` | Code for converting `txt` files to `json`; also contains the main generation code.                           | [README](./2_GenerateDescriptions/README.md)  |
| `3_GenerateResults`      | Code for analyzing study results and data presentation.                                                      | [README](./3_GenerateResults/README.md)       |
