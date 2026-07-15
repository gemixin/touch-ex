# Touch-Ex

## Overview

**Touch-Ex** is a set of scripts for building and evaluating models using the Touch-Ex dataset, a visuo-tactile dataset collected using a [DIGIT](https://digit.ml/) sensor. It includes the following directories:

- `configs`: Config and cache files.
- `data`: Dataset modules and classes plus the baseline background subtraction image.
- `models`: Model, training and evaluation modules and classes.
- `notebooks`: Exploration notebooks.
- `results`: Model results including plots.
- `saved_models`: Saved PyTorch models.
- `scripts`: Various experiment implementations.


## Requirements

- **Operating System:** Windows, Mac, Linux
- **Tested Environment:** Ubuntu 22.04, Python 3.12.13
- **Python Environment:** Regular Python or Anaconda environment
- **Packages:** See `requirements.txt`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/gemixin/touch-ex.git
cd touch-ex
```

### 2. Install dependencies

#### Option A: With pip

1. **(Optional) Set up a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. **Install the required package:**  
    ```bash
    python3 -m pip install -r requirements.txt
    ```

#### Option B: With Anaconda

Create a new conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate touch-ex
```

## Citation

If you use DIGIT or this repo in your research, please cite:

**DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor with Application to In-Hand Manipulation**  
Mike Lambeta, Po-Wei Chou, Stephen Tian, Brian Yang, Benjamin Maloon, Victoria Rose Most, Dave Stroud, Raymond Santos, Ahmad Byagowi, Gregg Kammerer, Dinesh Jayaraman, Roberto Calandra  
_IEEE Robotics and Automation Letters (RA-L), vol. 5, no. 3, pp. 3838–3845, 2020_  
[https://doi.org/10.1109/LRA.2020.2977257](https://doi.org/10.1109/LRA.2020.2977257)