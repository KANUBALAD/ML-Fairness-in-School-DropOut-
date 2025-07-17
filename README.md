<<<<<<< HEAD
# ML-Fairness-in-School-DropOut

This project aims to address the issue of fairness in machine learning models for predicting school dropout rates. The goal is to develop and evaluate models that are not biased towards any particular group, ensuring equal opportunities for all students.

```
python main.py ./yaml/brazil.yaml
```

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The aim of this project is to utilize various machine learning models to predict the school dropout rate. Additionally, we evaluate the fairness of these models in reference with different definitions of fairness.

## Installation
To facilitate an easy setup, we have provided a .env file. You can use this file to install all the necessary packages required for this project by executing a single command.

## Data
This dataset originates from Kaggle, titled "Predict students' dropout and academic success: Investigating the Impact of Social and Economic Factors." The dataset is comprehensively described, including its source, format in the Kaggle website. For public datasets, you can [downlaod it here](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data).

## Model Training
We trained 3 different ML models:
 - Logistic Regression
 - Random Forest
 - Decision Trees

## Evaluation
Our evaluation metrics include:

- **Accuracy Parity:** The accuracy rate is a measure of how many instances were correctly classified by the model, where a higher accuracy rate generally indicates a better-performing model.

- **False Positive Rate:**  Calculates the false positive rate (FPR) for a binary classification task. The FPR is a measure of how many negative instances (i.e., instances that should have been classified as 0) were incorrectly classified as positive (1).


- **Demographic Parity:** Demographic parity is a fairness metric that measures whether the positive prediction rates are equal across different groups. In other words, it checks if individuals from different groups have an equal chance of being predicted as positive, regardless of their true labels.

- **Equality of Opportunity:** Equality of opportunity is a fairness metric that measures whether the true positive rates are equal across different groups. In other words, it checks if individuals who should have been classified as positive (i.e., true label is 1) have an equal chance of being correctly classified as positive, regardless of their group membership.



## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

Copyright (c) 2024 School dropout rate

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Contact
If you have any questions or suggestions, feel free to contact [Deborah D Kanubala](mailto:dkanubala@aimsammi.org) or [Alidu Abubakari](mailto:alidu.abubakari@acity.edu.gh).

=======
# Fairness in AI for Education: A Modular Framework for Mitigating Bias in Dropout Prediction

This project provides a comprehensive and modular framework for studying and mitigating algorithmic bias in student dropout prediction models. It uses real-world datasets from different continents to explore the effectiveness of fairness interventions, with a special focus on how dataset imbalance impacts their performance.

The framework includes a complete, end-to-end experimental pipeline, from model selection and baseline fairness analysis to advanced studies on the impact of data augmentation using various synthetic data generation techniques (Faker, LLM, GANs).

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [How to Use the Framework](#how-to-use-the-framework)
  - [Understanding Configuration](#understanding-configuration)
  - [Running a Standard ML Experiment](#running-a-standard-ml-experiment)
  - [Running a Balanced Augmentation Experiment](#running-a-balanced-augmentation-experiment)
- [The Reproducible Experimental Pipeline](#the-reproducible-experimental-pipeline)
  - [Experiment 1: Model Selection](#experiment-1-model-selection)
  - [Experiment 2: Baseline Fairness Assessment](#experiment-2-baseline-fairness-assessment)
  - [Experiment 3: Imbalance Impact Study](#experiment-3-imbalance-impact-study)
  - [Experiment 4: Method Selection Criteria Study](#experiment-4-method-selection-criteria-study)
- [Source Code Modules (`src/`) Explained](#source-code-modules-src-explained)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Key Features

*   **Modular Architecture**: Cleanly separated modules for data loading, preprocessing, modeling, and fairness evaluation.
*   **Multi-Dataset Support**: Comes pre-configured for three real-world student dropout datasets from Brazil, Africa, and India.
*   **Standard ML Models**: Includes implementations for Logistic Regression, Decision Trees, and Random Forests.
*   **Fairness-Aware Training**: Implements pre-processing mitigation techniques like Reweighting to reduce model bias.
*   **Comprehensive Fairness Metrics**: Evaluates models using a suite of metrics including Demographic Parity, Equalized Odds, and Accuracy Difference.
*   **Advanced Data Augmentation**: Features a powerful system for generating targeted synthetic data to rebalance datasets according to specific fairness goals.
*   **Multiple Generation Methods**:
    *   **Faker**: Fast, rule-based generation.
    *   **LLM-based**: Sophisticated, context-aware generation using Large Language Models (e.g., DeepSeek).
    *   **GAN-based**: Support for CTGAN and TVAE models via the `sdv` library (optional).
*   **Reproducible Research Pipeline**: A set of four shell scripts that fully automate the experimental workflow, from initial model selection to a final, evidence-based comparison of data generation methods.

## Project Structure

The project is organized in a modular and intuitive structure to facilitate research and development.

```
ModularCode/
├── data/
│   ├── asia_dropout.csv            # India dataset
│   ├── brazil_data.csv             # Brazil dataset
│   └── Secondary_school_dropout_dataset.csv # Africa dataset
├── experiments/
│   ├── 01_model_selection.sh       # Finds the best model for each dataset
│   ├── 02_baseline_fairness.sh     # Establishes baseline fairness metrics
│   ├── 03_imbalance_impact.sh      # Studies how imbalance affects fairness
│   ├── 04_method_comparison.sh     # Compares Faker vs. LLM generation methods
│   └── utils/                      # Helper scripts for experiments
├── results/                        # Directory for saving all experiment outputs
├── src/
│   ├── dataload.py                 # Loads and performs initial data transformation
│   ├── fairness.py                 # Core fairness metrics and mitigation logic
│   ├── llm_synthetic_generator.py  # LLM-based synthetic data generator
│   ├── model.py                    # ML model training and cross-validation
│   ├── preprocess.py               # Feature preprocessing pipeline (scaling, encoding)
│   ├── synthetic_generator.py      # Faker-based synthetic data generator
│   └── utils.py                    # Helper functions (data splitting, metrics)
├── yaml/
│   ├── africa.yaml                 # Configuration for the Africa dataset
│   ├── brazil.yaml                 # Configuration for the Brazil dataset
│   └── india.yaml                  # Configuration for the India dataset
├── main.py                         # Main entry point for running all experiments
├── requirements.txt                # Python package dependencies
└── README.md                       # This file
```

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   `pip` package manager
*   `bash` shell (for running the experiment scripts)
*   (Optional) A `yq` installation for easier YAML manipulation in the shell scripts. A `sed` fallback is included.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ModularCode
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *The `requirements.txt` should contain:*
    ```
    pandas
    numpy
    scikit-learn
    pyyaml
    faker
    chardet
    aiohttp
    sdv 
    ```

4.  **Set up your API Key (for LLM generation):**
    To use the `llm_async` generation method, you need an API key from a provider like DeepSeek. You can pass this key directly via the command line.
    For example, in `experiments/03_imbalance_impact.sh` and `04_method_comparison.sh`, set the `API_KEY` variable:
    ```bash
    API_KEY="your_deepseek_api_key_here"
    ```

## How to Use the Framework

The primary entry point is `main.py`, which is controlled by a YAML configuration file and command-line arguments.

### Understanding Configuration

Each dataset has a corresponding `.yaml` file in the `yaml/` directory. These files control the basic experiment setup:

```yaml
# ModularCode/yaml/brazil.yaml
dataname: brazil
datapath: "./data/brazil_data.csv"
test_size: 0.2
random_state: 42

unawareness: True       # Drop the gender column from features
model: compare          # "compare", "logistic_regression", etc.

# --- FAIRNESS EXTENSIONS ---
fairness: true          # Enable fairness-aware training
fair_technique: reweighting # Technique to use
```

### Running a Standard ML Experiment

This runs the standard pipeline: load data, preprocess, train model(s), and evaluate for performance and fairness.

**Command:**
```bash
python main.py <path_to_yaml_config> [options]
```

**Examples:**

1.  **Run model comparison for the Brazil dataset:**
    *(This runs cross-validation for all models specified in `src/model.py`)*
    ```bash
    python main.py yaml/brazil.yaml
    ```

2.  **Run a single model (Random Forest) with fairness mitigation:**
    *(The `--fairness` flag overrides the YAML setting if you want to force it on)*
    ```bash
    python main.py yaml/brazil.yaml --fairness
    ```
    *(Note: The `brazil.yaml` already has `model: compare` and `fairness: true`, so the script will run a full comparison with fairness interventions by default.)*

3.  **Save results to a dedicated folder:**
    ```bash
    python main.py yaml/africa.yaml --save_results
    ```

### Running a Balanced Augmentation Experiment

This is the core feature for studying imbalance. It generates synthetic data to meet target ratios for a sensitive attribute (e.g., gender) and the outcome label (e.g., dropout status), then runs the ML pipeline on the newly balanced dataset.

**Command:**
```bash
python main.py <path_to_config> --balanced --sensitive_ratio <0-1> --label_ratio <0-1> --method <name> [options]
```

**Arguments:**
*   `--balanced`: Activates the data augmentation mode.
*   `--sensitive_ratio`: The target ratio for the privileged group (e.g., `0.5` for a 50/50 male/female split).
*   `--label_ratio`: The target ratio for the positive outcome (e.g., `0.5` for a 50/50 dropout/graduate split).
*   `--method`: The data generation method. Choices: `faker`, `llm_async`, `ctgan`, `tvae`.
*   `--api_key`: Required if `method` is `llm_async`.

**Examples:**

1.  **Balance the India dataset to 50/50 gender and dropout ratios using Faker:**
    ```bash
    python main.py yaml/india.yaml \
        --balanced \
        --sensitive_ratio 0.5 \
        --label_ratio 0.5 \
        --method faker \
        --save_results
    ```

2.  **Balance the Brazil dataset to have 30% male students and 70% dropout rate using the LLM method:**
    ```bash
    python main.py yaml/brazil.yaml \
        --balanced \
        --sensitive_ratio 0.3 \
        --label_ratio 0.7 \
        --method llm_async \
        --api_key "your_deepseek_api_key_here" \
        --fairness \
        --save_results
    ```

## The Reproducible Experimental Pipeline

The `experiments/` directory contains a series of shell scripts that automate the entire research process. **Run them in order** to reproduce the study's findings.

### Experiment 1: Model Selection
*   **Objective**: To identify the best-performing baseline ML model for each dataset.
*   **Method**: Runs a 5x2 cross-validation comparing Logistic Regression, Decision Tree, and Random Forest.
*   **Output**: Saves a `best_models.json` file, which is used by subsequent experiments.
*   **How to Run**:
    ```bash
    bash experiments/01_model_selection.sh
    ```

### Experiment 2: Baseline Fairness Assessment
*   **Objective**: To establish the baseline fairness and performance metrics for the best model on each original (unmodified) dataset.
*   **Method**: Uses the `best_models.json` from Experiment 1 to run a detailed fairness analysis on each dataset.
*   **Output**: A detailed report of fairness metrics (Demographic Parity, TPR Difference, etc.) for each dataset, which serves as the "before" state.
*   **How to Run**:
    ```bash
    bash experiments/02_baseline_fairness.sh
    ```

### Experiment 3: Imbalance Impact Study
*   **Objective**: To investigate a critical research question: **Can rebalancing a dataset make a failing fairness intervention succeed?**
*   **Method**: Tests a wide range of imbalance scenarios (e.g., perfect balance, extreme majorities, reversed imbalances). For each scenario, it generates an augmented dataset and runs the fairness pipeline to see if the intervention's effectiveness improves. It specifically looks for "breakthrough" scenarios where bias is significantly reduced.
*   **Output**: A rich set of results showing which data balance configurations are most effective for improving fairness.
*   **How to Run**:
    ```bash
    bash experiments/03_imbalance_impact.sh
    ```

### Experiment 4: Method Selection Criteria Study
*   **Objective**: To perform a head-to-head comparison between the `faker` and `llm_async` data generation methods.
*   **Method**: Re-runs the most impactful scenarios from Experiment 3, once with Faker and once with the LLM. It compares their results on fairness improvement, accuracy cost, and reliability.
*   **Output**: An evidence-based decision framework outlining when to choose Faker versus an LLM for fairness-aware data augmentation.
*   **How to Run**:
    ```bash
    bash experiments/04_method_comparison.sh
    ```

## Source Code Modules (`src/`) Explained

*   `dataload.py`: Handles loading the CSV files. It correctly handles encoding issues (like in the India dataset) and maps the target labels (e.g., 'DROPOUT', 'Graduate') to a binary format (1/0).
*   `preprocess.py`: Implements a standard `scikit-learn` pipeline to prepare the feature data. It imputes missing values, standardizes numerical features, and one-hot encodes categorical features.
*   `model.py`: Contains the training logic for the ML models and the `run_cross_validation` function used in Experiment 1.
*   `fairness.py`: The heart of the fairness analysis.
    *   `FairnessMetrics`: A class to calculate all key fairness metrics.
    *   `FairnessMitigation`: A class that implements fairness techniques. Currently focuses on `reweighting`.
    *   `run_fairness_aware_training`: The main function that trains a baseline model and a fairness-mitigated model and compares them.
*   `utils.py`: A collection of essential helper functions for splitting data, extracting the sensitive attribute column (e.g., 'gender') from the raw data, and printing formatted fairness reports.
*   `synthetic_generator.py`: Implements the `faker`-based data generator. It uses statistical patterns from the reference data and the Faker library to create realistic, but rule-based, synthetic samples.
*   `llm_synthetic_generator.py`: Implements the advanced LLM-based generator. It uses pattern-based generation as a reliable fallback, creating more nuanced and contextually aware synthetic data points based on gender-faculty correlations and other discovered patterns.

## Results

All outputs from running `main.py` or the experiment scripts are saved in the `results/` directory. A new timestamped sub-folder is created for each run (e.g., `results/experiment_20231027_103000/`).

Inside, you will find:
*   **JSON files (`.json`)**: Machine-readable files containing detailed metrics, configurations, and analysis results. These are ideal for programmatic analysis and plotting.
*   **Text summaries (`.txt`)**: Human-readable logs and summary reports, including the formatted fairness reports.
*   **CSV files (`.csv`)**: Data-centric outputs, such as the comparison matrix from Experiment 4.
*   **Augmented Datasets**: The balanced data augmentation experiments save the generated datasets in the `data/` folder for inspection.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request. Areas for potential contribution include:
*   Adding new fairness mitigation techniques (e.g., adversarial debiasing, post-processing).
*   Integrating new datasets.
*   Adding more advanced ML models (e.g., XGBoost, Neural Networks).
*   Improving the synthetic data generation logic.
*   Expanding the experimental pipeline with new research questions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
>>>>>>> fairness-mitigation
