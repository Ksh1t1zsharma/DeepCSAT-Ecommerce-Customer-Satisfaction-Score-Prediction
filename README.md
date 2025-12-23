# DeepCSAT — E-commerce Customer Satisfaction Score Prediction

DeepCSAT is a reproducible repository for training deep learning models that predict customer satisfaction scores (CSAT) for e-commerce use cases. It provides code, documentation, and helper scripts to preprocess data, train an MLP baseline, and evaluate models.

## Features
- End-to-end example pipeline (data loading  training  evaluation)
- Simple, configurable PyTorch MLP baseline
- Scripts to reproduce experiments and export trained models
- Guidance for dataset format and preprocessing

## Repository structure
- README.md - this file
- requirements.txt - Python dependencies
- .gitignore - files to ignore
- LICENSE - project license (MIT)
- CONTRIBUTING.md - how to contribute
- data/README.md - dataset format and download instructions
- src/ - training/evaluation scripts and utilities
- models/ - trained model artifacts (not committed)

## Quick start
1. Clone the repo:
   git clone https://github.com/Ksh1t1zsharma/DeepCSAT-Ecommerce-Customer-Satisfaction-Score-Prediction.git
   cd DeepCSAT-Ecommerce-Customer-Satisfaction-Score-Prediction

2. Create a Python environment and install dependencies:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows
   pip install -r requirements.txt

3. Prepare your dataset according to data/README.md (a CSV with a numeric feature matrix or preprocessing script). Place it in data/

4. Train a baseline model:
   python src/train.py --data-path data/dataset.csv --target satisfaction_score --output-dir models/experiment1

5. Evaluate:
   python src/evaluate.py --data-path data/dataset.csv --model-path models/experiment1/model.pt --scaler-path models/experiment1/scaler.joblib

## Reproducibility
- Use a fixed random seed (provided in scripts)
- Log hyperparameters and results in a structured format (e.g. CSV or MLflow)

## Cite
If you use this repository for research, please cite the repository and include a link to the GitHub page.

## Contact
Repository owner: https://github.com/Ksh1t1zsharma
