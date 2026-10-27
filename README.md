Customer Support Classifier

A machine learning project for classifying customer support tickets by category and priority.

Features
Customer support ticket preprocessing
Baseline machine learning models
DistilBERT-based classification
Model comparison and evaluation reports
API application and automated tests
Installation

Create and activate a Python virtual environment, then install the dependencies:

bash
pip install -r requirements.txt

Project Structure
app/ — application and API tests
data/ — dataset files
reports/ — model evaluation reports
results/ — metrics, confusion matrices, and error analysis
prepare_dataset.py — dataset preparation
train_baseline.py — baseline model training
train_distilbert.py — DistilBERT training
compare_models.py — model comparison
tests/ — project tests
Usage

Run the relevant training scripts from the project root. Refer to the source code and requirements.txt for implementation details and dependencies.

Notes

Large trained model checkpoints are excluded from Git using .gitignore. Generate or download them separately when needed.