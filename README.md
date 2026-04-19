# Drug-Drug Interaction (DDI) Prediction using Graph Neural Networks

This project implements a Graph Neural Network (GNN) model to predict Drug-Drug Interactions (DDIs).

## Project Structure
- `data/raw/`: Place the raw dataset (e.g., DrugBank XML or CSV formats) here.
- `data/processed/`: Preprocessed graphs, node/edge features, and split datasets will be saved here.
- `src/`: Source code for data preprocessing, model definitions, and training logic.
- `models/`: Saved model checkpoints.
- `notebooks/`: Jupyter notebooks for data exploration and experimentation.

## Getting Started

### Prerequisites
Install the required packages using:
```bash
pip install -r requirements.txt
```

### Dataset
This project uses the DrugBank dataset. 
**Important**: Please download the dataset from the official [DrugBank website](https://go.drugbank.com/releases/latest) (requires an academic/commercial license as applicable) and place the raw files (such as the `full database.xml` or extracted CSVs) into the `data/raw/` directory.

### Preprocessing
Once the dataset is in place, you can run the preprocessing script to parse the raw data and prepare the graphs:
```bash
python src/preprocessing.py
```
