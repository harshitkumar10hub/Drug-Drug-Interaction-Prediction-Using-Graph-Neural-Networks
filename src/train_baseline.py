import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

def get_morgan_fingerprint(smiles, radius=2, n_bits=512):
    """
    Computes Morgan fingerprint from a SMILES string.
    Using 512 bits per fingerprint (total 1024) to keep baseline fast and manageable.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((n_bits,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros((n_bits,))

if __name__ == '__main__':
    print("Loading dataset: data/processed/drug_pairs.csv ...")
    df = pd.read_csv('data/processed/drug_pairs.csv')
    
    print(f"Total pairs in dataset: {len(df)}")
    
    # Subsample data. Training XGBoost on 4.5 million interactions locally is too slow for a baseline setup.
    # Restrict to 5,000 pairs
    sample_size = min(5000, len(df))
    print(f"Sampling {sample_size} records to train baseline model quickly...")
    df = df.sample(n=sample_size, random_state=42)
    
    print("Filtering rare interaction types so train/test splits are stable...")
    v_counts = df['interaction_label'].value_counts()
    valid_labels = v_counts[v_counts >= 5].index
    df = df[df['interaction_label'].isin(valid_labels)]
    print(f"Pairs remaining after dropping rare labels: {len(df)}")

    print("Computing Morgan fingerprints using RDKit...")
    X1 = np.array([get_morgan_fingerprint(s) for s in df['smiles1']])
    X2 = np.array([get_morgan_fingerprint(s) for s in df['smiles2']])
    
    print("Concatenating features...")
    X = np.concatenate([X1, X2], axis=1)
    
    print("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(df['interaction_label'].astype(str))
    
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training XGBoost classifier...")
    n_classes = len(np.unique(y))
    
    if n_classes > 2:
        model = xgb.XGBClassifier(
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1
        )
    else:
        model = xgb.XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1
        )
        
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    if n_classes > 2:
        y_prob = model.predict_proba(X_test)
        # multi_class='ovr' for multiclass AUROC
        auroc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- Baseline Results ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUROC    : {auroc:.4f}")
    print("------------------------\n")
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/baseline_model.pkl'
    # Save the model and label encoder together
    joblib.dump({'model': model, 'label_encoder': le}, model_path)
    print(f"Trained model saved to {model_path}")
