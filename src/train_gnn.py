import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
import os

from model import DDIGNN, smiles_to_graph

class DrugPairDataset(Dataset):
    def __init__(self, g1_list, g2_list, labels):
        self.g1_list = g1_list
        self.g2_list = g2_list
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.g1_list[idx], self.g2_list[idx], self.labels[idx]

def custom_collate(batch):
    g1_list, g2_list, y_list = zip(*batch)
    batch_g1 = Batch.from_data_list(g1_list)
    batch_g2 = Batch.from_data_list(g2_list)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return batch_g1, batch_g2, y_tensor

def train_gnn():
    print("Loading dataset: data/processed/drug_pairs.csv ...")
    df = pd.read_csv('data/processed/drug_pairs.csv')
    print("Filtering logic for binary classification (labels 0 and 1)...")
    df = df[df['interaction_label'].isin([0, 1])]
    
    print("Balancing positive and negative samples...")
    if len(df) > 0 and len(df['interaction_label'].unique()) == 2:
        min_class_size = df['interaction_label'].value_counts().min()
        df_0 = df[df['interaction_label'] == 0].sample(n=min_class_size, random_state=42)
        df_1 = df[df['interaction_label'] == 1].sample(n=min_class_size, random_state=42)
        df = pd.concat([df_0, df_1])
    
    # Subsample due to local testing speed constraints
    sample_size = min(200000, len(df))
    print(f"Sampling {sample_size} records for balanced GNN training...")
    df = df.sample(n=sample_size, random_state=42)
    print(f"Total balanced pairs ready for training: {len(df)}")
    
    print("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(df['interaction_label'].astype(str))
    
    print("Converting SMILES to Graph Data objects...")
    g1_list = [smiles_to_graph(s) for s in df['smiles1']]
    g2_list = [smiles_to_graph(s) for s in df['smiles2']]
    
    print("Splitting dataset into train and test sets...")
    split_res = train_test_split(g1_list, g2_list, y, test_size=0.2, random_state=42, stratify=y)
    train_g1, test_g1, train_g2, test_g2, y_train, y_test = split_res
    
    train_dataset = DrugPairDataset(train_g1, train_g2, y_train)
    test_dataset = DrugPairDataset(test_g1, test_g2, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_node_features = 4  # from get_atom_features
    num_classes = len(np.unique(y))
    hidden_channels = 64
    
    print(f"Initializing GNN model (classes: {num_classes}, device: {device})...")
    model = DDIGNN(num_node_features, hidden_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    epochs = 5
    print("Training the GNN...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for b_g1, b_g2, b_y in train_loader:
            b_g1, b_g2, b_y = b_g1.to(device), b_g2.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_g1, b_g2)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * b_y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == b_y).sum().item()
            total += b_y.size(0)
        
        avg_loss = total_loss / total
        train_acc = correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
    print("Evaluating GNN...")
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for b_g1, b_g2, b_y in test_loader:
            b_g1, b_g2, b_y = b_g1.to(device), b_g2.to(device), b_y.to(device)
            out = model(b_g1, b_g2)
            
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(b_y.cpu().numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    acc = accuracy_score(all_labels, all_preds)
    
    if num_classes > 2:
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
    else:
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
        f1 = f1_score(all_labels, all_preds)
        
    print("\n--- GNN Evaluation Results ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUROC    : {auroc:.4f}")
    print("------------------------------\n")
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/gnn_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_gnn()
