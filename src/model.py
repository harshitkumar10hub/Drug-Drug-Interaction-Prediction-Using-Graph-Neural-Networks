import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
from torch_geometric.data import Data

def get_atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic())
    ]

def smiles_to_graph(smiles):
    """
    Converts a SMILES string into a PyTorch Geometric Data object.
    atoms = nodes, bonds = edges.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Fallback to single dummy node for invalid SMILES
        return Data(x=torch.zeros((1, 4), dtype=torch.float), edge_index=torch.empty((2, 0), dtype=torch.long))

    x = []
    for atom in mol.GetAtoms():
        x.append(get_atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Molecular graphs are typically modeled as undirected
        edges.append([i, j])
        edges.append([j, i])
        
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return Data(x=x, edge_index=edge_index)

class DDIGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(DDIGNN, self).__init__()
        # 3 GATConv layers
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        
        # Concatenate both arrays later, hence hidden_channels * 2
        self.lin = Linear(hidden_channels * 2, num_classes)

    def forward_graph(self, x, edge_index, batch):
        # Pass through the 3 GATConv layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global Mean Pooling
        x = global_mean_pool(x, batch)
        return x

    def forward(self, g1, g2):
        # Extract embeddings for both drug graphs
        emb1 = self.forward_graph(g1.x, g1.edge_index, g1.batch)
        emb2 = self.forward_graph(g2.x, g2.edge_index, g2.batch)
        
        # Concatenate graph embeddings of both drugs
        emb = torch.cat([emb1, emb2], dim=1)
        
        # Final Linear layer for classification
        out = self.lin(emb)
        return out
