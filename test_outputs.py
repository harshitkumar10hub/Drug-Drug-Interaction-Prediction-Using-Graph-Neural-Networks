import torch
from src.model import DDIGNN, smiles_to_graph
from torch_geometric.data import Batch

def test_model():
    print("Loading model for test...")
    state_dict = torch.load('models/gnn_model.pt', map_location='cpu')
    num_classes = state_dict['lin.weight'].shape[0]
    model = DDIGNN(4, 64, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    
    g1 = smiles_to_graph('C')
    g2 = smiles_to_graph('O')
    out1 = model(Batch.from_data_list([g1]), Batch.from_data_list([g2]))
    print(f"Output for C-O: {out1.detach().numpy()}")
    
    g3 = smiles_to_graph('CC(=O)OC1=CC=CC=C1C(=O)O')
    g4 = smiles_to_graph('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
    out2 = model(Batch.from_data_list([g3]), Batch.from_data_list([g4]))
    print(f"Output for Complex-Complex: {out2.detach().numpy()}")
    
    if torch.allclose(out1, out2):
        print("TEST FAILED: Outputs are identical!")
    else:
        print("TEST PASSED: Outputs are different.")

if __name__ == "__main__":
    test_model()
