import os
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template_string
from model import DDIGNN, smiles_to_graph
from torch_geometric.data import Batch

app = Flask(__name__)

# Dynamically load the model based on its saved dimensions
MODEL_PATH = 'models/gnn_model.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading GNN model...")
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # The linear layer weight shape is (num_classes, hidden_channels * 2)
    # Using this to instantiate the architecture locally
    num_classes = state_dict['lin.weight'].shape[0]
    model = DDIGNN(num_node_features=4, hidden_channels=64, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found!")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DDI Prediction via GNN</title>
    <style>
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-image: radial-gradient(circle at top right, #1e293b, #0f172a);
        }
        .container {
            background: rgba(30, 41, 59, 0.7);
            padding: 2rem 3rem;
            border-radius: 16px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            max-width: 420px;
            width: 100%;
        }
        h2 {
            text-align: center;
            margin-top: 0;
            margin-bottom: 28px;
            color: #38bdf8;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-size: 14px;
            font-weight: 500;
            color: #cbd5e1;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid #334155;
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.8);
            color: #fff;
            box-sizing: border-box;
            font-family: monospace;
            font-size: 15px;
            transition: all 0.2s ease;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #38bdf8;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
            background: rgba(15, 23, 42, 1);
        }
        button {
            width: 100%;
            padding: 14px;
            margin-top: 8px;
            background: linear-gradient(135deg, #0ea5e9, #38bdf8);
            color: #0f172a;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.3);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.4);
            background: linear-gradient(135deg, #0284c7, #0ea5e9);
        }
        button:active {
            transform: translateY(0);
        }
        .result {
            margin-top: 28px;
            text-align: center;
            padding: 18px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 700;
            animation: fadeIn 0.4s ease-out forwards;
        }
        .interaction-detected {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
            box-shadow: inset 0 0 20px rgba(239, 68, 68, 0.05);
        }
        .no-interaction {
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
            box-shadow: inset 0 0 20px rgba(34, 197, 94, 0.05);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>DDI Predictor</h2>
        <form method="POST" action="/predict">
            <div class="form-group">
                <label for="smiles1">Drug 1 (SMILES):</label>
                <input type="text" id="smiles1" name="smiles1" required placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O">
            </div>
            <div class="form-group">
                <label for="smiles2">Drug 2 (SMILES):</label>
                <input type="text" id="smiles2" name="smiles2" required placeholder="e.g. CN1C=NC2=C1C(=O)N(C(=O)N2C)C">
            </div>
            <button type="submit">Predict Interaction</button>
        </form>

        {% if result %}
            <div class="result {% if 'Interaction Detected' in result %}interaction-detected{% else %}no-interaction{% endif %}">
                {{ result }}
                {% if prob %}
                <div style="font-size: 13px; font-weight: normal; margin-top: 6px; opacity: 0.8; font-family: monospace;">
                    Confidence Score: {{ "%.2f"|format(prob * 100) }}%
                </div>
                {% endif %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, result="Model not loaded!")
        
    smiles1 = request.form.get('smiles1', '').strip()
    smiles2 = request.form.get('smiles2', '').strip()
    
    if not smiles1 or not smiles2:
        return render_template_string(HTML_TEMPLATE, result="Please provide both SMILES.")
        
    g1 = smiles_to_graph(smiles1)
    g2 = smiles_to_graph(smiles2)
    
    # Identify if RDKit failed (dummy graph with 0 edges inside)
    if g1.x.sum() == 0 or g2.x.sum() == 0:
        return render_template_string(HTML_TEMPLATE, result="Invalid SMILES structure.")
        
    # Collate batch
    b_g1 = Batch.from_data_list([g1]).to(device)
    b_g2 = Batch.from_data_list([g2]).to(device)
    
    # Predict
    with torch.no_grad():
        out = model(b_g1, b_g2)
        probs = F.softmax(out, dim=1)
        max_prob = probs.max().item()
        pred_class = probs.argmax().item()
        
        # Binary translation from our multiclass formulation
        # For demonstration purposes, if confidence is very low it triggers NO INTERACTION
        if pred_class == 1:
            result = 'Interaction Detected'
        else:
            result = 'No Interaction'
            
    return render_template_string(HTML_TEMPLATE, result=result, prob=max_prob)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
