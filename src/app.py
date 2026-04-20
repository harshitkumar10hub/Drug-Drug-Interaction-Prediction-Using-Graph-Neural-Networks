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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg: #f4f1eb;
            --surface: #fdfcf9;
            --surface-2: #f0ede6;
            --border: #ddd8ce;
            --border-strong: #c8c2b6;
            --text: #1a1714;
            --text-mid: #4a4540;
            --text-muted: #8a847a;
            --accent: #2d6a4f;
            --accent-light: #e8f5ef;
            --accent-2: #c9714a;
            --accent-2-light: #fdf0ea;
            --danger: #b53a2f;
            --danger-light: #fdf0ef;
            --shadow-sm: 0 1px 3px rgba(26,23,20,0.07), 0 1px 2px rgba(26,23,20,0.05);
            --shadow-md: 0 4px 16px rgba(26,23,20,0.08), 0 2px 6px rgba(26,23,20,0.06);
            --shadow-lg: 0 12px 40px rgba(26,23,20,0.1), 0 4px 12px rgba(26,23,20,0.07);
        }

        body {
            font-family: 'DM Sans', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            background-image:
                radial-gradient(circle at 20% 20%, rgba(45,106,79,0.06) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(201,113,74,0.05) 0%, transparent 50%);
        }

        /* ── Header ── */
        .header {
            text-align: center;
            margin-bottom: 2.5rem;
            animation: rise 0.6s ease both;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--accent-light);
            color: var(--accent);
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 5px 12px;
            border-radius: 100px;
            border: 1px solid rgba(45,106,79,0.2);
            margin-bottom: 1rem;
        }
        .badge::before { content: '●'; font-size: 8px; }
        h1 {
            font-family: 'DM Serif Display', serif;
            font-size: clamp(2rem, 5vw, 2.8rem);
            color: var(--text);
            line-height: 1.15;
            letter-spacing: -0.02em;
        }
        h1 em {
            font-style: italic;
            color: var(--accent);
        }
        .subtitle {
            margin-top: 0.6rem;
            font-size: 14px;
            color: var(--text-muted);
            font-weight: 400;
            max-width: 340px;
            margin-left: auto;
            margin-right: auto;
        }

        /* ── Card ── */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow-lg);
            width: 100%;
            max-width: 480px;
            overflow: hidden;
            animation: rise 0.6s 0.1s ease both;
        }
        .card-body { padding: 2rem; }

        /* ── Drug inputs grid ── */
        .drugs-grid {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 12px;
            align-items: end;
            margin-bottom: 1.5rem;
        }
        .vs-divider {
            display: flex;
            align-items: center;
            justify-content: center;
            padding-bottom: 12px;
        }
        .vs-pill {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--surface-2);
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.05em;
        }

        .form-group { display: flex; flex-direction: column; gap: 6px; }

        label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .drug-num {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--accent);
            color: white;
            font-size: 10px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .drug-num.two { background: var(--accent-2); }

        textarea {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--bg);
            color: var(--text);
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            font-weight: 400;
            line-height: 1.5;
            resize: vertical;
            min-height: 72px;
            transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
        }
        textarea::placeholder { color: var(--border-strong); }
        textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(45,106,79,0.12);
            background: var(--surface);
        }

        /* ── Example pills ── */
        .examples-row {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
        }
        .ex-label {
            font-size: 11px;
            color: var(--text-muted);
            font-weight: 500;
            align-self: center;
            white-space: nowrap;
        }
        .ex-pill {
            font-size: 11px;
            font-family: 'DM Mono', monospace;
            padding: 4px 10px;
            border-radius: 100px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            color: var(--text-mid);
            cursor: pointer;
            transition: all 0.15s;
        }
        .ex-pill:hover {
            background: var(--accent-light);
            border-color: rgba(45,106,79,0.3);
            color: var(--accent);
        }

        /* ── Divider ── */
        .divider {
            height: 1px;
            background: var(--border);
            margin: 0 0 1.5rem;
        }

        /* ── Submit button ── */
        button[type="submit"] {
            width: 100%;
            padding: 14px;
            background: var(--text);
            color: var(--surface);
            border: none;
            border-radius: 12px;
            font-family: 'DM Sans', sans-serif;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            letter-spacing: -0.01em;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: var(--shadow-sm);
        }
        button[type="submit"]:hover {
            background: #2d2925;
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }
        button[type="submit"]:active { transform: translateY(0); }
        .btn-icon { font-size: 18px; }

        /* ── Result ── */
        .result-block {
            margin-top: 1rem;
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            animation: rise 0.4s ease both;
            border: 1px solid;
        }
        .result-block.danger {
            background: var(--danger-light);
            border-color: rgba(181,58,47,0.2);
            color: var(--danger);
        }
        .result-block.safe {
            background: var(--accent-light);
            border-color: rgba(45,106,79,0.2);
            color: var(--accent);
        }
        .result-icon { font-size: 28px; display: block; margin-bottom: 4px; }
        .result-title {
            font-family: 'DM Serif Display', serif;
            font-size: 1.4rem;
            line-height: 1.2;
        }
        .result-subtitle {
            font-size: 12px;
            font-weight: 400;
            opacity: 0.75;
            margin-top: 2px;
        }
        .confidence-bar-wrap {
            margin-top: 12px;
            background: rgba(0,0,0,0.06);
            border-radius: 100px;
            height: 6px;
            overflow: hidden;
        }
        .confidence-bar {
            height: 100%;
            border-radius: 100px;
            background: currentColor;
            opacity: 0.5;
            transition: width 0.6s ease;
        }
        .confidence-label {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            font-weight: 500;
            margin-top: 5px;
            opacity: 0.7;
        }

        /* ── Footer ── */
        .card-footer {
            background: var(--surface-2);
            border-top: 1px solid var(--border);
            padding: 0.85rem 2rem;
            display: flex;
            gap: 1.5rem;
            justify-content: center;
        }
        .stat {
            text-align: center;
        }
        .stat-val {
            font-family: 'DM Serif Display', serif;
            font-size: 1.1rem;
            color: var(--text);
        }
        .stat-lbl {
            font-size: 10px;
            color: var(--text-muted);
            font-weight: 500;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .stat-sep {
            width: 1px;
            background: var(--border);
            align-self: stretch;
        }

        /* ── Info strip under card ── */
        .info-strip {
            margin-top: 1.2rem;
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--text-muted);
            animation: rise 0.6s 0.2s ease both;
        }
        .info-strip span { color: var(--border-strong); }

        @keyframes rise {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <header class="header">
        <div class="badge">Graph Neural Network</div>
        <h1>Drug–Drug<br><em>Interaction</em> Predictor</h1>
        <p class="subtitle">Enter two SMILES strings to predict whether a pharmacological interaction exists.</p>
    </header>

    <div class="card">
        <div class="card-body">
            <form method="POST" action="/predict">

                <div class="drugs-grid">
                    <div class="form-group">
                        <label><span class="drug-num">1</span>Drug A</label>
                        <textarea id="smiles1" name="smiles1" required placeholder="SMILES…">{{ request.form.get('smiles1', '') }}</textarea>
                    </div>

                    <div class="vs-divider">
                        <div class="vs-pill">vs</div>
                    </div>

                    <div class="form-group">
                        <label><span class="drug-num two">2</span>Drug B</label>
                        <textarea id="smiles2" name="smiles2" required placeholder="SMILES…">{{ request.form.get('smiles2', '') }}</textarea>
                    </div>
                </div>

                <div class="examples-row">
                    <span class="ex-label">Try:</span>
                    <button type="button" class="ex-pill" onclick="fillExample('aspirin','caffeine')">Aspirin + Caffeine</button>
                    <button type="button" class="ex-pill" onclick="fillExample('warfarin','ibuprofen')">Warfarin + Ibuprofen</button>
                </div>

                <div class="divider"></div>

                <button type="submit">
                    <span class="btn-icon">⬡</span> Analyse Interaction
                </button>

                {% if result %}
                <div class="result-block {% if 'Interaction Detected' in result %}danger{% else %}safe{% endif %}">
                    <span class="result-icon">{% if 'Interaction Detected' in result %}⚠️{% else %}✅{% endif %}</span>
                    <div class="result-title">{{ result }}</div>
                    {% if prob %}
                    <div class="result-subtitle">Model confidence</div>
                    <div class="confidence-bar-wrap">
                        <div class="confidence-bar" style="width: {{ '%.1f'|format(prob * 100) }}%"></div>
                    </div>
                    <div class="confidence-label">
                        <span>0%</span>
                        <span>{{ '%.1f'|format(prob * 100) }}%</span>
                        <span>100%</span>
                    </div>
                    {% endif %}
                </div>
                {% endif %}

            </form>
        </div>

        <div class="card-footer">
            <div class="stat">
                <div class="stat-val">3×</div>
                <div class="stat-lbl">GAT Layers</div>
            </div>
            <div class="stat-sep"></div>
            <div class="stat">
                <div class="stat-val">64</div>
                <div class="stat-lbl">Hidden Dim</div>
            </div>
            <div class="stat-sep"></div>
            <div class="stat">
                <div class="stat-val">GNN</div>
                <div class="stat-lbl">Architecture</div>
            </div>
            <div class="stat-sep"></div>
            <div class="stat">
                <div class="stat-val">RDKit</div>
                <div class="stat-lbl">Mol Parser</div>
            </div>
        </div>
    </div>

    <p class="info-strip">
        <span>⚠</span> For research use only — not a substitute for clinical drug review.
    </p>

    <script>
        const examples = {
            aspirin:   'CC(=O)Oc1ccccc1C(=O)O',
            caffeine:  'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
            warfarin:  'CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O',
            ibuprofen: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'
        };
        function fillExample(d1, d2) {
            document.getElementById('smiles1').value = examples[d1] || '';
            document.getElementById('smiles2').value = examples[d2] || '';
        }
    </script>
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
    
    if g1.x.sum() == 0 or g2.x.sum() == 0:
        return render_template_string(HTML_TEMPLATE, result="Invalid SMILES structure.")
        
    b_g1 = Batch.from_data_list([g1]).to(device)
    b_g2 = Batch.from_data_list([g2]).to(device)
    
    with torch.no_grad():
        out = model(b_g1, b_g2)
        probs = F.softmax(out, dim=1)
        max_prob = probs.max().item()
        pred_class = probs.argmax().item()
        
        if pred_class == 1:
            result = 'Interaction Detected'
        else:
            result = 'No Interaction'
            
    return render_template_string(HTML_TEMPLATE, result=result, prob=max_prob)

if __name__ == '__main__':
    app.run(debug=True, port=5000)