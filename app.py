import streamlit as st
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import requests
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="QML Drug Toxicity Predictor", page_icon="⚛️")

# ==============================
# QUANTUM DEVICE
# ==============================
dev = qml.device("default.qubit", wires=2)

# ==============================
# QML CIRCUIT
# ==============================
@qml.qnode(dev)
def quantum_layer(inputs):
    qml.RX(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(inputs[2], wires=0)
    qml.RX(inputs[3], wires=1)
    return qml.expval(qml.PauliZ(0))

# ==============================
# HYBRID MODEL
# ==============================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        x = torch.tanh(x)

        q_out = []
        for i in range(len(x)):
            q_val = quantum_layer(x[i].detach().numpy())
            q_out.append(q_val)

        q_out = torch.tensor(q_out).float().unsqueeze(1)
        return torch.sigmoid(q_out)

model = HybridModel()

# ==============================
# PUBCHEM API
# ==============================
def get_features(name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount/JSON"
        data = requests.get(url).json()
        props = data['PropertyTable']['Properties'][0]

        mw = props.get('MolecularWeight', 0)
        donors = props.get('HBondDonorCount', 0)
        acceptors = props.get('HBondAcceptorCount', 0)
        logp = props.get('XLogP', 0) or 0

        return [float(mw), float(donors), float(acceptors), float(logp)]
    except:
        return None

# ==============================
# NORMALIZATION
# ==============================
def normalize(f):
    return np.array([
        f[0] / 500,
        f[1] / 5,
        f[2] / 5,
        (f[3] + 5) / 10
    ])

# ==============================
# MAIN APP
# ==============================
st.title("⚛️ QML Drug Toxicity Predictor")

# QML Explanation
with st.expander("⚛️ What is happening here?"):
    st.write("""
    This app uses **Quantum Machine Learning**:

    1. Chemical features → encoded as quantum rotations
    2. Quantum circuit → processes data
    3. Measurement → gives prediction

    This is a **hybrid model**:
    Classical NN + Quantum Layer
    """)

name = st.text_input("Enter Chemical Name")

if st.button("Predict"):

    with st.spinner("Fetching data & running quantum circuit..."):

        features = get_features(name)

        if features is None:
            st.error("❌ Failed to fetch data")
        else:
            st.success("✅ Data fetched")

            st.write(f"⚖️ MW: {features[0]}")
            st.write(f"🔗 Donors: {features[1]}")
            st.write(f"🔗 Acceptors: {features[2]}")
            st.write(f"🧪 LogP: {features[3]}")

            norm = normalize(features)
            x = torch.tensor([norm], dtype=torch.float32)

            prob = model(x).item()

            st.subheader("📊 Result")
            st.write(f"Toxicity Risk: {prob*100:.2f}%")

            if prob > 0.7:
                st.error("⚠️ Toxic")
            else:
                st.success("✅ Non-Toxic")

            # 📊 Visualization
            fig, ax = plt.subplots()
            ax.bar(["MW", "Donors", "Acceptors", "LogP"], features)
            st.pyplot(fig)
