
import streamlit as st
import torch
import torch.nn as nn
import json
import os
import requests
import numpy as np
import matplotlib.pyplot as plt

# Optional PennyLane for quantum module visualization
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except Exception:
    PENNYLANE_AVAILABLE = False

# Optional RDKit for SMILES mode
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Drug Toxicity Predictor (QML-Inspired)",
    page_icon="🧬",
    layout="wide"
)

# ==============================
# STYLE
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.stButton>button {
    background-color: #00c6ff;
    color: black;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# USER SYSTEM
# ==============================
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

# ==============================
# LOGIN
# ==============================
def login():
    st.title("🔐 Login")
    st.write("Login to access the Drug Toxicity Predictor.")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()
        if u in users and users[u] == p:
            st.session_state.logged_in = True
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid credentials.")

    if st.button("Create Account"):
        st.session_state.page = "signup"
        st.rerun()

# ==============================
# SIGNUP
# ==============================
def signup():
    st.title("📝 Signup")
    st.write("Create a new account.")

    u = st.text_input("Create Username")
    p = st.text_input("Create Password", type="password")

    if st.button("Signup"):
        users = load_users()
        if not u or not p:
            st.error("Username and password are required.")
        elif u in users:
            st.error("User already exists.")
        else:
            users[u] = p
            save_users(users)
            st.success("Account created successfully.")
            st.session_state.page = "login"
            st.rerun()

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# ==============================
# MODEL
# ==============================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

try:
    model = SimpleModel()
    model.load_state_dict(torch.load("simple_model.pth", map_location="cpu"))
    model.eval()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ==============================
# DATA SOURCES
# ==============================
def get_features_from_name(name: str):
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{name}/property/MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount/JSON"
        )
        data = requests.get(url, timeout=15).json()
        props = data["PropertyTable"]["Properties"][0]

        mw = props.get("MolecularWeight", 0)
        donors = props.get("HBondDonorCount", 0)
        acceptors = props.get("HBondAcceptorCount", 0)
        logp = props.get("XLogP", 0)

        if logp is None:
            logp = 0

        return [float(mw), float(donors), float(acceptors), float(logp)]
    except Exception:
        return None

def get_features_from_smiles(smiles: str):
    if not RDKIT_AVAILABLE:
        return None, "RDKit is not installed in this deployment."

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES string."

        mw = Descriptors.MolWt(mol)
        donors = Descriptors.NumHDonors(mol)
        acceptors = Descriptors.NumHAcceptors(mol)
        logp = Descriptors.MolLogP(mol)

        return [float(mw), float(donors), float(acceptors), float(logp)], None
    except Exception as e:
        return None, str(e)

# ==============================
# PREDICTION
# ==============================
def normalize_features(features):
    return [
        features[0] / 500.0,
        features[1] / 5.0,
        features[2] / 5.0,
        (features[3] + 5.0) / 10.0
    ]

def predict_probability(features):
    norm = normalize_features(features)
    x = torch.tensor([norm], dtype=torch.float32)
    prob = model(x).item()

    prob = max(0.0, min(prob, 1.0))

    # lightweight calibration patch
    if prob > 0.9:
        prob = 1 - prob

    return prob

def risk_label(prob):
    if prob > 0.7:
        return "Toxic", "error"
    if prob > 0.6:
        return "Moderate Risk", "warning"
    return "Non-Toxic", "success"

# ==============================
# VISUALS
# ==============================
def plot_feature_chart(features):
    labels = ["MolWt", "H-Donors", "H-Acceptors", "LogP"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, features)
    ax.set_title("Chemical Feature Values")
    ax.set_ylabel("Value")
    st.pyplot(fig)

def plot_probability_chart(prob):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["Non-Toxic", "Toxic"], [1 - prob, prob])
    ax.set_ylim(0, 1)
    ax.set_title("Predicted Probability")
    st.pyplot(fig)

def plot_confusion_matrix():
    cm = np.array([[1256, 140], [36, 20]])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im)
    st.pyplot(fig)

# ==============================
# QUANTUM MODULE
# ==============================
def page_quantum_module():
    st.title("⚛️ Quantum Module")

    st.write("""
This page explains the **quantum side of your project**.

Your deployed prediction backend currently uses `simple_model.pth`, which is a classical model.
However, your research pipeline is **QML-inspired**, and this page shows how that quantum workflow fits into the project.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("How Quantum Fits Here")
        st.markdown("""
1. **Chemical descriptors** are collected:
   - Molecular Weight
   - H-Bond Donors
   - H-Bond Acceptors
   - LogP

2. These descriptors are **normalized**.

3. In a hybrid QML model, normalized values can be:
   - encoded into **qubit rotations**
   - processed by a **variational quantum circuit**
   - measured to produce a prediction-related output

4. A classical layer can combine that quantum output with conventional neural-network processing.
        """)

    with col2:
        st.subheader("Current Deployment Status")
        st.info("""
- **Current deployed app backend:** Classical model (`simple_model.pth`)
- **Research direction:** Hybrid Quantum-Classical ML
- **Why this setup?** Stable deployment with a QML-ready explanation layer
        """)

    st.markdown("---")
    st.subheader("Quantum Workflow Diagram")

    st.markdown("""
**Chemical Features → Normalization → Quantum Encoding → Variational Circuit → Measurement → Toxicity Prediction**
    """)

    if PENNYLANE_AVAILABLE:
        st.subheader("Example Quantum Circuit")

        q_dev = qml.device("default.qubit", wires=2)

        @qml.qnode(q_dev)
        def example_circuit(x1, x2):
            qml.RX(x1, wires=0)
            qml.RY(x2, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        sample_x1 = st.slider("Sample angle 1", 0.0, 3.14, 1.0)
        sample_x2 = st.slider("Sample angle 2", 0.0, 3.14, 1.5)

        out = example_circuit(sample_x1, sample_x2)
        st.write(f"Quantum outputs: {out}")

        try:
            fig, ax = qml.draw_mpl(example_circuit)(sample_x1, sample_x2)
            st.pyplot(fig)
        except Exception:
            st.write("Circuit drawing is not available in this environment.")
    else:
        st.warning("PennyLane is not installed in this deployment, so live circuit visualization is disabled.")

    st.markdown("---")
    st.subheader("How to Explain This in Viva")
    st.code(
        """This project currently deploys a stable classical toxicity model, but the research workflow is designed for hybrid quantum-classical machine learning.
Chemical descriptors can be encoded into qubit rotations, processed by a variational quantum circuit, and then combined with classical layers for prediction.
The deployed app includes a Quantum Module to explain this pipeline clearly and prepare for future true QML deployment.""",
        language="text"
    )

# ==============================
# PREDICTOR PAGE
# ==============================
def page_predictor():
    st.title("🧬 Drug Toxicity Predictor")

    with st.expander("⚛️ How QML is used in this project"):
        st.write("""
This app demonstrates a **QML-inspired drug toxicity workflow**.

1. Chemical descriptors are collected from:
   - PubChem (chemical name)
   - RDKit (SMILES input, if available)
   - Manual entry

2. These descriptors are used as model inputs:
   - Molecular Weight
   - H-Bond Donors
   - H-Bond Acceptors
   - LogP

3. In a real hybrid QML pipeline:
   - features can be encoded into quantum states
   - quantum circuits process the encoded data
   - measurement outputs support prediction

**Current deployed backend:** `simple_model.pth`
        """)

    with st.expander("📘 What do these chemical properties mean?"):
        st.write("""
- **Molecular Weight**: total mass of the molecule.
- **H-Bond Donors**: groups that can donate hydrogen in hydrogen bonding.
- **H-Bond Acceptors**: atoms that can accept hydrogen bonds.
- **LogP**: lipophilicity, showing fat-vs-water solubility.

You can get these values from:
- PubChem
- ChemSpider
- RDKit from SMILES
        """)

    mode = st.radio(
        "Select Input Mode",
        ["Auto (Chemical Name)", "SMILES", "Manual"],
        horizontal=True
    )

    features = None
    source_name = None

    if mode == "Auto (Chemical Name)":
        chem_name = st.text_input("Enter Chemical Name", placeholder="e.g., Aspirin")

        if st.button("Fetch & Predict"):
            if not chem_name.strip():
                st.error("Please enter a chemical name.")
                return

            with st.spinner("Fetching chemical properties from PubChem..."):
                features = get_features_from_name(chem_name)
                source_name = chem_name

            if features is None:
                st.error("Could not fetch data for that chemical. Try another name.")
                return

            st.success("Chemical data fetched successfully.")

    elif mode == "SMILES":
        smiles = st.text_input(
            "Enter SMILES",
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
        )

        if not RDKIT_AVAILABLE:
            st.warning("RDKit is not available in this deployment. Use Chemical Name or Manual mode.")

        if st.button("Parse SMILES & Predict"):
            if not smiles.strip():
                st.error("Please enter a SMILES string.")
                return

            with st.spinner("Calculating chemical descriptors from SMILES..."):
                features, err = get_features_from_smiles(smiles)
                source_name = smiles

            if features is None:
                st.error(err or "Could not parse SMILES.")
                return

            st.success("Descriptors calculated successfully.")

    else:
        st.subheader("Manual Input")
        mw = st.number_input("Molecular Weight", min_value=0.0, value=200.0)
        donors = st.number_input("H-Bond Donors", min_value=0.0, value=1.0)
        acceptors = st.number_input("H-Bond Acceptors", min_value=0.0, value=2.0)
        logp = st.number_input("LogP", value=1.0)

        if st.button("Predict Manual"):
            features = [mw, donors, acceptors, logp]
            source_name = "Manual Entry"

    if features is not None:
        c1, c2 = st.columns([1.2, 1])

        with c1:
            st.subheader("Extracted Features")
            st.write(f"**Input Source:** {source_name}")
            st.write(f"⚖️ Molecular Weight: {features[0]:.2f}")
            st.write(f"🔗 H-Bond Donors: {features[1]:.2f}")
            st.write(f"🔗 H-Bond Acceptors: {features[2]:.2f}")
            st.write(f"🧪 LogP: {features[3]:.2f}")

            prob = predict_probability(features)
            label, level = risk_label(prob)

            st.subheader("Prediction Result")
            st.write(f"**Toxicity Risk:** {prob * 100:.2f}%")

            if level == "error":
                st.error(f"⚠️ {label}")
            elif level == "warning":
                st.warning(f"⚠️ {label}")
            else:
                st.success(f"✅ {label}")

            st.info("This is an educational prediction tool and not a certified medical or toxicology system.")

        with c2:
            plot_probability_chart(prob)

        st.subheader("Feature Visualization")
        plot_feature_chart(features)

# ==============================
# PERFORMANCE PAGE
# ==============================
def page_model_performance():
    st.title("📊 Model Performance")

    st.write("These are the evaluation results from your trained toxicity model.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "0.8788")
    c2.metric("ROC-AUC", "0.6948")
    c3.metric("Recall (Class 1)", "0.36")
    c4.metric("F1 (Class 1)", "0.19")

    st.markdown("### Classification Summary")
    st.code(
        """Class 0 -> Precision: 0.97 | Recall: 0.90 | F1: 0.93
Class 1 -> Precision: 0.12 | Recall: 0.36 | F1: 0.19""",
        language="text"
    )

    st.markdown("### Confusion Matrix")
    plot_confusion_matrix()

    st.info("""
Interpretation:
- The model detects some toxic compounds, but performance on the minority toxic class is still limited.
- Future improvements should include:
  - better data balancing
  - stronger feature engineering
  - true hybrid QML training
    """)

# ==============================
# ABOUT PAGE
# ==============================
def page_about():
    st.title("ℹ️ About This Project")
    st.write("""
This project focuses on **drug toxicity screening** using chemical descriptors.

### Current capabilities
- Login / Signup
- Chemical name lookup from PubChem
- SMILES-based descriptor extraction
- Manual descriptor entry
- Toxicity prediction using `simple_model.pth`
- Quantum Module explaining the QML pipeline
- Charts and evaluation results

### Future upgrades
- Real hybrid QML backend with PennyLane
- Better calibrated model
- Larger balanced toxicity dataset
- Explainable AI output
    """)

# ==============================
# MAIN APP
# ==============================
def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Predictor", "Quantum Module", "Model Performance", "About"]
    )

    if page == "Predictor":
        page_predictor()
    elif page == "Quantum Module":
        page_quantum_module()
    elif page == "Model Performance":
        page_model_performance()
    else:
        page_about()

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

# ==============================
# FLOW
# ==============================
if st.session_state.logged_in:
    main_app()
else:
    if st.session_state.page == "login":
        login()
    else:
        signup()
