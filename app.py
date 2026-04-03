import streamlit as st
import torch
import torch.nn as nn
import json
import os

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Drug Toxicity Predictor", page_icon="🧬")

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
}
</style>
""", unsafe_allow_html=True)

# ==============================
# USER SYSTEM
# ==============================
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        return json.load(open(USER_FILE))
    return {}

def save_users(users):
    json.dump(users, open(USER_FILE, "w"))

# Session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

# ==============================
# LOGIN / SIGNUP
# ==============================
def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()
        if u in users and users[u] == p:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    if st.button("Signup"):
        st.session_state.page = "signup"
        st.rerun()

def signup():
    st.title("📝 Signup")
    u = st.text_input("Create Username")
    p = st.text_input("Create Password", type="password")

    if st.button("Create Account"):
        users = load_users()
        if u in users:
            st.error("User exists")
        else:
            users[u] = p
            save_users(users)
            st.success("Account created")
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
    st.error(f"Model load error: {e}")
    st.stop()

# ==============================
# FEATURE EXTRACTION (RDKit)
# ==============================
def get_features_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    donors = Descriptors.NumHDonors(mol)
    acceptors = Descriptors.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)

    return [mw, donors, acceptors, logp]

# ==============================
# MAIN APP
# ==============================
def main_app():
    st.title("🧬 Drug Toxicity Predictor")

    mode = st.radio("Select Input Mode", ["Auto (SMILES)", "Manual"])

    # ==========================
    # AUTO MODE
    # ==========================
    if mode == "Auto (SMILES)":
        st.subheader("⚛️ Enter SMILES")

        smiles = st.text_input(
            "SMILES String",
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
        )

        if st.button("Auto Calculate & Predict"):
            features = get_features_from_smiles(smiles)

            if features is None:
                st.error("Invalid SMILES ❌")
                return

            st.success("Features extracted ✅")

            st.write(f"⚖️ Molecular Weight: {features[0]:.2f}")
            st.write(f"🔗 H-Bond Donors: {features[1]}")
            st.write(f"🔗 H-Bond Acceptors: {features[2]}")
            st.write(f"🧪 LogP: {features[3]:.2f}")

            x = torch.tensor([features], dtype=torch.float32)
            prob = model(x).item()

            st.subheader("📊 Result")
            st.write(f"Toxicity Risk: {prob*100:.2f}%")

            if prob > 0.6:
                st.error("⚠️ Toxic")
            else:
                st.success("✅ Non-Toxic")

    # ==========================
    # MANUAL MODE
    # ==========================
    else:
        st.subheader("📊 Manual Input")

        f1 = st.number_input("Molecular Weight", value=200.0)
        f2 = st.number_input("H-Bond Donors", value=1.0)
        f3 = st.number_input("H-Bond Acceptors", value=2.0)
        f4 = st.number_input("LogP", value=1.0)

        if st.button("Predict"):
            x = torch.tensor([[f1, f2, f3, f4]], dtype=torch.float32)
            prob = model(x).item()

            st.write(f"Toxicity Risk: {prob*100:.2f}%")

            if prob > 0.6:
                st.error("⚠️ Toxic")
            else:
                st.success("✅ Non-Toxic")

    # Logout
    if st.button("Logout"):
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
