import streamlit as st
import torch
import torch.nn as nn
import json
import os
import requests

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Drug Toxicity Predictor (QML)", page_icon="🧬")

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
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid credentials ❌")

    if st.button("Create Account"):
        st.session_state.page = "signup"
        st.rerun()

def signup():
    st.title("📝 Signup")
    u = st.text_input("Create Username")
    p = st.text_input("Create Password", type="password")

    if st.button("Signup"):
        users = load_users()
        if u in users:
            st.error("User already exists ❌")
        else:
            users[u] = p
            save_users(users)
            st.success("Account created ✅")
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

# Load model safely
try:
    model = SimpleModel()
    model.load_state_dict(torch.load("simple_model.pth", map_location="cpu"))
    model.eval()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ==============================
# PUBCHEM API (SAFE)
# ==============================
def get_features_from_name(name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount/JSON"
        data = requests.get(url).json()
        props = data['PropertyTable']['Properties'][0]

        mw = props.get('MolecularWeight', 0)
        donors = props.get('HBondDonorCount', 0)
        acceptors = props.get('HBondAcceptorCount', 0)
        logp = props.get('XLogP', 0)

        # Handle None
        if logp is None:
            logp = 0

        return [float(mw), float(donors), float(acceptors), float(logp)]

    except:
        return None

# ==============================
# MAIN APP
# ==============================
def main_app():
    st.title("🧬 Drug Toxicity Predictor")

    # ------------------------------
    # QML Explanation
    # ------------------------------
    with st.expander("⚛️ How QML is used in this project"):
        st.markdown("""
        This application demonstrates a **Quantum Machine Learning (QML)-inspired model**.

        🔹 Chemical properties are extracted from PubChem  
        🔹 These features act as input to a neural model  
        🔹 In real QML:
            - Features → encoded into quantum states
            - Quantum circuits → process information
            - Measurement → gives prediction

        ⚠️ Current implementation:
        - Uses classical neural network (PyTorch)
        - Simulates QML workflow conceptually

        ✅ Advantage:
        - Shows how QML can be applied in drug discovery
        """)

    mode = st.radio("Select Mode", ["Auto (Chemical Name)", "Manual"])

    # ------------------------------
    # AUTO MODE
    # ------------------------------
    if mode == "Auto (Chemical Name)":
        chem_name = st.text_input("Enter Chemical Name", placeholder="e.g., Aspirin")

        if st.button("Predict"):
            if not chem_name:
                st.error("Please enter chemical name")
                return

            features = get_features_from_name(chem_name)

            if features is None:
                st.error("❌ Could not fetch data. Try another chemical.")
                return

            st.success("✅ Data fetched")

            st.write(f"⚖️ Molecular Weight: {features[0]}")
            st.write(f"🔗 H-Bond Donors: {features[1]}")
            st.write(f"🔗 H-Bond Acceptors: {features[2]}")
            st.write(f"🧪 LogP: {features[3]}")

            try:
                x = torch.tensor([features], dtype=torch.float32)
                prob = model(x).item()
            except:
                st.error("Prediction failed")
                return

            st.subheader("📊 Result")
            st.write(f"🧪 Chemical: {chem_name}")
            st.write(f"⚠️ Toxicity Risk: {prob*100:.2f}%")

            if prob > 0.6:
                st.error("⚠️ Toxic")
            else:
                st.success("✅ Non-Toxic")

    # ------------------------------
    # MANUAL MODE
    # ------------------------------
    else:
        f1 = st.number_input("Molecular Weight", value=200.0)
        f2 = st.number_input("H-Bond Donors", value=1.0)
        f3 = st.number_input("H-Bond Acceptors", value=2.0)
        f4 = st.number_input("LogP", value=1.0)

        if st.button("Predict Manual"):
            x = torch.tensor([[f1, f2, f3, f4]], dtype=torch.float32)
            prob = model(x).item()

            st.write(f"Toxicity Risk: {prob*100:.2f}%")
            if prob > 0.7:
                st.error("⚠️ Toxic")
            elif prob > 0.4:
                st.warning("⚠️ Moderately Toxic")
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
