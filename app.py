import streamlit as st
import torch
import torch.nn as nn
import json
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Drug Toxicity Predictor", page_icon="🧬")

# ==============================
# BACKGROUND STYLE
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
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# USER FILE
# ==============================
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# ==============================
# SESSION STATE
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

# ==============================
# SIGNUP PAGE
# ==============================
def signup():
    st.title("📝 Signup")

    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")

    if st.button("Signup"):
        users = load_users()

        if username in users:
            st.error("User already exists ❌")
        else:
            users[username] = password
            save_users(users)
            st.success("Account created successfully ✅")
            st.session_state.page = "login"
            st.rerun()

    if st.button("Go to Login"):
        st.session_state.page = "login"
        st.rerun()

# ==============================
# LOGIN PAGE
# ==============================
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()

        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")

    if st.button("Create Account"):
        st.session_state.page = "signup"
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
        x = self.fc2(x)
        return self.sigmoid(x)

# Safe loading
try:
    model = SimpleModel()
    model.load_state_dict(torch.load("simple_model.pth", map_location="cpu"))
    model.eval()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ==============================
# MAIN APP
# ==============================
def main_app():
    st.title("🧬 Drug Toxicity Predictor")

    st.markdown("### 🧪 Enter Chemical Details")

    # Chemical Name
    chem_name = st.text_input("🔎 Chemical Name (Optional)", placeholder="e.g., Aspirin")

    if chem_name:
        st.info(f"You are analyzing: **{chem_name}**")

    st.markdown("---")

    # Inputs
    st.subheader("📊 Chemical Properties")

    f1 = st.number_input(
        "Molecular Weight",
        value=200.0,
        help="Total molecular mass (g/mol). Available on PubChem."
    )

    f2 = st.number_input(
        "H-Bond Donors",
        value=1.0,
        help="Groups like -OH, -NH that donate hydrogen."
    )

    f3 = st.number_input(
        "H-Bond Acceptors",
        value=2.0,
        help="Atoms like O, N that accept hydrogen bonds."
    )

    f4 = st.number_input(
        "LogP (Lipophilicity)",
        value=1.0,
        help="Measures fat vs water solubility."
    )

    # Help Section
    with st.expander("📘 How to find these values?"):
        st.markdown("""
        🔗 PubChem: https://pubchem.ncbi.nlm.nih.gov  
        🔗 ChemSpider: http://www.chemspider.com  

        Steps:
        1. Search chemical name  
        2. Open compound page  
        3. Find:
           - Molecular Weight  
           - H-Bond Donors/Acceptors  
           - LogP  

        Example:
        Aspirin → MW: 180.16, Donors: 1, Acceptors: 4
        """)

    features = [f1, f2, f3, f4]

    st.markdown("---")

    # Prediction
    if st.button("Predict"):
        x = torch.tensor([features], dtype=torch.float32)

        with torch.no_grad():
            prob = model(x).item()

        st.subheader("📊 Result")

        if chem_name:
            st.write(f"🧪 Chemical: **{chem_name}**")

        st.write(f"🔢 Probability: {prob:.4f}")
        st.write(f"⚠️ Toxicity Risk: {prob*100:.2f}%")

        if prob > 0.6:
            st.error("⚠️ Toxic Compound")
        else:
            st.success("✅ Non-Toxic Compound")

    st.markdown("---")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

# ==============================
# APP FLOW
# ==============================
if st.session_state.logged_in:
    main_app()
else:
    if st.session_state.page == "login":
        login()
    else:
        signup()
