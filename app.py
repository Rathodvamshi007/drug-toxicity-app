import streamlit as st
import torch
import torch.nn as nn
import json
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Drug Toxicity App", page_icon="🧬")

# ==============================
# BACKGROUND STYLE
# ==============================
def set_bg():
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True
    )

set_bg()

# ==============================
# USER DATA FILE
# ==============================
USER_FILE = "users.json"

# Load users
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

# Save users
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

    if st.button("Go to Login"):
        st.session_state.page = "login"

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
        else:
            st.error("Invalid Credentials ❌")

    if st.button("Create Account"):
        st.session_state.page = "signup"

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

model = SimpleModel()
model.load_state_dict(torch.load("simple_model.pth", map_location="cpu"))
model.eval()

# ==============================
# MAIN APP
# ==============================
def main_app():
    st.title("🧬 Drug Toxicity Predictor")

    st.subheader("Enter Chemical Properties")

    f1 = st.number_input("Molecular Weight", value=200.0)
    f2 = st.number_input("H-Bond Donors", value=1.0)
    f3 = st.number_input("H-Bond Acceptors", value=2.0)
    f4 = st.number_input("LogP (Lipophilicity)", value=1.0)

    features = [f1, f2, f3, f4]

    if st.button("Predict"):
        x = torch.tensor([features], dtype=torch.float32)

        with torch.no_grad():
            prob = model(x).item()

        st.subheader("Result")
        st.write(f"🔢 Probability: {prob:.4f}")
        st.write(f"🧪 Toxicity Risk: {prob*100:.2f}%")

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
