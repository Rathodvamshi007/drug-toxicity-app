import streamlit as st
import torch
import torch.nn as nn

# ==============================
# MODEL DEFINITION
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

# ==============================
# LOAD MODEL
# ==============================
model = SimpleModel()
model.load_state_dict(torch.load("simple_model.pth", map_location="cpu"))
model.eval()

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Drug Toxicity Predictor", page_icon="🧬")

st.title("🧬 Drug Toxicity Predictor")
st.write("Enter chemical properties to predict toxicity.")

# ==============================
# INPUT FEATURES
# ==============================
f1 = st.number_input("Molecular Weight", min_value=0.0, value=200.0)
f2 = st.number_input("H-Bond Donors", min_value=0.0, value=1.0)
f3 = st.number_input("H-Bond Acceptors", min_value=0.0, value=2.0)
f4 = st.number_input("LogP (Lipophilicity)", value=1.0)

features = [f1, f2, f3, f4]

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):
    x = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        prob = model(x).item()

    st.subheader("Result")
    st.write(f"🔢 Raw Probability: {prob:.4f}")
    st.write(f"🧪 Toxicity Risk: {prob*100:.2f}%")

    # Threshold
    threshold = 0.6

    if prob > threshold:
        st.error("⚠️ Toxic Compound Detected")
    else:
        st.success("✅ Non-Toxic Compound")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("ML-based Drug Toxicity Prediction App")
