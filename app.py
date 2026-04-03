def main_app():
    st.title("🧬 Drug Toxicity Predictor")

    st.markdown("### 🧪 Enter Chemical Details")

    # ------------------------------
    # Chemical Name Input
    # ------------------------------
    chem_name = st.text_input("🔎 Chemical Name (Optional)", placeholder="e.g., Aspirin, Paracetamol")

    if chem_name:
        st.info(f"You are analyzing: **{chem_name}**")

    st.markdown("---")

    # ------------------------------
    # Feature Inputs with Help
    # ------------------------------
    st.subheader("📊 Chemical Properties")

    f1 = st.number_input(
        "Molecular Weight",
        value=200.0,
        help="Total mass of the molecule (g/mol). You can find this on PubChem or ChemSpider."
    )

    f2 = st.number_input(
        "H-Bond Donors",
        value=1.0,
        help="Number of hydrogen atoms that can donate a hydrogen bond (e.g., -OH, -NH groups)."
    )

    f3 = st.number_input(
        "H-Bond Acceptors",
        value=2.0,
        help="Atoms that can accept hydrogen bonds (e.g., Oxygen, Nitrogen)."
    )

    f4 = st.number_input(
        "LogP (Lipophilicity)",
        value=1.0,
        help="Measures how soluble the compound is in fat vs water. Higher = more lipophilic."
    )

    # ------------------------------
    # Help Section
    # ------------------------------
    with st.expander("📘 How to find these values?"):
        st.markdown("""
        You can get chemical properties from:

        🔗 **PubChem**: https://pubchem.ncbi.nlm.nih.gov  
        🔗 **ChemSpider**: http://www.chemspider.com  

        👉 Steps:
        1. Search your chemical name (e.g., Aspirin)
        2. Open the compound page
        3. Look for:
           - Molecular Weight
           - Hydrogen Bond Donors/Acceptors
           - LogP

        💡 Example:
        - Aspirin → MW ≈ 180.16
        - H-Bond Donors = 1
        - H-Bond Acceptors = 4
        """)

    features = [f1, f2, f3, f4]

    st.markdown("---")

    # ------------------------------
    # Prediction
    # ------------------------------
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
