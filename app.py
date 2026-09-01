import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Neonatal Diabetes AI Twin",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# FILE NAMES
# ============================================================

MODEL_FILE = "ndm_clean_neural_network.keras"
SCALER_FILE = "ndm_clean_33_feature_scaler.pkl"
FEATURE_FILE = "ndm_feature_names.pkl"
IMPORTANCE_FILE = "NDM_feature_importance.xlsx"

# ============================================================
# LOAD MODEL AND SUPPORTING FILES
# ============================================================

@st.cache_resource
def load_model_and_scaler():

    model = tf.keras.models.load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    with open(FEATURE_FILE, "rb") as f:
        feature_names = joblib.load(f)

    return model, scaler, feature_names


model, scaler, feature_names = load_model_and_scaler()

# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_feature_importance():

    if os.path.exists(IMPORTANCE_FILE):
        df = pd.read_excel(IMPORTANCE_FILE)
        return df

    return None


importance_df = load_feature_importance()

# ============================================================
# TITLE
# ============================================================

st.title("🧬 Neonatal Diabetes AI Twin")

st.markdown(
    """
    ### Early Neonatal Diabetes Risk Prediction

    This research prototype estimates the probability of neonatal
    diabetes using maternal, fetal, genetic and gene-expression features.
    """
)

st.warning(
    "⚠️ Research / educational prototype only. "
    "This system is not a clinical diagnostic tool."
)

st.divider()

# ============================================================
# PATIENT INPUT FORM
# ============================================================

st.header("👩‍⚕️ Patient Information")

with st.form("patient_form"):

    st.subheader("Maternal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        maternal_age = st.number_input(
            "Maternal Age (years)",
            min_value=15.0,
            max_value=60.0,
            value=28.0
        )

        maternal_bmi = st.number_input(
            "Maternal BMI",
            min_value=10.0,
            max_value=60.0,
            value=24.0
        )

        systolic_bp = st.number_input(
            "Maternal Systolic BP",
            min_value=70.0,
            max_value=250.0,
            value=120.0
        )

    with col2:
        diastolic_bp = st.number_input(
            "Maternal Diastolic BP",
            min_value=40.0,
            max_value=150.0,
            value=80.0
        )

        fasting_glucose = st.number_input(
            "Maternal Fasting Glucose (mg/dL)",
            min_value=40.0,
            max_value=400.0,
            value=90.0
        )

        hba1c = st.number_input(
            "Maternal HbA1c (%)",
            min_value=3.0,
            max_value=20.0,
            value=5.5
        )

    with col3:
        gestational_age = st.number_input(
            "Gestational Age (weeks)",
            min_value=20.0,
            max_value=45.0,
            value=32.0
        )

        family_history = st.selectbox(
            "Family History of Diabetes",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        previous_gdm = st.selectbox(
            "Previous Gestational Diabetes",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    st.subheader("Fetal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        fetal_hr = st.number_input(
            "Fetal Heart Rate (bpm)",
            min_value=50.0,
            max_value=220.0,
            value=140.0
        )

        fetal_movement = st.number_input(
            "Fetal Movement per Hour",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )

    with col2:
        growth_percentile = st.number_input(
            "Ultrasound Growth Percentile",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

        fetal_weight = st.number_input(
            "Estimated Fetal Weight (g)",
            min_value=200.0,
            max_value=6000.0,
            value=2000.0
        )

    with col3:
        consanguinity = st.selectbox(
            "Consanguinity",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        autoimmune_history = st.selectbox(
            "Maternal Autoimmune History",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    st.subheader("Genetic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        kcnj11_variant = st.selectbox(
            "KCNJ11 Variant",
            [0, 1],
            format_func=lambda x: "Present" if x == 1 else "Absent"
        )

        abcc8_variant = st.selectbox(
            "ABCC8 Variant",
            [0, 1],
            format_func=lambda x: "Present" if x == 1 else "Absent"
        )

        ins_variant = st.selectbox(
            "INS Variant",
            [0, 1],
            format_func=lambda x: "Present" if x == 1 else "Absent"
        )

    with col2:
        chr6q24 = st.selectbox(
            "Chr6q24 Abnormality",
            [0, 1],
            format_func=lambda x: "Present" if x == 1 else "Absent"
        )

        gck_variant = st.selectbox(
            "GCK Variant",
            [0, 1],
            format_func=lambda x: "Present" if x == 1 else "Absent"
        )

        hnf1b_variant = st.selectbox(
            "HNF1B Variant",
            [0, 1],
            format_func=lambda x: "Present" if x == 1 else "Absent"
        )

    with col3:
        gata6_variant = st.selectbox(
            "GATA6 Variant",
            [0, 1],
            format_func=lambda x: "Present" if x == 1 else "Absent"
        )

        glis3_variant = st.selectbox(
            "GLIS3 Variant",
            [0, 1],
            format_func=lambda x: "Present" if x == 1 else "Absent"
        )

    st.subheader("Gene Expression Values")

    col1, col2, col3 = st.columns(3)

    with col1:
        ins_expr = st.number_input("INS expression", value=0.0)
        pdx1_expr = st.number_input("PDX1 expression", value=0.0)
        nkx6_1_expr = st.number_input("NKX6_1 expression", value=0.0)
        mafa_expr = st.number_input("MAFA expression", value=0.0)

    with col2:
        gck_expr = st.number_input("GCK expression", value=0.0)
        slc2a2_expr = st.number_input("SLC2A2 expression", value=0.0)
        abcc8_expr = st.number_input("ABCC8 expression", value=0.0)
        kcnj11_expr = st.number_input("KCNJ11 expression", value=0.0)

    with col3:
        neurod1_expr = st.number_input("NEUROD1 expression", value=0.0)
        hnf1b_expr = st.number_input("HNF1B expression", value=0.0)

    st.divider()

    predict_button = st.form_submit_button(
        "🔍 Predict NDM Risk",
        use_container_width=True
    )

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create input in EXACT model feature order
    # --------------------------------------------------------

    input_data = [
        maternal_age,
        maternal_bmi,
        systolic_bp,
        diastolic_bp,
        fasting_glucose,
        hba1c,
        gestational_age,
        fetal_hr,
        fetal_movement,
        growth_percentile,
        fetal_weight,
        family_history,
        consanguinity,
        previous_gdm,
        autoimmune_history,
        kcnj11_variant,
        abcc8_variant,
        ins_variant,
        chr6q24,
        gck_variant,
        hnf1b_variant,
        gata6_variant,
        glis3_variant,
        ins_expr,
        pdx1_expr,
        nkx6_1_expr,
        mafa_expr,
        gck_expr,
        slc2a2_expr,
        abcc8_expr,
        kcnj11_expr,
        neurod1_expr,
        hnf1b_expr
    ]

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    scaled_input = scaler.transform(input_df)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    probability = float(
        model.predict(scaled_input, verbose=0)[0][0]
    )

    risk_percentage = probability * 100

    # Your selected threshold
    threshold = 0.70

    if probability >= threshold:
        risk_category = "HIGH RISK"
    else:
        risk_category = "LOWER RISK"

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.header("📊 Digital Twin Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "NDM Probability",
            f"{probability:.4f}"
        )

    with col2:
        st.metric(
            "Risk Percentage",
            f"{risk_percentage:.2f}%"
        )

    with col3:
        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )

    if probability >= threshold:
        st.error(
            f"🔴 {risk_category}"
        )
    else:
        st.success(
            f"🟢 {risk_category}"
        )

    # ========================================================
    # PATIENT-SPECIFIC EXPLANATION
    # ========================================================

    st.subheader("🔎 Why did the Digital Twin give this result?")

    if importance_df is not None:

        # Detect feature and importance columns
        feature_col = None
        importance_col = None

        for col in importance_df.columns:

            if str(col).lower() == "feature":
                feature_col = col

            if str(col).lower() == "importance":
                importance_col = col

        if feature_col and importance_col:

            explanation_df = importance_df.copy()

            explanation_df["Patient Value"] = [
                input_df.iloc[0].get(feature, np.nan)
                for feature in explanation_df[feature_col]
            ]

            # Focus on positive importance
            explanation_df = explanation_df[
                explanation_df[importance_col] > 0
            ]

            explanation_df = explanation_df.sort_values(
                importance_col,
                ascending=False
            )

            st.write(
                "The following features had the strongest positive "
                "contribution in the trained model:"
            )

            for _, row in explanation_df.head(10).iterrows():

                feature = row[feature_col]
                importance = row[importance_col]
                patient_value = row["Patient Value"]

                st.write(
                    f"**{feature}** — "
                    f"Patient value: `{patient_value}` | "
                    f"Model importance: `{importance:.6f}`"
                )

    else:

        st.info(
            "Feature importance file was not found."
        )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.divider()

    st.subheader("ℹ️ Model Information")

    st.write(
        f"Number of model features: **{len(feature_names)}**"
    )

    st.write(
        "Model type: **Neural Network**"
    )

    st.write(
        "Prediction threshold: **0.70**"
    )

    st.caption(
        "This application is a research/educational prototype "
        "and should not be used for clinical diagnosis or treatment decisions."
  )
