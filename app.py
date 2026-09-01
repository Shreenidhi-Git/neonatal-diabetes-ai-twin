import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# ============================================================
# PAGE SETTINGS
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
REFERENCE_FILE = "ndm_reference_values.pkl"

DECISION_THRESHOLD = 0.70


# ============================================================
# 33 FEATURES
# ============================================================

EXPECTED_FEATURES = [

    "Maternal_Age_years",
    "Maternal_BMI",
    "Maternal_Systolic_BP",
    "Maternal_Diastolic_BP",
    "Maternal_Fasting_Glucose_mg_dL",
    "Maternal_HbA1c_percent",
    "Gestational_Age_weeks",
    "Fetal_Heart_Rate_bpm",
    "Fetal_Movement_per_hour",
    "Ultrasound_Growth_Percentile",
    "Estimated_Fetal_Weight_g",

    "Family_History_Diabetes",
    "Consanguinity",
    "Previous_Gestational_Diabetes",
    "Maternal_Autoimmune_History",

    "KCNJ11_Variant",
    "ABCC8_Variant",
    "INS_Variant",
    "Chr6q24_Abnormality",
    "GCK_Variant",
    "HNF1B_Variant",
    "GATA6_Variant",
    "GLIS3_Variant",

    "INS_expr",
    "PDX1_expr",
    "NKX6_1_expr",
    "MAFA_expr",
    "GCK_expr",
    "SLC2A2_expr",
    "ABCC8_expr",
    "KCNJ11_expr",
    "NEUROD1_expr",
    "HNF1B_expr"
]


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_FILE
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error("❌ Required AI Twin files are missing.")

    st.write("Please make sure these files are in your repository:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.stop()


# ============================================================
# LOAD MODEL AND SUPPORT FILES
# ============================================================

@st.cache_resource
def load_ai_twin():

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    with open(FEATURE_FILE, "rb") as file:
        feature_names = joblib.load(file)

    reference_values = None

    if os.path.exists(REFERENCE_FILE):

        with open(REFERENCE_FILE, "rb") as file:
            reference_values = joblib.load(file)

    return (
        model,
        scaler,
        list(feature_names),
        reference_values
    )


# ============================================================
# LOAD
# ============================================================

try:

    model, scaler, feature_names, reference_values = (
        load_ai_twin()
    )

except Exception as error:

    st.error("❌ Could not load the AI Twin.")

    st.exception(error)

    st.stop()


# ============================================================
# VALIDATE FEATURES
# ============================================================

if len(feature_names) != 33:

    st.error(
        f"""
        ❌ Feature count error.

        The model requires 33 features.

        Loaded features: {len(feature_names)}
        """
    )

    st.stop()


if feature_names != EXPECTED_FEATURES:

    st.error(
        """
        ❌ Feature order mismatch.

        The feature order saved with the model does not
        match the expected 33-feature order.
        """
    )

    st.stop()


# ============================================================
# VALIDATE SCALER
# ============================================================

if getattr(scaler, "n_features_in_", None) != 33:

    st.error(
        """
        ❌ Scaler error.

        The saved scaler does not contain 33 features.
        """
    )

    st.stop()


# ============================================================
# GET REFERENCE VALUE
# ============================================================

def get_reference_value(feature):

    try:

        if isinstance(reference_values, dict):

            value = reference_values.get(
                feature,
                None
            )

            if value is not None:

                value = float(value)

                if np.isfinite(value):

                    return value

    except Exception:

        pass

    index = feature_names.index(feature)

    return float(
        scaler.mean_[index]
    )


# ============================================================
# YES / NO INPUT
# ============================================================

def yes_no(label):

    return st.selectbox(
        label,
        [0, 1],
        format_func=lambda x:
        "Yes" if x == 1 else "No"
    )


# ============================================================
# GENETIC VARIANT INPUT
# ============================================================

def variant_input(label):

    return st.selectbox(
        label,
        [0, 1],
        format_func=lambda x:
        "Present" if x == 1 else "Absent"
    )


# ============================================================
# PATIENT-SPECIFIC EXPLANATION
# ============================================================

def calculate_patient_contributions(patient_df):

    original_scaled = scaler.transform(
        patient_df
    )

    original_probability = float(
        model.predict(
            original_scaled,
            verbose=0
        )[0][0]
    )

    contributions = []

    for index, feature in enumerate(feature_names):

        modified_patient = patient_df.copy()

        reference_value = get_reference_value(
            feature
        )

        modified_patient.iloc[
            0,
            index
        ] = reference_value

        modified_scaled = scaler.transform(
            modified_patient
        )

        modified_probability = float(
            model.predict(
                modified_scaled,
                verbose=0
            )[0][0]
        )

        contribution = (
            original_probability -
            modified_probability
        )

        contributions.append({

            "Feature": feature,

            "Patient Value":
                patient_df.iloc[0, index],

            "Reference Value":
                reference_value,

            "Contribution":
                contribution,

            "Absolute":
                abs(contribution)
        })

    result = pd.DataFrame(
        contributions
    )

    result = result.sort_values(
        "Absolute",
        ascending=False
    )

    return result


# ============================================================
# TITLE
# ============================================================

st.title(
    "🧬 Neonatal Diabetes AI Twin"
)

st.subheader(
    "Early Neonatal Diabetes Risk Prediction"
)

st.write(
    """
    This research prototype estimates the model-predicted
    risk of neonatal diabetes using maternal, fetal,
    genetic and gene-expression features.
    """
)


# ============================================================
# IMPORTANT NOTE
# ============================================================

st.warning(
    """
    ⚠️ **Research / Educational Prototype**

    This AI Twin is intended for research and educational
    demonstration only. It is **not a clinical diagnostic
    tool** and should not be used alone for medical
    diagnosis or treatment decisions.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧬 About the AI Twin")

    st.write(
        """
        **Model:** Neural Network

        **Input features:** 33

        **Decision threshold:** 70%

        **Purpose:** Early risk prediction research
        """
    )

    st.divider()

    st.write(
        """
        The model uses:

        • Maternal information

        • Fetal information

        • Genetic variants

        • Gene-expression features
        """
    )


# ============================================================
# PATIENT INPUT
# ============================================================

st.header(
    "👩‍⚕️ Patient Information"
)

with st.form(
    "patient_form"
):

    # ========================================================
    # MATERNAL INFORMATION
    # ========================================================

    st.subheader(
        "👩 Maternal Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        maternal_age = st.number_input(
            "Maternal Age (years)",
            min_value=15.0,
            max_value=60.0,
            value=28.0,
            step=0.1
        )

        maternal_bmi = st.number_input(
            "Maternal BMI",
            min_value=10.0,
            max_value=70.0,
            value=24.0,
            step=0.1
        )

        systolic_bp = st.number_input(
            "Maternal Systolic BP",
            min_value=70.0,
            max_value=250.0,
            value=120.0,
            step=1.0
        )

    with col2:

        diastolic_bp = st.number_input(
            "Maternal Diastolic BP",
            min_value=40.0,
            max_value=150.0,
            value=80.0,
            step=1.0
        )

        fasting_glucose = st.number_input(
            "Maternal Fasting Glucose (mg/dL)",
            min_value=40.0,
            max_value=400.0,
            value=90.0,
            step=0.1
        )

        hba1c = st.number_input(
            "Maternal HbA1c (%)",
            min_value=3.0,
            max_value=20.0,
            value=5.5,
            step=0.1
        )

    with col3:

        gestational_age = st.number_input(
            "Gestational Age (weeks)",
            min_value=20.0,
            max_value=45.0,
            value=32.0,
            step=0.1
        )

        family_history = yes_no(
            "Family History of Diabetes"
        )

        previous_gdm = yes_no(
            "Previous Gestational Diabetes"
        )


    # ========================================================
    # FETAL INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "👶 Fetal Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        fetal_heart_rate = st.number_input(
            "Fetal Heart Rate (bpm)",
            min_value=50.0,
            max_value=220.0,
            value=140.0,
            step=1.0
        )

        fetal_movement = st.number_input(
            "Fetal Movement per hour",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        )

    with col2:

        ultrasound_growth = st.number_input(
            "Ultrasound Growth Percentile",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

        estimated_weight = st.number_input(
            "Estimated Fetal Weight (g)",
            min_value=200.0,
            max_value=6000.0,
            value=2000.0,
            step=10.0
        )

    with col3:

        consanguinity = yes_no(
            "Consanguinity"
        )

        autoimmune_history = yes_no(
            "Maternal Autoimmune History"
        )


    # ========================================================
    # GENETIC INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "🧬 Genetic Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        kcnj11_variant = variant_input(
            "KCNJ11 Variant"
        )

        abcc8_variant = variant_input(
            "ABCC8 Variant"
        )

        ins_variant = variant_input(
            "INS Variant"
        )

    with col2:

        chr6q24 = variant_input(
            "Chr6q24 Abnormality"
        )

        gck_variant = variant_input(
            "GCK Variant"
        )

        hnf1b_variant = variant_input(
            "HNF1B Variant"
        )

    with col3:

        gata6_variant = variant_input(
            "GATA6 Variant"
        )

        glis3_variant = variant_input(
            "GLIS3 Variant"
        )


    # ========================================================
    # GENE EXPRESSION
    # ========================================================

    st.divider()

    st.subheader(
        "🧪 Gene Expression"
    )

    expression_available = st.radio(
        "Are gene-expression measurements available?",
        [
            "No - use reference values",
            "Yes - enter measurements"
        ],
        horizontal=True
    )


    # ========================================================
    # GENE EXPRESSION INPUT
    # ========================================================

    if expression_available == (
        "Yes - enter measurements"
    ):

        st.caption(
            "Enter the available gene-expression values."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            ins_expr = st.number_input(
                "INS expression",
                value=get_reference_value(
                    "INS_expr"
                ),
                step=0.01
            )

            pdx1_expr = st.number_input(
                "PDX1 expression",
                value=get_reference_value(
                    "PDX1_expr"
                ),
                step=0.01
            )

            nkx6_1_expr = st.number_input(
                "NKX6_1 expression",
                value=get_reference_value(
                    "NKX6_1_expr"
                ),
                step=0.01
            )

            mafa_expr = st.number_input(
                "MAFA expression",
                value=get_reference_value(
                    "MAFA_expr"
                ),
                step=0.01
            )

        with col2:

            gck_expr = st.number_input(
                "GCK expression",
                value=get_reference_value(
                    "GCK_expr"
                ),
                step=0.01
            )

            slc2a2_expr = st.number_input(
                "SLC2A2 expression",
                value=get_reference_value(
                    "SLC2A2_expr"
                ),
                step=0.01
            )

            abcc8_expr = st.number_input(
                "ABCC8 expression",
                value=get_reference_value(
                    "ABCC8_expr"
                ),
                step=0.01
            )

            kcnj11_expr = st.number_input(
                "KCNJ11 expression",
                value=get_reference_value(
                    "KCNJ11_expr"
                ),
                step=0.01
            )

        with col3:

            neurod1_expr = st.number_input(
                "NEUROD1 expression",
                value=get_reference_value(
                    "NEUROD1_expr"
                ),
                step=0.01
            )

            hnf1b_expr = st.number_input(
                "HNF1B expression",
                value=get_reference_value(
                    "HNF1B_expr"
                ),
                step=0.01
            )

    else:

        ins_expr = get_reference_value(
            "INS_expr"
        )

        pdx1_expr = get_reference_value(
            "PDX1_expr"
        )

        nkx6_1_expr = get_reference_value(
            "NKX6_1_expr"
        )

        mafa_expr = get_reference_value(
            "MAFA_expr"
        )

        gck_expr = get_reference_value(
            "GCK_expr"
        )

        slc2a2_expr = get_reference_value(
            "SLC2A2_expr"
        )

        abcc8_expr = get_reference_value(
            "ABCC8_expr"
        )

        kcnj11_expr = get_reference_value(
            "KCNJ11_expr"
        )

        neurod1_expr = get_reference_value(
            "NEUROD1_expr"
        )

        hnf1b_expr = get_reference_value(
            "HNF1B_expr"
        )

        st.info(
            """
            Gene-expression values were not entered.
            The AI Twin will use the saved reference values.
            """
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.divider()

    predict = st.form_submit_button(
        "🔍 Predict NDM Risk",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # ========================================================
    # CREATE PATIENT DATA
    # ========================================================

    patient_values = [

        maternal_age,
        maternal_bmi,
        systolic_bp,
        diastolic_bp,
        fasting_glucose,
        hba1c,
        gestational_age,
        fetal_heart_rate,
        fetal_movement,
        ultrasound_growth,
        estimated_weight,

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


    patient_df = pd.DataFrame(
        [patient_values],
        columns=EXPECTED_FEATURES
    )


    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if patient_df.shape != (1, 33):

        st.error(
            "❌ The AI Twin requires exactly 33 features."
        )

        st.stop()


    if patient_df.isnull().any().any():

        st.error(
            "❌ Please fill in all required values."
        )

        st.stop()


    if diastolic_bp >= systolic_bp:

        st.error(
            """
            ❌ Maternal diastolic blood pressure should
            be lower than systolic blood pressure.
            """
        )

        st.stop()


    # ========================================================
    # SCALE INPUT
    # ========================================================

    try:

        scaled_patient = scaler.transform(
            patient_df
        )

    except Exception as error:

        st.error(
            "❌ Error while processing the patient data."
        )

        st.exception(error)

        st.stop()


    # ========================================================
    # PREDICT
    # ========================================================

    try:

        probability = float(
            model.predict(
                scaled_patient,
                verbose=0
            )[0][0]
        )

    except Exception as error:

        st.error(
            "❌ Error while generating the prediction."
        )

        st.exception(error)

        st.stop()


    probability = np.clip(
        probability,
        0,
        1
    )

    percentage = probability * 100


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if probability >= DECISION_THRESHOLD:

        risk_category = "HIGH RISK"

    else:

        risk_category = "LOWER RISK"


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header(
        "🧬 Digital Twin Prediction"
    )


    # ========================================================
    # SIMPLE RESULT CARDS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Estimated NDM Risk",
            f"{percentage:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Classification",
            risk_category
        )


    # ========================================================
    # OVERALL GRAPH
    # ========================================================

    st.subheader(
        "📊 Overall Prediction"
    )

    st.write(
        f"""
        The model estimates an NDM risk of
        **{percentage:.2f}%** for the entered patient data.
        """
    )

    st.progress(
        float(probability)
    )

    st.caption(
        "0% ───────────────────────────── 100%"
    )

    graph_col1, graph_col2, graph_col3 = st.columns(3)

    with graph_col1:

        st.write("🟢 **LOWER RISK**")

        st.caption("0% – 30%")

    with graph_col2:

        st.write("🟡 **INTERMEDIATE**")

        st.caption("30% – 70%")

    with graph_col3:

        st.write("🔴 **HIGHER RISK**")

        st.caption("70% – 100%")


    # ========================================================
    # RISK NOTE
    # ========================================================

    st.subheader(
        "ℹ️ What does this result mean?"
    )

    if probability >= DECISION_THRESHOLD:

        st.error(
            f"""
            The model estimated a risk of **{percentage:.2f}%**.

            This is above the configured **70% decision
            threshold**, so the model classifies this input
            as **HIGH RISK**.

            This does **not** mean that the baby has
            neonatal diabetes. It means that the entered
            feature combination produced a high-risk
            prediction from this trained model.
            """
        )

    else:

        st.success(
            f"""
            The model estimated a risk of **{percentage:.2f}%**.

            This is below the configured **70% decision
            threshold**, so the model classifies this input
            as **LOWER RISK**.

            This does **not** guarantee that neonatal
            diabetes will not occur. It means that the
            entered feature combination produced a
            lower-risk prediction from this trained model.
            """
        )


    # ========================================================
    # WHY THIS RESULT?
    # ========================================================

    st.divider()

    st.header(
        "🔎 Why did the Digital Twin give this result?"
    )

    st.write(
        """
        The AI Twin looks at the combination of the patient's
        maternal, fetal, genetic and gene-expression values.

        To explain this particular prediction, each feature
        is compared with its reference value.

        **⬆️ Positive contribution:** the feature moved the
        model prediction toward higher risk.

        **⬇️ Negative contribution:** the feature moved the
        model prediction toward lower risk.

        These are model-based explanations. They do not mean
        that a feature medically caused the disease.
        """
    )


    # ========================================================
    # CALCULATE EXPLANATIONS
    # ========================================================

    with st.spinner(
        "🔎 Analyzing the factors behind this prediction..."
    ):

        try:

            contribution_df = (
                calculate_patient_contributions(
                    patient_df
                )
            )

        except Exception as error:

            st.error(
                "❌ Could not generate the explanation."
            )

            st.exception(error)

            st.stop()


    # ========================================================
    # POSITIVE FEATURES
    # ========================================================

    increasing = contribution_df[
        contribution_df["Contribution"] > 0
    ].sort_values(
        "Contribution",
        ascending=False
    ).head(5)


    # ========================================================
    # NEGATIVE FEATURES
    # ========================================================

    decreasing = contribution_df[
        contribution_df["Contribution"] < 0
    ].sort_values(
        "Contribution",
        ascending=True
    ).head(5)


    # ========================================================
    # SHOW EXPLANATION
    # ========================================================

    left, right = st.columns(2)


    # ========================================================
    # INCREASING
    # ========================================================

    with left:

        st.subheader(
            "⬆️ Factors increasing the predicted risk"
        )

        if increasing.empty:

            st.write(
                "No strong positive contributions were detected."
            )

        else:

            for _, row in increasing.iterrows():

                st.markdown(
                    f"""
                    **{row["Feature"]}**

                    Patient value: **{row["Patient Value"]}**

                    Reference value: **{row["Reference Value"]:.4f}**

                    Estimated contribution:
                    **+{row["Contribution"]:.5f}**
                    """
                )

                st.divider()


    # ========================================================
    # DECREASING
    # ========================================================

    with right:

        st.subheader(
            "⬇️ Factors decreasing the predicted risk"
        )

        if decreasing.empty:

            st.write(
                "No strong negative contributions were detected."
            )

        else:

            for _, row in decreasing.iterrows():

                st.markdown(
                    f"""
                    **{row["Feature"]}**

                    Patient value: **{row["Patient Value"]}**

                    Reference value: **{row["Reference Value"]:.4f}**

                    Estimated contribution:
                    **{row["Contribution"]:.5f}**
                    """
                )

                st.divider()


    # ========================================================
    # FINAL NOTE
    # ========================================================

    st.info(
        """
        ### 🧬 Important Note

        The risk percentage and feature contributions are
        generated by the trained machine-learning model.

        They are intended to help explain the model's output,
        not to replace professional medical assessment.

        **This Digital Twin is a research and educational
        prototype, not a clinical diagnostic system.**
        """
        )
