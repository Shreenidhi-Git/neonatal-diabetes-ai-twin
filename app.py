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

THRESHOLD = 0.70


# ============================================================
# EXPECTED 33 FEATURES
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
    f for f in required_files
    if not os.path.exists(f)
]

if missing_files:

    st.error("❌ Required AI Twin files are missing.")

    for f in missing_files:
        st.write(f"- `{f}`")

    st.info(
        "Upload the missing files to the same GitHub repository "
        "as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL
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

    with open(FEATURE_FILE, "rb") as f:
        feature_names = joblib.load(f)

    reference_values = None

    if os.path.exists(REFERENCE_FILE):

        with open(REFERENCE_FILE, "rb") as f:
            reference_values = joblib.load(f)

    return (
        model,
        scaler,
        list(feature_names),
        reference_values
    )


try:

    model, scaler, feature_names, reference_values = (
        load_ai_twin()
    )

except Exception as e:

    st.error("❌ Failed to load the AI Twin.")

    st.exception(e)

    st.stop()


# ============================================================
# CHECK FEATURES
# ============================================================

if len(feature_names) != 33:

    st.error(
        f"❌ The model expects 33 features, "
        f"but {len(feature_names)} were loaded."
    )

    st.stop()


if feature_names != EXPECTED_FEATURES:

    st.error(
        "❌ Feature order does not match the trained model."
    )

    st.stop()


# ============================================================
# CHECK SCALER
# ============================================================

if getattr(scaler, "n_features_in_", None) != 33:

    st.error(
        "❌ The scaler does not contain 33 features."
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

    # Fallback to scaler mean

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

def variant(label):

    return st.selectbox(
        label,
        [0, 1],
        format_func=lambda x:
            "Present" if x == 1 else "Absent"
    )


# ============================================================
# PATIENT-SPECIFIC EXPLANATION
# ============================================================

def calculate_contributions(patient_df):

    # Original prediction

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

    for i, feature in enumerate(feature_names):

        modified_df = patient_df.copy()

        reference = get_reference_value(
            feature
        )

        # Replace only this feature

        modified_df.iloc[0, i] = reference

        modified_scaled = scaler.transform(
            modified_df
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
                patient_df.iloc[0, i],

            "Reference Value":
                reference,

            "Contribution":
                contribution
        })

    result = pd.DataFrame(
        contributions
    )

    result["Absolute"] = (
        result["Contribution"]
        .abs()
    )

    result = result.sort_values(
        "Absolute",
        ascending=False
    )

    return result, original_probability


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧬 Neonatal Diabetes AI Twin"
)

st.subheader(
    "Early Neonatal Diabetes Risk Prediction"
)

st.write(
    """
    This research prototype estimates the probability of
    neonatal diabetes using maternal, fetal, genetic and
    gene-expression features.
    """
)

st.warning(
    """
    ⚠️ Research / educational prototype only.
    This system is not a clinical diagnostic tool and should
    not be used alone for diagnosis or treatment decisions.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🧬 AI Twin"
    )

    st.write(
        "Model: Neural Network"
    )

    st.write(
        "Features: 33"
    )

    st.write(
        "Decision threshold: 70%"
    )

    st.divider()

    st.write(
        """
        The Digital Twin uses:

        • Maternal data
        • Fetal data
        • Genetic variants
        • Gene-expression data
        """
    )


# ============================================================
# PATIENT FORM
# ============================================================

st.header(
    "👩‍⚕️ Enter Patient Information"
)

with st.form(
    "patient_form"
):

    # ========================================================
    # MATERNAL
    # ========================================================

    st.subheader(
        "👩 Maternal Information"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        maternal_age = st.number_input(
            "Maternal Age (years)",
            15.0,
            60.0,
            28.0,
            0.1
        )

        maternal_bmi = st.number_input(
            "Maternal BMI",
            10.0,
            70.0,
            24.0,
            0.1
        )

        systolic_bp = st.number_input(
            "Maternal Systolic BP",
            70.0,
            250.0,
            120.0,
            1.0
        )

    with c2:

        diastolic_bp = st.number_input(
            "Maternal Diastolic BP",
            40.0,
            150.0,
            80.0,
            1.0
        )

        fasting_glucose = st.number_input(
            "Maternal Fasting Glucose (mg/dL)",
            40.0,
            400.0,
            90.0,
            0.1
        )

        hba1c = st.number_input(
            "Maternal HbA1c (%)",
            3.0,
            20.0,
            5.5,
            0.1
        )

    with c3:

        gestational_age = st.number_input(
            "Gestational Age (weeks)",
            20.0,
            45.0,
            32.0,
            0.1
        )

        family_history = yes_no(
            "Family History of Diabetes"
        )

        previous_gdm = yes_no(
            "Previous Gestational Diabetes"
        )


    # ========================================================
    # FETAL
    # ========================================================

    st.divider()

    st.subheader(
        "👶 Fetal Information"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        fetal_hr = st.number_input(
            "Fetal Heart Rate (bpm)",
            50.0,
            220.0,
            140.0,
            1.0
        )

        fetal_movement = st.number_input(
            "Fetal Movement per hour",
            0.0,
            100.0,
            10.0,
            0.1
        )

    with c2:

        growth_percentile = st.number_input(
            "Ultrasound Growth Percentile",
            0.0,
            100.0,
            50.0,
            1.0
        )

        fetal_weight = st.number_input(
            "Estimated Fetal Weight (g)",
            200.0,
            6000.0,
            2000.0,
            10.0
        )

    with c3:

        consanguinity = yes_no(
            "Consanguinity"
        )

        autoimmune_history = yes_no(
            "Maternal Autoimmune History"
        )


    # ========================================================
    # GENETICS
    # ========================================================

    st.divider()

    st.subheader(
        "🧬 Genetic Information"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        kcnj11_variant = variant(
            "KCNJ11 Variant"
        )

        abcc8_variant = variant(
            "ABCC8 Variant"
        )

        ins_variant = variant(
            "INS Variant"
        )

    with c2:

        chr6q24 = variant(
            "Chr6q24 Abnormality"
        )

        gck_variant = variant(
            "GCK Variant"
        )

        hnf1b_variant = variant(
            "HNF1B Variant"
        )

    with c3:

        gata6_variant = variant(
            "GATA6 Variant"
        )

        glis3_variant = variant(
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


    if expression_available == (
        "Yes - enter measurements"
    ):

        st.info(
            "Enter the actual gene-expression measurements."
        )

        e1, e2, e3 = st.columns(3)

        with e1:

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

        with e2:

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

        with e3:

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

        # IMPORTANT:
        # Use reference values rather than zero.

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
            Gene-expression measurements were not entered.
            The AI Twin is using the saved reference values
            for these features.
            """
        )


    # ========================================================
    # PREDICT
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

    values = [

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


    patient_df = pd.DataFrame(
        [values],
        columns=EXPECTED_FEATURES
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if patient_df.shape != (1, 33):

        st.error(
            "❌ Exactly 33 features are required."
        )

        st.stop()


    if patient_df.isnull().any().any():

        st.error(
            "❌ Missing values detected."
        )

        st.stop()


    if diastolic_bp >= systolic_bp:

        st.error(
            "❌ Diastolic BP must be lower than "
            "systolic BP."
        )

        st.stop()


    # ========================================================
    # SCALE
    # ========================================================

    try:

        scaled = scaler.transform(
            patient_df
        )

    except Exception as e:

        st.error(
            "❌ Scaling error."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # PREDICT
    # ========================================================

    try:

        probability = float(
            model.predict(
                scaled,
                verbose=0
            )[0][0]
        )

    except Exception as e:

        st.error(
            "❌ Prediction error."
        )

        st.exception(e)

        st.stop()


    probability = np.clip(
        probability,
        0,
        1
    )

    percentage = probability * 100


    # ========================================================
    # RISK CATEGORY
    # ========================================================

    if probability >= THRESHOLD:

        category = "HIGH RISK"

    else:

        category = "LOWER RISK"


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header(
        "🧬 Digital Twin Prediction"
    )


    # ========================================================
    # BIG PROBABILITY
    # ========================================================

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:25px;
            border-radius:18px;
            border:1px solid rgba(128,128,128,0.35);
            margin-bottom:20px;
        ">

        <div style="
            font-size:18px;
            font-weight:600;
        ">
        NDM RISK PROBABILITY
        </div>

        <div style="
            font-size:64px;
            font-weight:800;
            margin:10px;
        ">
        {percentage:.2f}%
        </div>

        <div style="
            font-size:25px;
            font-weight:700;
        ">
        {category}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # RISK GRAPH
    # ========================================================

    st.subheader(
        "📊 Risk Visualization"
    )

    # Create 100-point risk graph

    chart_df = pd.DataFrame({
        "Risk": [percentage]
    })

    st.progress(
        float(probability)
    )

    g1, g2, g3 = st.columns(3)

    with g1:

        st.caption(
            "🟢 0–30%"
        )

    with g2:

        st.caption(
            "🟡 30–70%"
        )

    with g3:

        st.caption(
            "🔴 70–100%"
        )


    # ========================================================
    # THRESHOLD MESSAGE
    # ========================================================

    if probability >= THRESHOLD:

        st.error(
            f"""
            The predicted probability is {percentage:.2f}%,
            which is above the configured decision threshold
            of {THRESHOLD * 100:.0f}%.
            """
        )

    else:

        st.success(
            f"""
            The predicted probability is {percentage:.2f}%,
            which is below the configured decision threshold
            of {THRESHOLD * 100:.0f}%.
            """
        )


    # ========================================================
    # PATIENT-SPECIFIC EXPLANATION
    # ========================================================

    st.divider()

    st.header(
        "🔎 Why did the Digital Twin give this result?"
    )

    st.write(
        """
        These are estimated patient-specific contributions.
        They show which features moved the model's prediction
        upward or downward compared with their saved reference
        values.
        """
    )


    # ========================================================
    # CALCULATE CONTRIBUTIONS
    # ========================================================

    with st.spinner(
        "Analyzing the patient's features..."
    ):

        try:

            contribution_df, verified_probability = (
                calculate_contributions(
                    patient_df
                )
            )

        except Exception as e:

            st.error(
                "❌ Could not calculate the explanation."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # INCREASING RISK
    # ========================================================

    increasing = contribution_df[
        contribution_df["Contribution"] > 0
    ].head(5)


    # ========================================================
    # DECREASING RISK
    # ========================================================

    decreasing = contribution_df[
        contribution_df["Contribution"] < 0
    ].sort_values(
        "Contribution"
    ).head(5)


    left, right = st.columns(2)


    # ========================================================
    # INCREASING
    # ========================================================

    with left:

        st.subheader(
            "⬆️ Features increasing predicted risk"
        )

        if increasing.empty:

            st.info(
                "No significant positive contributions."
            )

        else:

            for _, row in increasing.iterrows():

                st.markdown(
                    f"""
                    **{row['Feature']}**

                    Patient value: `{row['Patient Value']}`

                    Reference value: `{row['Reference Value']}`

                    Contribution:
                    **+{row['Contribution']:.5f}**
                    """
                )

                st.divider()


    # ========================================================
    # DECREASING
    # ========================================================

    with right:

        st.subheader(
            "⬇️ Features decreasing predicted risk"
        )

        if decreasing.empty:

            st.info(
                "No significant negative contributions."
            )

        else:

            for _, row in decreasing.iterrows():

                st.markdown(
                    f"""
                    **{row['Feature']}**

                    Patient value: `{row['Patient Value']}`

                    Reference value: `{row['Reference Value']}`

                    Contribution:
                    **{row['Contribution']:.5f}**
                    """
                )

                st.divider()


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    st.divider()

    st.info(
        """
        🧬 The Digital Twin prediction is generated from
        the trained machine-learning model using the entered
        patient information.

        ⚠️ This is a research/educational prototype and
        is not a clinical diagnostic system.
        """
    )
