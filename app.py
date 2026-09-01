import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


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
REFERENCE_FILE = "ndm_reference_values.pkl"

DECISION_THRESHOLD = 0.70


# ============================================================
# 33 FEATURES USED BY THE TRAINED MODEL
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
# CHECK REQUIRED FILES
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

    st.write("Please make sure these files are uploaded to GitHub:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.stop()


# ============================================================
# LOAD MODEL, SCALER AND FEATURES
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
# LOAD FILES
# ============================================================

try:

    model, scaler, feature_names, reference_values = (
        load_ai_twin()
    )

except Exception as error:

    st.error("❌ Unable to load the AI Twin model.")

    st.exception(error)

    st.stop()


# ============================================================
# VALIDATE FEATURE COUNT
# ============================================================

if len(feature_names) != 33:

    st.error(
        f"""
        ❌ Feature count mismatch.

        Expected: 33

        Loaded: {len(feature_names)}
        """
    )

    st.stop()


# ============================================================
# VALIDATE FEATURE ORDER
# ============================================================

if feature_names != EXPECTED_FEATURES:

    st.error(
        """
        ❌ Feature order does not match the trained model.

        Please use the same feature order that was used
        during model training.
        """
    )

    st.stop()


# ============================================================
# VALIDATE SCALER
# ============================================================

if getattr(scaler, "n_features_in_", None) != 33:

    st.error(
        """
        ❌ Scaler feature count mismatch.

        The AI Twin requires a 33-feature scaler.
        """
    )

    st.stop()


# ============================================================
# REFERENCE VALUE FUNCTION
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

    feature_index = feature_names.index(feature)

    return float(
        scaler.mean_[feature_index]
    )


# ============================================================
# YES / NO SELECTOR
# ============================================================

def yes_no(label):

    return st.selectbox(
        label,
        [0, 1],
        format_func=lambda value:
        "Yes" if value == 1 else "No"
    )


# ============================================================
# GENETIC VARIANT SELECTOR
# ============================================================

def variant_input(label):

    return st.selectbox(
        label,
        [0, 1],
        format_func=lambda value:
        "Present" if value == 1 else "Absent"
    )


# ============================================================
# PATIENT-SPECIFIC CONTRIBUTION ANALYSIS
# ============================================================

def calculate_patient_contributions(patient_df):

    # Original prediction

    scaled_patient = scaler.transform(
        patient_df
    )

    original_probability = float(
        model.predict(
            scaled_patient,
            verbose=0
        )[0][0]
    )

    contributions = []

    # Change one feature at a time to its reference value

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
                contribution
        })

    result = pd.DataFrame(
        contributions
    )

    result["Absolute Contribution"] = (
        result["Contribution"].abs()
    )

    result = result.sort_values(
        "Absolute Contribution",
        ascending=False
    )

    return (
        result,
        original_probability
    )


# ============================================================
# APPLICATION HEADER
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
    ⚠️ **Research / Educational Prototype Only**

    This system is not a clinical diagnostic tool.
    The prediction should not be used alone for diagnosis,
    treatment or medical decision-making.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🧬 AI Twin Information"
    )

    st.write(
        "**Model:** Neural Network"
    )

    st.write(
        "**Input features:** 33"
    )

    st.write(
        "**Decision threshold:** 70%"
    )

    st.divider()

    st.write(
        """
        The Digital Twin uses:

        • Maternal information

        • Fetal information

        • Genetic variants

        • Gene-expression measurements
        """
    )


# ============================================================
# PATIENT INPUT
# ============================================================

st.header(
    "👩‍⚕️ Enter Patient Information"
)

with st.form(
    "patient_information_form"
):

    # ========================================================
    # MATERNAL INFORMATION
    # ========================================================

    st.subheader(
        "👩 Maternal Information"
    )

    column1, column2, column3 = st.columns(3)

    with column1:

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

    with column2:

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

    with column3:

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

    column1, column2, column3 = st.columns(3)

    with column1:

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

    with column2:

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

    with column3:

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

    column1, column2, column3 = st.columns(3)

    with column1:

        kcnj11_variant = variant_input(
            "KCNJ11 Variant"
        )

        abcc8_variant = variant_input(
            "ABCC8 Variant"
        )

        ins_variant = variant_input(
            "INS Variant"
        )

    with column2:

        chr6q24 = variant_input(
            "Chr6q24 Abnormality"
        )

        gck_variant = variant_input(
            "GCK Variant"
        )

        hnf1b_variant = variant_input(
            "HNF1B Variant"
        )

    with column3:

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

    expression_option = st.radio(
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

    if expression_option == (
        "Yes - enter measurements"
    ):

        st.info(
            """
            Enter the available gene-expression measurements.
            """
        )

        expression1, expression2, expression3 = (
            st.columns(3)
        )

        with expression1:

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

        with expression2:

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

        with expression3:

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

        # Use reference values if expression data
        # is not available.

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
            Gene-expression data were not entered.

            The Digital Twin will use the saved reference
            values for these features.
            """
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.divider()

    predict_button = st.form_submit_button(
        "🔍 Predict NDM Risk",
        use_container_width=True
    )


# ============================================================
# RUN PREDICTION
# ============================================================

if predict_button:

    # ========================================================
    # CREATE DATAFRAME
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
            "❌ Maternal diastolic BP must be lower than "
            "maternal systolic BP."
        )

        st.stop()


    # ========================================================
    # SCALE PATIENT DATA
    # ========================================================

    try:

        scaled_patient = scaler.transform(
            patient_df
        )

    except Exception as error:

        st.error(
            "❌ Error while scaling patient data."
        )

        st.exception(error)

        st.stop()


    # ========================================================
    # MODEL PREDICTION
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
            "❌ Error while generating prediction."
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
    # RISK CATEGORY
    # ========================================================

    if probability >= DECISION_THRESHOLD:

        risk_category = "HIGH RISK"

    else:

        risk_category = "LOWER RISK"


    # ========================================================
    # RESULT SECTION
    # ========================================================

    st.divider()

    st.header(
        "🧬 Digital Twin Prediction"
    )


    # ========================================================
    # RISK NOTE
    # ========================================================

    st.info(
        f"""
        ### ℹ️ Risk Note

        The Digital Twin estimates a model-predicted
        probability of **{percentage:.2f}%** for neonatal
        diabetes.

        The configured decision threshold is
        **{DECISION_THRESHOLD * 100:.0f}%**.

        The predicted probability is
        **{"above" if probability >= DECISION_THRESHOLD else "below"}**
        this threshold, so the current model classification
        is **{risk_category}**.

        This result represents the output of the trained
        machine-learning model. It is a research and
        educational prediction and **not a medical diagnosis**.
        """
    )


    # ========================================================
    # LARGE RISK DISPLAY
    # ========================================================

    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 30px;
            border-radius: 20px;
            border: 1px solid rgba(128,128,128,0.35);
            margin-top: 10px;
            margin-bottom: 25px;
        ">

            <div style="
                font-size: 20px;
                font-weight: 600;
            ">
                NDM RISK PROBABILITY
            </div>

            <div style="
                font-size: 68px;
                font-weight: 800;
                margin-top: 10px;
                margin-bottom: 10px;
            ">
                {percentage:.2f}%
            </div>

            <div style="
                font-size: 28px;
                font-weight: 700;
            ">
                {risk_category}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # RISK GRAPH
    # ========================================================

    st.subheader(
        "📊 Overall Prediction"
    )

    st.progress(
        float(probability)
    )

    graph_col1, graph_col2, graph_col3 = st.columns(3)

    with graph_col1:

        st.markdown(
            "**🟢 LOWER RISK**"
        )

        st.caption(
            "0% – 30%"
        )

    with graph_col2:

        st.markdown(
            "**🟡 INTERMEDIATE**"
        )

        st.caption(
            "30% – 70%"
        )

    with graph_col3:

        st.markdown(
            "**🔴 HIGHER RISK**"
        )

        st.caption(
            "70% – 100%"
        )


    # ========================================================
    # THRESHOLD RESULT
    # ========================================================

    if probability >= DECISION_THRESHOLD:

        st.error(
            f"""
            **Model classification: HIGH RISK**

            The predicted probability of **{percentage:.2f}%**
            is above the **70% decision threshold**.
            """
        )

    else:

        st.success(
            f"""
            **Model classification: LOWER RISK**

            The predicted probability of **{percentage:.2f}%**
            is below the **70% decision threshold**.
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
        **Why this patient received this result**

        The Digital Twin compares the patient's entered
        values with reference values used by the model.

        A **positive contribution** means that the feature
        moved the model prediction upward for this patient.

        A **negative contribution** means that the feature
        moved the model prediction downward.

        These contributions describe the model's estimated
        influence on this particular prediction. They should
        not be interpreted as proof that a feature caused
        neonatal diabetes.
        """
    )


    # ========================================================
    # CALCULATE CONTRIBUTIONS
    # ========================================================

    with st.spinner(
        "🔎 Analyzing patient-specific factors..."
    ):

        try:

            contribution_df, verified_probability = (
                calculate_patient_contributions(
                    patient_df
                )
            )

        except Exception as error:

            st.error(
                "❌ Unable to calculate patient-specific explanation."
            )

            st.exception(error)

            st.stop()


    # ========================================================
    # POSITIVE CONTRIBUTIONS
    # ========================================================

    increasing_risk = contribution_df[
        contribution_df["Contribution"] > 0
    ].sort_values(
        "Contribution",
        ascending=False
    ).head(5)


    # ========================================================
    # NEGATIVE CONTRIBUTIONS
    # ========================================================

    decreasing_risk = contribution_df[
        contribution_df["Contribution"] < 0
    ].sort_values(
        "Contribution",
        ascending=True
    ).head(5)


    # ========================================================
    # TWO-COLUMN EXPLANATION
    # ========================================================

    positive_column, negative_column = st.columns(2)


    # ========================================================
    # FEATURES INCREASING RISK
    # ========================================================

    with positive_column:

        st.subheader(
            "⬆️ Increasing predicted risk"
        )

        if increasing_risk.empty:

            st.success(
                "No positive feature contributions were detected."
            )

        else:

            for _, row in increasing_risk.iterrows():

                st.markdown(
                    f"""
                    **{row["Feature"]}**

                    Patient value: `{row["Patient Value"]}`

                    Reference value: `{row["Reference Value"]}`

                    Estimated contribution:
                    **+{row["Contribution"]:.5f}**
                    """
                )

                st.divider()


    # ========================================================
    # FEATURES DECREASING RISK
    # ========================================================

    with negative_column:

        st.subheader(
            "⬇️ Decreasing predicted risk"
        )

        if decreasing_risk.empty:

            st.success(
                "No negative feature contributions were detected."
            )

        else:

            for _, row in decreasing_risk.iterrows():

                st.markdown(
                    f"""
                    **{row["Feature"]}**

                    Patient value: `{row["Patient Value"]}`

                    Reference value: `{row["Reference Value"]}`

                    Estimated contribution:
                    **{row["Contribution"]:.5f}**
                    """
                )

                st.divider()


    # ========================================================
    # FINAL INTERPRETATION NOTE
    # ========================================================

    st.divider()

    st.info(
        """
        ### 🧬 Interpretation Note

        The risk percentage represents the probability
        estimated by the trained AI model for the entered
        feature combination.

        The feature contributions are model explanations
        intended to help understand the prediction. They do
        not establish medical causation.

        **This Digital Twin is a research/educational
        prototype and is not a clinical diagnostic system.**
        """
    )
