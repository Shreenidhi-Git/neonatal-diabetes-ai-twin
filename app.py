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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE NAMES
# ============================================================

MODEL_FILE = "ndm_clean_neural_network.keras"
SCALER_FILE = "ndm_clean_33_feature_scaler.pkl"
FEATURE_FILE = "ndm_feature_names.pkl"
REFERENCE_FILE = "ndm_reference_values.pkl"
IMPORTANCE_FILE = "NDM_feature_importance.xlsx"

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

    st.error("❌ Required model files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Make sure these files are uploaded to the same GitHub "
        "repository as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL AND SUPPORT FILES
# ============================================================

@st.cache_resource
def load_model_files():

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

    return model, scaler, feature_names, reference_values


try:

    model, scaler, feature_names, reference_values = (
        load_model_files()
    )

except Exception as e:

    st.error("❌ Could not load the AI Twin model files.")

    st.exception(e)

    st.stop()


# ============================================================
# FEATURE VALIDATION
# ============================================================

feature_names = list(feature_names)

if len(feature_names) != 33:

    st.error(
        f"❌ Expected 33 model features, but found "
        f"{len(feature_names)}."
    )

    st.stop()


if feature_names != EXPECTED_FEATURES:

    st.error(
        "❌ Feature order does not match the trained model."
    )

    st.write("Expected feature order:")

    for i, feature in enumerate(EXPECTED_FEATURES, 1):
        st.write(f"{i:02d}. {feature}")

    st.write("Loaded feature order:")

    for i, feature in enumerate(feature_names, 1):
        st.write(f"{i:02d}. {feature}")

    st.stop()


# ============================================================
# SCALER VALIDATION
# ============================================================

try:

    scaler_feature_count = scaler.n_features_in_

except Exception:

    scaler_feature_count = None


if scaler_feature_count != 33:

    st.error(
        f"❌ Scaler expects {scaler_feature_count} features, "
        f"but the AI Twin requires 33."
    )

    st.stop()


# ============================================================
# LOAD GLOBAL FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_feature_importance():

    if not os.path.exists(IMPORTANCE_FILE):

        return None

    try:

        df = pd.read_excel(
            IMPORTANCE_FILE
        )

        return df

    except Exception:

        return None


importance_df = load_feature_importance()


# ============================================================
# HELPER: REFERENCE VALUE
# ============================================================

def get_reference_value(feature, fallback=0.0):

    """
    Get the saved reference value for a feature.

    Supports dictionary-like reference files.
    """

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

    return float(fallback)


# ============================================================
# HELPER: YES / NO INPUT
# ============================================================

def yes_no_input(label):

    return st.selectbox(
        label,
        options=[0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


# ============================================================
# HELPER: VARIANT INPUT
# ============================================================

def variant_input(label):

    return st.selectbox(
        label,
        options=[0, 1],
        format_func=lambda x:
            "Present" if x == 1 else "Absent"
    )


# ============================================================
# PATIENT-SPECIFIC CONTRIBUTIONS
# ============================================================

def calculate_local_contributions(input_df):

    """
    Calculates estimated patient-specific contributions.

    Method:

    Original prediction:
        patient values for all 33 features

    For each feature:
        replace only that feature with its saved reference value

    Then:

        contribution =
        original probability - modified probability

    Positive contribution:
        feature increased the model's predicted probability

    Negative contribution:
        feature decreased the model's predicted probability
    """

    original_scaled = scaler.transform(
        input_df
    )

    original_probability = float(
        model.predict(
            original_scaled,
            verbose=0
        )[0][0]
    )

    results = []

    for i, feature in enumerate(feature_names):

        modified_df = input_df.copy()

        # ----------------------------------------------------
        # Get reference value
        # ----------------------------------------------------

        reference_value = get_reference_value(
            feature,
            fallback=scaler.mean_[i]
        )

        # ----------------------------------------------------
        # Replace ONLY this feature
        # ----------------------------------------------------

        modified_df.iloc[0, i] = reference_value

        modified_scaled = scaler.transform(
            modified_df
        )

        modified_probability = float(
            model.predict(
                modified_scaled,
                verbose=0
            )[0][0]
        )

        # ----------------------------------------------------
        # Contribution
        # ----------------------------------------------------

        contribution = (
            original_probability -
            modified_probability
        )

        results.append({
            "Feature": feature,
            "Patient Value": input_df.iloc[0, i],
            "Reference Value": reference_value,
            "Contribution": contribution,
            "Absolute Contribution": abs(
                contribution
            )
        })

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "Absolute Contribution",
        ascending=False
    )

    return result_df, original_probability


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
    This system is NOT a clinical diagnostic tool.
    Do not use this prediction alone for diagnosis,
    treatment, or clinical decision-making.
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
        "Model features: **33**"
    )

    st.write(
        f"Decision threshold: **{THRESHOLD:.2f}**"
    )

    st.divider()

    st.write(
        """
        The AI Twin combines:

        • Maternal information  
        • Fetal information  
        • Genetic variants  
        • Gene-expression measurements
        """
    )

    st.divider()

    st.caption(
        "Research / educational prototype"
    )


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header(
    "👩‍⚕️ Patient Information"
)

with st.form(
    "patient_information_form"
):

    # ========================================================
    # MATERNAL INFORMATION
    # ========================================================

    st.subheader(
        "Maternal Information"
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

        family_history = yes_no_input(
            "Family History of Diabetes"
        )

        previous_gdm = yes_no_input(
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

        fetal_hr = st.number_input(
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

        growth_percentile = st.number_input(
            "Ultrasound Growth Percentile",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

        fetal_weight = st.number_input(
            "Estimated Fetal Weight (g)",
            min_value=200.0,
            max_value=6000.0,
            value=2000.0,
            step=10.0
        )

    with col3:

        consanguinity = yes_no_input(
            "Consanguinity"
        )

        autoimmune_history = yes_no_input(
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
        "🧪 Gene Expression Information"
    )

    st.info(
        """
        Gene-expression measurements are optional.

        If measurements are available, select YES and enter
        the patient's actual measurements.

        If measurements are unavailable, select NO and the
        saved reference values will be used instead of
        incorrectly assuming that the measurements are zero.
        """
    )

    expression_available = st.radio(
        "Are gene-expression measurements available?",
        [
            "Yes - I have the measurements",
            "No - use saved reference values"
        ],
        horizontal=True
    )


    # ========================================================
    # ACTUAL EXPRESSION VALUES
    # ========================================================

    if (
        expression_available ==
        "Yes - I have the measurements"
    ):

        st.success(
            "Enter the patient's actual gene-expression measurements."
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


    # ========================================================
    # REFERENCE VALUES
    # ========================================================

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

        st.warning(
            """
            Gene-expression measurements were not provided.

            The saved reference values will be used for the
            10 gene-expression features.
            """
        )

        expression_reference_df = pd.DataFrame({
            "Feature": [
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
            ],
            "Reference Value": [
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
        })

        with st.expander(
            "View gene-expression reference values"
        ):

            st.dataframe(
                expression_reference_df,
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # PREDICT BUTTON
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
    # CREATE INPUT VECTOR
    # ========================================================

    input_values = [

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


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    input_df = pd.DataFrame(
        [input_values],
        columns=EXPECTED_FEATURES
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    if input_df.shape != (1, 33):

        st.error(
            "❌ Input data does not contain exactly 33 features."
        )

        st.stop()


    if input_df.isnull().any().any():

        st.error(
            "❌ Missing values detected."
        )

        st.stop()


    # --------------------------------------------------------
    # Blood pressure logical validation
    # --------------------------------------------------------

    if diastolic_bp >= systolic_bp:

        st.error(
            "❌ Maternal diastolic BP should be lower "
            "than maternal systolic BP."
        )

        st.stop()


    # ========================================================
    # SCALE INPUT
    # ========================================================

    try:

        scaled_input = scaler.transform(
            input_df
        )

    except Exception as e:

        st.error(
            "❌ Error while scaling patient data."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    try:

        probability = float(
            model.predict(
                scaled_input,
                verbose=0
            )[0][0]
        )

    except Exception as e:

        st.error(
            "❌ Model prediction failed."
        )

        st.exception(e)

        st.stop()


    probability = float(
        np.clip(
            probability,
            0,
            1
        )
    )

    risk_percentage = probability * 100


    # ========================================================
    # RISK CATEGORY
    # ========================================================

    if probability >= THRESHOLD:

        risk_category = "HIGH RISK"

    else:

        risk_category = "LOWER RISK"


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header(
        "📊 Digital Twin Prediction"
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "NDM Probability",
            f"{probability:.4f}"
        )

    with r2:

        st.metric(
            "Risk Percentage",
            f"{risk_percentage:.2f}%"
        )

    with r3:

        st.metric(
            "Decision Threshold",
            f"{THRESHOLD:.2f}"
        )


    if probability >= THRESHOLD:

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

    st.divider()

    st.header(
        "🔎 Why did the Digital Twin give this result?"
    )

    st.write(
        """
        The explanation below is an estimated
        patient-specific contribution.

        For each feature, the AI Twin compares the patient's
        prediction with a prediction obtained after replacing
        that one feature with its saved reference value.

        Positive contribution means the feature increased the
        model's predicted probability relative to its reference.

        Negative contribution means the feature decreased the
        model's predicted probability relative to its reference.
        """
    )


    # ========================================================
    # CALCULATE CONTRIBUTIONS
    # ========================================================

    with st.spinner(
        "Calculating patient-specific contributions..."
    ):

        try:

            local_df, verified_probability = (
                calculate_local_contributions(
                    input_df
                )
            )

        except Exception as e:

            st.error(
                "❌ Could not calculate feature contributions."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # VERIFY PREDICTION
    # ========================================================

    prediction_difference = abs(
        probability -
        verified_probability
    )

    if prediction_difference < 0.0001:

        st.success(
            "✅ Patient-specific explanation calculated "
            "from the same model prediction."
        )

    else:

        st.warning(
            "⚠️ Small prediction difference detected "
            "during explanation calculation."
        )


    # ========================================================
    # POSITIVE CONTRIBUTIONS
    # ========================================================

    positive_df = local_df[
        local_df["Contribution"] > 0
    ].head(10)


    # ========================================================
    # NEGATIVE CONTRIBUTIONS
    # ========================================================

    negative_df = (
        local_df[
            local_df["Contribution"] < 0
        ]
        .sort_values(
            "Contribution"
        )
        .head(10)
    )


    # ========================================================
    # DISPLAY CONTRIBUTIONS
    # ========================================================

    left, right = st.columns(2)


    # --------------------------------------------------------
    # INCREASING RISK
    # --------------------------------------------------------

    with left:

        st.subheader(
            "⬆️ Features increasing predicted risk"
        )

        if len(positive_df) == 0:

            st.info(
                "No positive contributions detected."
            )

        else:

            for _, row in positive_df.iterrows():

                st.markdown(
                    f"""
                    **{row['Feature']}**

                    Patient value: `{row['Patient Value']}`

                    Reference value: `{row['Reference Value']}`

                    Estimated contribution:
                    `+{row['Contribution']:.5f}`
                    """
                )

                st.divider()


    # --------------------------------------------------------
    # DECREASING RISK
    # --------------------------------------------------------

    with right:

        st.subheader(
            "⬇️ Features decreasing predicted risk"
        )

        if len(negative_df) == 0:

            st.info(
                "No negative contributions detected."
            )

        else:

            for _, row in negative_df.iterrows():

                st.markdown(
                    f"""
                    **{row['Feature']}**

                    Patient value: `{row['Patient Value']}`

                    Reference value: `{row['Reference Value']}`

                    Estimated contribution:
                    `{row['Contribution']:.5f}`
                    """
                )

                st.divider()


    # ========================================================
    # TOP 10 PATIENT-SPECIFIC FEATURES
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Top Patient-Specific Contributors"
    )

    top_local = local_df.head(10).copy()

    top_local = top_local[
        [
            "Feature",
            "Patient Value",
            "Reference Value",
            "Contribution"
        ]
    ]

    top_local["Contribution"] = (
        top_local["Contribution"]
        .round(5)
    )

    st.dataframe(
        top_local,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # GLOBAL MODEL IMPORTANCE
    # ========================================================

    st.divider()

    st.subheader(
        "📈 General Model Feature Importance"
    )

    st.caption(
        """
        These values describe overall feature importance
        in the trained model. They are NOT the same as the
        patient-specific contributions above.
        """
    )

    if importance_df is not None:

        st.dataframe(
            importance_df.head(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Feature-importance file was not found."
        )


    # ========================================================
    # PATIENT INPUT SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Patient Input Summary"
    )

    patient_summary = input_df.T.copy()

    patient_summary.columns = [
        "Patient Value"
    ]

    st.dataframe(
        patient_summary,
        use_container_width=True
    )


    # ========================================================
    # FINAL SAFETY MESSAGE
    # ========================================================

    st.divider()

    st.warning(
        """
        ⚠️ IMPORTANT:

        This AI Twin is a research and educational prototype.
        The probability represents the output of the trained
        machine-learning model and is not a medical diagnosis.

        Clinical decisions must not be made from this system
        without appropriate medical evaluation and validation.
        """
            )
