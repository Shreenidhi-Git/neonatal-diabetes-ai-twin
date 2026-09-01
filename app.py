import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Neonatal Diabetes AI Twin",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# FILES
# ============================================================

MODEL_FILE = "ndm_clean_neural_network.keras"
SCALER_FILE = "ndm_clean_33_feature_scaler.pkl"
FEATURE_FILE = "ndm_feature_names.pkl"
REFERENCE_FILE = "ndm_reference_values.pkl"
IMPORTANCE_FILE = "NDM_feature_importance.xlsx"

THRESHOLD = 0.70

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_and_files():

    model = tf.keras.models.load_model(MODEL_FILE)

    scaler = joblib.load(SCALER_FILE)

    with open(FEATURE_FILE, "rb") as f:
        feature_names = joblib.load(f)

    reference_values = None

    if os.path.exists(REFERENCE_FILE):
        with open(REFERENCE_FILE, "rb") as f:
            reference_values = joblib.load(f)

    return model, scaler, feature_names, reference_values


model, scaler, feature_names, reference_values = load_model_and_files()

# ============================================================
# LOAD GLOBAL FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_importance():

    if os.path.exists(IMPORTANCE_FILE):
        return pd.read_excel(IMPORTANCE_FILE)

    return None


importance_df = load_importance()

# ============================================================
# CHECK MODEL
# ============================================================

if len(feature_names) != 33:

    st.error(
        f"Model expects 33 features, but {len(feature_names)} "
        "feature names were found."
    )

    st.stop()

# ============================================================
# TITLE
# ============================================================

st.title("🧬 Neonatal Diabetes AI Twin")

st.subheader(
    "Early Neonatal Diabetes Risk Prediction"
)

st.write(
    "This research prototype estimates neonatal diabetes risk "
    "using maternal, fetal, genetic and gene-expression features."
)

st.warning(
    "⚠️ Research / educational prototype only. "
    "This system is NOT a clinical diagnostic tool."
)

st.divider()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def yes_no(label, default=0):

    return st.selectbox(
        label,
        [0, 1],
        index=default,
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


def variant_input(label):

    return st.selectbox(
        label,
        [0, 1],
        format_func=lambda x: "Present" if x == 1 else "Absent"
    )


def get_reference_range(feature, default_min, default_max, default_value):

    """
    Try to obtain a realistic range from reference values.
    Falls back to supplied safe range if reference information
    is unavailable.
    """

    try:

        if isinstance(reference_values, dict):

            value = reference_values.get(feature)

            if value is not None:

                if np.isscalar(value) and np.isfinite(value):

                    center = float(value)

                    if center != 0:

                        low = center * 0.5
                        high = center * 1.5

                        return low, high, center

    except Exception:
        pass

    return default_min, default_max, default_value


def local_feature_contributions(input_df):

    """
    Estimate patient-specific feature contribution.

    For each feature:
    1. Keep the patient's complete input.
    2. Replace that feature with its scaled reference/median value.
    3. Recalculate probability.
    4. Difference between original and modified probability
       is treated as that feature's local contribution.

    Positive = feature pushed probability higher.
    Negative = feature pushed probability lower.
    """

    original_scaled = scaler.transform(input_df)

    original_probability = float(
        model.predict(original_scaled, verbose=0)[0][0]
    )

    contributions = []

    for i, feature in enumerate(feature_names):

        modified = input_df.copy()

        # Determine reference value
        reference_value = None

        try:

            if isinstance(reference_values, dict):

                reference_value = reference_values.get(feature)

        except Exception:
            reference_value = None

        # If reference unavailable, use scaler mean transformed
        # back to original feature space.
        if reference_value is None:

            try:
                reference_value = float(scaler.mean_[i])

            except Exception:
                reference_value = input_df.iloc[0, i]

        modified.iloc[0, i] = reference_value

        modified_scaled = scaler.transform(modified)

        modified_probability = float(
            model.predict(modified_scaled, verbose=0)[0][0]
        )

        contribution = original_probability - modified_probability

        contributions.append(
            {
                "Feature": feature,
                "Patient Value": input_df.iloc[0, i],
                "Reference Value": reference_value,
                "Contribution": contribution
            }
        )

    result = pd.DataFrame(contributions)

    result["Absolute Contribution"] = result["Contribution"].abs()

    result = result.sort_values(
        "Absolute Contribution",
        ascending=False
    )

    return result, original_probability


# ============================================================
# PATIENT INPUT
# ============================================================

st.header("👩‍⚕️ Patient Information")

with st.form("patient_form"):

    # ========================================================
    # MATERNAL
    # ========================================================

    st.subheader("Maternal Information")

    c1, c2, c3 = st.columns(3)

    with c1:

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
            max_value=60.0,
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

    with c2:

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

    with c3:

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
    # FETAL
    # ========================================================

    st.subheader("Fetal Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        fetal_hr = st.number_input(
            "Fetal Heart Rate (bpm)",
            min_value=50.0,
            max_value=220.0,
            value=140.0,
            step=1.0
        )

        fetal_movement = st.number_input(
            "Fetal Movement per Hour",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        )

    with c2:

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

    with c3:

        consanguinity = yes_no(
            "Consanguinity"
        )

        autoimmune_history = yes_no(
            "Maternal Autoimmune History"
        )

    # ========================================================
    # GENETIC
    # ========================================================

    st.subheader("Genetic Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        kcnj11_variant = variant_input(
            "KCNJ11 Variant"
        )

        abcc8_variant = variant_input(
            "ABCC8 Variant"
        )

        ins_variant = variant_input(
            "INS Variant"
        )

    with c2:

        chr6q24 = variant_input(
            "Chr6q24 Abnormality"
        )

        gck_variant = variant_input(
            "GCK Variant"
        )

        hnf1b_variant = variant_input(
            "HNF1B Variant"
        )

    with c3:

        gata6_variant = variant_input(
            "GATA6 Variant"
        )

        glis3_variant = variant_input(
            "GLIS3 Variant"
        )

    # ========================================================
    # GENE EXPRESSION
    # ========================================================

    st.subheader("Gene Expression Values")

    st.info(
        "Enter the gene-expression measurements from the "
        "available genetic/laboratory data. Do not enter values "
        "unless they are available."
    )

    e1, e2, e3 = st.columns(3)

    with e1:

        ins_expr = st.number_input(
            "INS expression",
            value=0.0,
            step=0.01
        )

        pdx1_expr = st.number_input(
            "PDX1 expression",
            value=0.0,
            step=0.01
        )

        nkx6_1_expr = st.number_input(
            "NKX6_1 expression",
            value=0.0,
            step=0.01
        )

        mafa_expr = st.number_input(
            "MAFA expression",
            value=0.0,
            step=0.01
        )

    with e2:

        gck_expr = st.number_input(
            "GCK expression",
            value=0.0,
            step=0.01
        )

        slc2a2_expr = st.number_input(
            "SLC2A2 expression",
            value=0.0,
            step=0.01
        )

        abcc8_expr = st.number_input(
            "ABCC8 expression",
            value=0.0,
            step=0.01
        )

        kcnj11_expr = st.number_input(
            "KCNJ11 expression",
            value=0.0,
            step=0.01
        )

    with e3:

        neurod1_expr = st.number_input(
            "NEUROD1 expression",
            value=0.0,
            step=0.01
        )

        hnf1b_expr = st.number_input(
            "HNF1B expression",
            value=0.0,
            step=0.01
        )

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
    # EXACT 33-FEATURE ORDER
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
    # VALIDATION
    # --------------------------------------------------------

    if input_df.shape[1] != 33:

        st.error(
            "Input feature count does not match the model."
        )

        st.stop()

    if input_df.isnull().any().any():

        st.error(
            "Please provide values for all required features."
        )

        st.stop()

    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    try:

        scaled_input = scaler.transform(input_df)

    except Exception as e:

        st.error(
            f"Scaler error: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    probability = float(
        model.predict(
            scaled_input,
            verbose=0
        )[0][0]
    )

    probability = np.clip(
        probability,
        0,
        1
    )

    risk_percentage = probability * 100

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.divider()

    st.header("📊 Digital Twin Prediction")

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
            "🔴 HIGH RISK"
        )

    else:

        st.success(
            "🟢 LOWER RISK"
        )

    # ========================================================
    # PATIENT-SPECIFIC EXPLANATION
    # ========================================================

    st.divider()

    st.header(
        "🔎 Why did the Digital Twin give this result?"
    )

    st.write(
        "This section estimates how individual patient inputs "
        "affected the model probability by comparing the patient's "
        "prediction with a reference value for each feature."
    )

    with st.spinner(
        "Calculating patient-specific feature contributions..."
    ):

        local_df, verified_probability = (
            local_feature_contributions(input_df)
        )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    difference = abs(
        probability - verified_probability
    )

    if difference < 0.0001:

        st.success(
            "✅ Patient-specific explanation calculated successfully."
        )

    # --------------------------------------------------------
    # POSITIVE CONTRIBUTORS
    # --------------------------------------------------------

    positive = local_df[
        local_df["Contribution"] > 0
    ].head(10)

    negative = local_df[
        local_df["Contribution"] < 0
    ].sort_values(
        "Contribution"
    ).head(10)

    c1, c2 = st.columns(2)

    with c1:

        st.subheader(
            "⬆️ Features increasing predicted risk"
        )

        if len(positive) == 0:

            st.write(
                "No positive contributors identified."
            )

        else:

            for _, row in positive.iterrows():

                st.write(
                    f"**{row['Feature']}**  \n"
                    f"Patient value: `{row['Patient Value']}`  \n"
                    f"Reference value: `{row['Reference Value']}`  \n"
                    f"Estimated contribution: "
                    f"`+{row['Contribution']:.5f}`"
                )

                st.divider()

    with c2:

        st.subheader(
            "⬇️ Features decreasing predicted risk"
        )

        if len(negative) == 0:

            st.write(
                "No negative contributors identified."
            )

        else:

            for _, row in negative.iterrows():

                st.write(
                    f"**{row['Feature']}**  \n"
                    f"Patient value: `{row['Patient Value']}`  \n"
                    f"Reference value: `{row['Reference Value']}`  \n"
                    f"Estimated contribution: "
                    f"`{row['Contribution']:.5f}`"
                )

                st.divider()

    # ========================================================
    # GLOBAL IMPORTANCE
    # ========================================================

    st.divider()

    st.subheader(
        "📈 General Model Feature Importance"
    )

    st.caption(
        "This is different from the patient-specific explanation "
        "above. These values describe overall model importance."
    )

    if importance_df is not None:

        st.dataframe(
            importance_df.head(10),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # PATIENT INPUT SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Patient Input Summary"
    )

    st.dataframe(
        input_df.T.rename(
            columns={0: "Patient Value"}
        ),
        use_container_width=True
    )

    # ========================================================
    # FINAL WARNING
    # ========================================================

    st.warning(
        "This AI Twin is a research/educational prototype. "
        "The prediction and feature contributions must not be "
        "used alone for diagnosis, treatment, or clinical decision-making."
    )
