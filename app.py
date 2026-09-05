import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Neonatal Diabetes AI Twin",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = "ndm_clean_neural_network.keras"
SCALER_FILE = "ndm_clean_33_feature_scaler.pkl"
FEATURE_FILE = "ndm_feature_names.pkl"
REFERENCE_FILE = "ndm_reference_values.pkl"


# ============================================================
# 33 FEATURES USED BY THE MODEL
# ============================================================

EXPECTED_FEATURES = [

    # Maternal
    "Maternal_Age_years",
    "Maternal_BMI",
    "Maternal_Systolic_BP",
    "Maternal_Diastolic_BP",
    "Maternal_Fasting_Glucose_mg_dL",
    "Maternal_HbA1c_percent",
    "Gestational_Age_weeks",

    # Fetal
    "Fetal_Heart_Rate_bpm",
    "Fetal_Movement_per_hour",
    "Ultrasound_Growth_Percentile",
    "Estimated_Fetal_Weight_g",

    # Maternal history
    "Family_History_Diabetes",
    "Consanguinity",
    "Previous_Gestational_Diabetes",
    "Maternal_Autoimmune_History",

    # Genetic
    "KCNJ11_Variant",
    "ABCC8_Variant",
    "INS_Variant",
    "Chr6q24_Abnormality",
    "GCK_Variant",
    "HNF1B_Variant",
    "GATA6_Variant",
    "GLIS3_Variant",

    # Gene expression
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
# HUMAN-READABLE FEATURE NAMES
# ============================================================

FEATURE_DISPLAY_NAMES = {

    "Maternal_Age_years":
        "Maternal age",

    "Maternal_BMI":
        "Maternal BMI",

    "Maternal_Systolic_BP":
        "Maternal systolic blood pressure",

    "Maternal_Diastolic_BP":
        "Maternal diastolic blood pressure",

    "Maternal_Fasting_Glucose_mg_dL":
        "Maternal fasting glucose",

    "Maternal_HbA1c_percent":
        "Maternal HbA1c",

    "Gestational_Age_weeks":
        "Gestational age",

    "Fetal_Heart_Rate_bpm":
        "Fetal heart rate",

    "Fetal_Movement_per_hour":
        "Fetal movement",

    "Ultrasound_Growth_Percentile":
        "Ultrasound growth percentile",

    "Estimated_Fetal_Weight_g":
        "Estimated fetal weight",

    "Family_History_Diabetes":
        "Family history of diabetes",

    "Consanguinity":
        "Consanguinity",

    "Previous_Gestational_Diabetes":
        "Previous gestational diabetes",

    "Maternal_Autoimmune_History":
        "Maternal autoimmune history",

    "KCNJ11_Variant":
        "KCNJ11 genetic information",

    "ABCC8_Variant":
        "ABCC8 genetic information",

    "INS_Variant":
        "INS genetic information",

    "Chr6q24_Abnormality":
        "Chr6q24 genetic information",

    "GCK_Variant":
        "GCK genetic information",

    "HNF1B_Variant":
        "HNF1B genetic information",

    "GATA6_Variant":
        "GATA6 genetic information",

    "GLIS3_Variant":
        "GLIS3 genetic information",

    "INS_expr":
        "INS gene expression",

    "PDX1_expr":
        "PDX1 gene expression",

    "NKX6_1_expr":
        "NKX6_1 gene expression",

    "MAFA_expr":
        "MAFA gene expression",

    "GCK_expr":
        "GCK gene expression",

    "SLC2A2_expr":
        "SLC2A2 gene expression",

    "ABCC8_expr":
        "ABCC8 gene expression",

    "KCNJ11_expr":
        "KCNJ11 gene expression",

    "NEUROD1_expr":
        "NEUROD1 gene expression",

    "HNF1B_expr":
        "HNF1B gene expression"
}


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_FILE,
    REFERENCE_FILE
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.error(
        "Required AI Twin files are missing."
    )

    st.write(
        "The following files were not found:"
    )

    for file in missing_files:

        st.write(
            f"❌ {file}"
        )

    st.info(
        "Make sure these files are uploaded to the "
        "same GitHub repository as app.py."
    )

    st.stop()


# ============================================================
# LOAD AI TWIN
# ============================================================

@st.cache_resource
def load_ai_twin():

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    feature_names = joblib.load(
        FEATURE_FILE
    )

    reference_values = joblib.load(
        REFERENCE_FILE
    )

    return (
        model,
        scaler,
        feature_names,
        reference_values
    )


try:

    model, scaler, feature_names, reference_values = (
        load_ai_twin()
    )

except Exception as e:

    st.error(
        "The AI Twin model could not be loaded."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# VALIDATE FEATURE COUNT
# ============================================================

if len(EXPECTED_FEATURES) != 33:

    st.error(
        "Internal error: expected exactly 33 features."
    )

    st.stop()


# ============================================================
# REFERENCE VALUE FUNCTION
# ============================================================

def get_reference_value(feature):

    try:

        if isinstance(reference_values, dict):

            return float(
                reference_values[feature]
            )

        elif isinstance(reference_values, pd.Series):

            return float(
                reference_values[feature]
            )

        elif isinstance(reference_values, pd.DataFrame):

            return float(
                reference_values[feature].iloc[0]
            )

    except Exception:

        return 0.0

    return 0.0


# ============================================================
# YES / NO INPUT
# ============================================================

def yes_no_input(label):

    value = st.selectbox(
        label,
        [
            "No",
            "Yes"
        ]
    )

    if value == "Yes":

        return 1

    return 0


# ============================================================
# GENETIC VARIANT INPUT
# ============================================================

def variant_input(label):

    value = st.selectbox(
        label,
        [
            "No variant detected",
            "Variant detected"
        ]
    )

    if value == "Variant detected":

        return 1

    return 0


# ============================================================
# PATIENT-SPECIFIC FEATURE CONTRIBUTIONS
# ============================================================

def calculate_patient_contributions(
    input_data,
    original_probability
):

    contributions = []

    for index, feature in enumerate(
        EXPECTED_FEATURES
    ):

        reference_value = get_reference_value(
            feature
        )

        modified_data = np.array(
            input_data,
            dtype=float
        ).copy()

        modified_data[index] = (
            reference_value
        )

        try:

            modified_scaled = scaler.transform(
                modified_data.reshape(1, -1)
            )

            modified_prediction = model.predict(
                modified_scaled,
                verbose=0
            )

            modified_probability = float(
                modified_prediction[0][0]
            )

            contribution = (
                original_probability
                - modified_probability
            )

        except Exception:

            contribution = 0.0

        contributions.append(
            {
                "feature": feature,
                "patient_value": input_data[index],
                "reference_value": reference_value,
                "contribution": contribution
            }
        )

    contributions.sort(
        key=lambda x: abs(
            x["contribution"]
        ),
        reverse=True
    )

    return contributions


# ============================================================
# TITLE
# ============================================================

st.title(
    "🧬 Neonatal Diabetes AI Digital Twin"
)

st.write(
    "AI-based neonatal diabetes mellitus risk estimation "
    "using maternal, fetal, genetic and molecular features."
)


# ============================================================
# DISCLAIMER
# ============================================================

st.info(
    "Research / educational prototype only. "
    "This system is not a clinical diagnostic tool."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧬 AI Twin"
)

st.sidebar.write(
    "Enter the available patient information "
    "to generate an estimated NDM risk."
)

st.sidebar.markdown("---")

st.sidebar.write(
    "**Model:** Neural Network"
)

st.sidebar.write(
    "**Input Features:** 33"
)

st.sidebar.write(
    "**Decision Threshold:** 70%"
)


# ============================================================
# MATERNAL INFORMATION
# ============================================================

st.header(
    "👩 Maternal Information"
)

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
        max_value=220.0,
        value=120.0
    )

    diastolic_bp = st.number_input(
        "Maternal Diastolic BP",
        min_value=40.0,
        max_value=140.0,
        value=80.0
    )


with col2:

    fasting_glucose = st.number_input(
        "Maternal Fasting Glucose (mg/dL)",
        min_value=40.0,
        max_value=300.0,
        value=90.0
    )

    hba1c = st.number_input(
        "Maternal HbA1c (%)",
        min_value=3.0,
        max_value=20.0,
        value=5.5
    )

    gestational_age = st.number_input(
        "Gestational Age (weeks)",
        min_value=20.0,
        max_value=45.0,
        value=38.0
    )


with col3:

    family_history = yes_no_input(
        "Family History of Diabetes"
    )

    consanguinity = yes_no_input(
        "Consanguinity"
    )

    previous_gdm = yes_no_input(
        "Previous Gestational Diabetes"
    )

    autoimmune_history = yes_no_input(
        "Maternal Autoimmune History"
    )


# ============================================================
# FETAL INFORMATION
# ============================================================

st.header(
    "👶 Fetal Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    fetal_heart_rate = st.number_input(
        "Fetal Heart Rate (bpm)",
        min_value=80.0,
        max_value=220.0,
        value=140.0
    )

    fetal_movement = st.number_input(
        "Fetal Movement (per hour)",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )


with col2:

    growth_percentile = st.number_input(
        "Ultrasound Growth Percentile",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )


with col3:

    fetal_weight = st.number_input(
        "Estimated Fetal Weight (g)",
        min_value=200.0,
        max_value=6000.0,
        value=3000.0
    )


# ============================================================
# GENETIC INFORMATION
# ============================================================

st.header(
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


# ============================================================
# GENE EXPRESSION INFORMATION
# ============================================================

st.header(
    "🔬 Gene Expression Information"
)

st.write(
    "Enter molecular gene-expression values when available."
)

col1, col2, col3 = st.columns(3)


with col1:

    ins_expr = st.number_input(
        "INS Expression",
        value=0.0
    )

    pdx1_expr = st.number_input(
        "PDX1 Expression",
        value=0.0
    )

    nkx6_1_expr = st.number_input(
        "NKX6_1 Expression",
        value=0.0
    )


with col2:

    mafa_expr = st.number_input(
        "MAFA Expression",
        value=0.0
    )

    gck_expr = st.number_input(
        "GCK Expression",
        value=0.0
    )

    slc2a2_expr = st.number_input(
        "SLC2A2 Expression",
        value=0.0
    )


with col3:

    abcc8_expr = st.number_input(
        "ABCC8 Expression",
        value=0.0
    )

    kcnj11_expr = st.number_input(
        "KCNJ11 Expression",
        value=0.0
    )

    neurod1_expr = st.number_input(
        "NEUROD1 Expression",
        value=0.0
    )

    hnf1b_expr = st.number_input(
        "HNF1B Expression",
        value=0.0
    )


# ============================================================
# CREATE 33-FEATURE INPUT
# ============================================================

input_data = [

    # Maternal
    maternal_age,
    maternal_bmi,
    systolic_bp,
    diastolic_bp,
    fasting_glucose,
    hba1c,
    gestational_age,

    # Fetal
    fetal_heart_rate,
    fetal_movement,
    growth_percentile,
    fetal_weight,

    # Maternal history
    family_history,
    consanguinity,
    previous_gdm,
    autoimmune_history,

    # Genetic
    kcnj11_variant,
    abcc8_variant,
    ins_variant,
    chr6q24,
    gck_variant,
    hnf1b_variant,
    gata6_variant,
    glis3_variant,

    # Gene expression
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


# ============================================================
# VERIFY INPUT COUNT
# ============================================================

if len(input_data) != 33:

    st.error(
        f"Input error: {len(input_data)} features "
        "were generated instead of 33."
    )

    st.stop()


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Neonatal Diabetes Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # CONVERT INPUT TO NUMPY ARRAY
        # ----------------------------------------------------

        input_array = np.array(
            input_data,
            dtype=float
        ).reshape(1, -1)


        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        scaled_input = scaler.transform(
            input_array
        )


        # ----------------------------------------------------
        # NEURAL NETWORK PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            scaled_input,
            verbose=0
        )


        probability = float(
            prediction[0][0]
        )


        # Keep probability between 0 and 1

        probability = max(
            0.0,
            min(
                1.0,
                probability
            )
        )


        percentage = (
            probability * 100
        )


        # ----------------------------------------------------
        # DECISION THRESHOLD
        # ----------------------------------------------------

        threshold = 0.70


        if probability >= threshold:

            risk_category = "HIGH RISK"

        else:

            risk_category = "LOWER RISK"


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.header(
            "🧬 Digital Twin Prediction"
        )


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Estimated NDM Risk",
                f"{percentage:.2f}%"
            )


        with result_col2:

            st.metric(
                "Risk Classification",
                risk_category
            )


        # ====================================================
        # DIGITAL TWIN PROFILE
        # ====================================================

        st.subheader(
            "👤 Digital Twin Profile"
        )


        profile1, profile2, profile3, profile4 = (
            st.columns(4)
        )


        with profile1:

            st.markdown(
                "### 👩 Maternal"
            )

            st.write(
                f"Age: {maternal_age:.1f} years"
            )

            st.write(
                f"BMI: {maternal_bmi:.1f}"
            )

            st.write(
                f"Fasting glucose: "
                f"{fasting_glucose:.1f} mg/dL"
            )

            st.write(
                f"HbA1c: {hba1c:.1f}%"
            )


        with profile2:

            st.markdown(
                "### 👶 Fetal"
            )

            st.write(
                f"Heart rate: "
                f"{fetal_heart_rate:.0f} bpm"
            )

            st.write(
                f"Movement: "
                f"{fetal_movement:.0f}/hour"
            )

            st.write(
                f"Growth percentile: "
                f"{growth_percentile:.0f}%"
            )

            st.write(
                f"Estimated weight: "
                f"{fetal_weight:.0f} g"
            )


        with profile3:

            st.markdown(
                "### 🧬 Genetic"
            )

            genetic_count = sum(
                [
                    kcnj11_variant,
                    abcc8_variant,
                    ins_variant,
                    chr6q24,
                    gck_variant,
                    hnf1b_variant,
                    gata6_variant,
                    glis3_variant
                ]
            )

            st.write(
                f"Variants/abnormalities: "
                f"{genetic_count}"
            )

            st.write(
                "8 genetic indicators"
            )


        with profile4:

            st.markdown(
                "### 🔬 Molecular"
            )

            st.write(
                "Gene-expression layer"
            )

            st.write(
                "10 molecular features"
            )

            st.write(
                "Patient-specific input"
            )


        # ====================================================
        # OVERALL PREDICTION GRAPH
        # ====================================================

        st.subheader(
            "📊 Overall Prediction"
        )

        st.write(
            "The graph shows the estimated NDM risk "
            "generated by the neural network."
        )


        fig, ax = plt.subplots(
            figsize=(10, 3)
        )


        ax.barh(
            ["NDM Risk"],
            [percentage],
            alpha=0.85
        )


        ax.axvline(
            x=70,
            linestyle="--",
            linewidth=2,
            label="70% Decision Threshold"
        )


        ax.text(
            percentage,
            0,
            f"  {percentage:.2f}%",
            va="center",
            ha="left",
            fontsize=14,
            fontweight="bold"
        )


        ax.text(
            70,
            0.35,
            "70% threshold",
            ha="center",
            fontsize=10
        )


        ax.set_xlim(
            0,
            100
        )


        ax.set_xlabel(
            "Estimated NDM Risk (%)"
        )


        ax.set_title(
            "AI Twin Risk Prediction"
        )


        ax.grid(
            axis="x",
            alpha=0.25
        )


        ax.legend(
            loc="upper left"
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        # ====================================================
        # RISK SCALE
        # ====================================================

        st.markdown(
            "**Risk scale:** "
            "0% ───────── 50% ───────── "
            "70% ───────── 100%"
        )

        st.caption(
            "Below 70% = Lower Risk | "
            "70% or above = High Risk"
        )


        # ====================================================
        # RISK INTERPRETATION
        # ====================================================

        st.subheader(
            "📌 Risk Interpretation"
        )


        if probability >= threshold:

            st.error(
                f"""
                The Digital Twin estimates an NDM risk
                of {percentage:.2f}%, which is above the
                configured 70% decision threshold.

                This is an estimated model output and
                should not be interpreted as a diagnosis.
                """
            )

        else:

            st.success(
                f"""
                The Digital Twin estimates an NDM risk
                of {percentage:.2f}%, which is below the
                configured 70% decision threshold.

                A lower model prediction does not guarantee
                that neonatal diabetes is absent.
                """
            )


        # ====================================================
        # WHY DID THE MODEL GIVE THIS RESULT?
        # ====================================================

        st.subheader(
            "🔎 Why did the AI Twin give this result?"
        )


        st.write(
            "The AI Twin analyzes the combination of "
            "maternal, fetal, genetic and molecular "
            "information entered for this patient."
        )

        st.write(
            "The factors below had the strongest estimated "
            "influence on the model's current prediction."
        )


        # ====================================================
        # CALCULATE CONTRIBUTIONS
        # ====================================================

        contributions = (
            calculate_patient_contributions(
                input_data,
                probability
            )
        )


        # ====================================================
        # TOP CONTRIBUTING FACTORS
        # ====================================================

        st.markdown(
            "### 🧠 Main Factors Influencing the Prediction"
        )


        top_contributors = contributions[:5]


        for rank, item in enumerate(
            top_contributors,
            start=1
        ):

            feature = item["feature"]

            influence = item["contribution"]

            display_name = (
                FEATURE_DISPLAY_NAMES.get(
                    feature,
                    feature
                )
            )


            if influence > 0:

                st.markdown(
                    f"""
                    **{rank}. 🔺 {display_name}**

                    This feature was estimated to
                    **increase the model's predicted risk**.

                    Model influence: **{influence:+.4f}**
                    """
                )


            elif influence < 0:

                st.markdown(
                    f"""
                    **{rank}. 🔻 {display_name}**

                    This feature was estimated to
                    **decrease the model's predicted risk**.

                    Model influence: **{influence:+.4f}**
                    """
                )


            else:

                st.markdown(
                    f"""
                    **{rank}. ➖ {display_name}**

                    This feature had little estimated
                    influence on the current prediction.
                    """
                )


        # ====================================================
        # OVERALL MODEL EXPLANATION
        # ====================================================

        st.markdown(
            "### 📋 Overall Explanation"
        )


        if probability >= threshold:

            st.warning(
                f"""
                The AI Twin estimated a relatively high
                model risk of **{percentage:.2f}%**.

                The prediction crossed the configured
                **70% decision threshold** because of the
                combined effect of the entered features.

                The prediction is based on the combination
                of features rather than on one feature alone.
                """
            )

        else:

            st.success(
                f"""
                The AI Twin estimated a lower model risk
                of **{percentage:.2f}%**.

                The prediction remained below the configured
                **70% decision threshold** because of the
                combined effect of the entered features.

                A lower prediction does not prove that
                neonatal diabetes is absent.
                """
            )


        # ====================================================
        # WHAT SHOULD HAPPEN NEXT?
        # ====================================================

        st.subheader(
            "🩺 What Should Happen Next?"
        )


        if probability >= threshold:

            st.markdown(
                """
                ### ⚠️ Elevated Model Risk

                This result should **not be interpreted as
                a diagnosis**.

                Based on the information available to the
                AI Twin, the following areas may be appropriate
                for review by a qualified healthcare professional.

                **1. 👩 Review maternal information**

                Review the available maternal glucose,
                HbA1c, blood pressure and pregnancy information.

                **2. 👶 Review fetal information**

                Review fetal growth, estimated fetal weight,
                ultrasound findings and other relevant
                prenatal measurements.

                **3. 🧬 Review family and genetic information**

                Review available family history and genetic
                information.

                **4. 🔬 Consider genetic evaluation**

                If neonatal diabetes is clinically suspected,
                a healthcare professional may consider whether
                genetic evaluation or testing is appropriate.

                **5. 📋 Continue appropriate prenatal monitoring**

                Further monitoring and evaluation should be
                determined by the treating healthcare
                professional.
                """
            )

        else:

            st.markdown(
                """
                ### ✅ Lower Model Risk

                The current AI Twin prediction is below the
                configured 70% decision threshold.

                **Recommended next steps:**

                **1. 👩 Continue appropriate prenatal care**

                Continue routine care and monitoring according
                to the treating healthcare professional.

                **2. 👶 Continue monitoring relevant fetal data**

                New fetal measurements can be incorporated
                when available.

                **3. 🧬 Update genetic information when available**

                New genetic information can change the
                Digital Twin's prediction.

                **4. 🔄 Update the Digital Twin**

                If new maternal, fetal, genetic or molecular
                information becomes available, the prediction
                can be recalculated.

                **5. 👨‍⚕️ Discuss clinical concerns**

                Any medical concern should be evaluated by
                a qualified healthcare professional.
                """
            )


        # ====================================================
        # DIGITAL TWIN UPDATE
        # ====================================================

        st.subheader(
            "🔄 How the Digital Twin Can Be Updated"
        )


        st.info(
            """
            The Digital Twin is designed as a patient-specific
            computational representation that can be updated
            when new information becomes available.

            **New maternal data**
            ↓

            **New fetal measurements**
            ↓

            **New genetic information**
            ↓

            **New molecular information**
            ↓

            **Updated AI prediction**
            ↓

            **Updated explanation**
            """
        )


        # ====================================================
        # WHAT CAN CHANGE THE PREDICTION?
        # ====================================================

        st.subheader(
            "📈 What Can Change the Prediction?"
        )


        st.write(
            "The prediction is generated from the combination "
            "of 33 input features. Therefore, adding or updating "
            "patient information can change the model output."
        )


        change_col1, change_col2, change_col3 = (
            st.columns(3)
        )


        with change_col1:

            st.markdown(
                "### 👩 Maternal"
            )

            st.write(
                "Glucose, HbA1c, BMI, blood pressure "
                "and gestational information."
            )


        with change_col2:

            st.markdown(
                "### 👶 Fetal"
            )

            st.write(
                "Fetal heart rate, movement, growth "
                "percentile and estimated weight."
            )


        with change_col3:

            st.markdown(
                "### 🧬 Genetic / Molecular"
            )

            st.write(
                "Genetic indicators and available "
                "gene-expression measurements."
            )


        # ====================================================
        # IMPORTANT EXPLANATION LIMITATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "⚠️ Important About the Explanation"
        )


        st.caption(
            "The feature influences shown above describe "
            "how the trained model responded to the entered "
            "values relative to its stored training-data "
            "baseline. They are model-based explanations and "
            "do not establish that a feature causes neonatal "
            "diabetes."
        )


        # ====================================================
        # TECHNICAL DETAILS - OPTIONAL
        # ====================================================

        with st.expander(
            "🔧 Technical Model Details"
        ):

            st.write(
                "Model type: Neural Network"
            )

            st.write(
                "Number of input features: 33"
            )

            st.write(
                "Decision threshold: 70%"
            )

            st.write(
                "Explanation method: "
                "reference-based feature perturbation"
            )

            st.write(
                "Reference values are used internally "
                "to estimate feature influence."
            )

            st.write(
                "The reference values are training-data "
                "baselines, not clinical normal ranges."
            )


        # ====================================================
        # IMPORTANT LIMITATION
        # ====================================================

        st.markdown("---")

        st.warning(
            """
            Important: This prediction is an AI-generated
            risk estimate from a research prototype.

            It is not a medical diagnosis and should not
            be used alone for clinical decision-making.
            """
        )


        st.caption(
            "The current prototype uses a synthetic training "
            "dataset. Therefore, the displayed probability "
            "should not be interpreted as clinically validated "
            "risk."
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.write(
            "Please check the model, scaler and feature files."
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Neonatal Diabetes AI Digital Twin • "
    "Research / Educational Prototype"
            )
