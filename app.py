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
# TITLE
# ============================================================

st.title("🧬 Neonatal Diabetes AI Twin")
st.markdown(
    """
    ### AI-based early risk assessment prototype for Neonatal Diabetes Mellitus

    This system combines maternal, fetal, genetic and molecular
    information to estimate the predicted risk of neonatal diabetes.
    """
)

st.divider()


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = "ndm_clean_neural_network.keras"
SCALER_FILE = "ndm_clean_33_feature_scaler.pkl"
FEATURE_FILE = "ndm_feature_names.pkl"
REFERENCE_FILE = "ndm_reference_values.pkl"


# ============================================================
# EXPECTED FEATURES
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
# USER-FRIENDLY FEATURE NAMES
# ============================================================

FEATURE_DISPLAY_NAMES = {

    "Maternal_Age_years":
        "Maternal Age",

    "Maternal_BMI":
        "Maternal BMI",

    "Maternal_Systolic_BP":
        "Maternal Systolic Blood Pressure",

    "Maternal_Diastolic_BP":
        "Maternal Diastolic Blood Pressure",

    "Maternal_Fasting_Glucose_mg_dL":
        "Maternal Fasting Glucose",

    "Maternal_HbA1c_percent":
        "Maternal HbA1c",

    "Gestational_Age_weeks":
        "Gestational Age",

    "Fetal_Heart_Rate_bpm":
        "Fetal Heart Rate",

    "Fetal_Movement_per_hour":
        "Fetal Movement",

    "Ultrasound_Growth_Percentile":
        "Ultrasound Growth Percentile",

    "Estimated_Fetal_Weight_g":
        "Estimated Fetal Weight",

    "Family_History_Diabetes":
        "Family History of Diabetes",

    "Consanguinity":
        "Consanguinity",

    "Previous_Gestational_Diabetes":
        "Previous Gestational Diabetes",

    "Maternal_Autoimmune_History":
        "Maternal Autoimmune History",

    "KCNJ11_Variant":
        "KCNJ11 Genetic Variant",

    "ABCC8_Variant":
        "ABCC8 Genetic Variant",

    "INS_Variant":
        "INS Genetic Variant",

    "Chr6q24_Abnormality":
        "Chromosome 6q24 Abnormality",

    "GCK_Variant":
        "GCK Genetic Variant",

    "HNF1B_Variant":
        "HNF1B Genetic Variant",

    "GATA6_Variant":
        "GATA6 Genetic Variant",

    "GLIS3_Variant":
        "GLIS3 Genetic Variant",

    "INS_expr":
        "INS Gene Expression",

    "PDX1_expr":
        "PDX1 Gene Expression",

    "NKX6_1_expr":
        "NKX6-1 Gene Expression",

    "MAFA_expr":
        "MAFA Gene Expression",

    "GCK_expr":
        "GCK Gene Expression",

    "SLC2A2_expr":
        "SLC2A2 Gene Expression",

    "ABCC8_expr":
        "ABCC8 Gene Expression",

    "KCNJ11_expr":
        "KCNJ11 Gene Expression",

    "NEUROD1_expr":
        "NEUROD1 Gene Expression",

    "HNF1B_expr":
        "HNF1B Gene Expression"
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
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:

    st.error("❌ Required model files are missing.")

    st.write("Please make sure these files are present:")

    for file in missing_files:
        st.write(f"- {file}")

    st.stop()


# ============================================================
# LOAD MODEL AND SUPPORTING FILES
# ============================================================

@st.cache_resource
def load_model_files():

    model = tf.keras.models.load_model(MODEL_FILE)

    scaler = joblib.load(SCALER_FILE)

    feature_names = joblib.load(FEATURE_FILE)

    reference_values = joblib.load(REFERENCE_FILE)

    return model, scaler, feature_names, reference_values


model, scaler, feature_names, reference_values = load_model_files()


# ============================================================
# FEATURE VALIDATION
# ============================================================

if len(feature_names) != 33:

    st.error(
        f"❌ Feature mismatch: expected 33 features, "
        f"but found {len(feature_names)}."
    )

    st.stop()


# ============================================================
# REFERENCE VALUE FUNCTION
# ============================================================

def get_reference_value(feature):

    if isinstance(reference_values, dict):

        return reference_values.get(
            feature,
            0
        )

    return 0


# ============================================================
# YES / NO INPUT FUNCTION
# ============================================================

def yes_no_input(label, key):

    value = st.selectbox(
        label,
        ["No", "Yes"],
        key=key
    )

    if value == "Yes":
        return 1

    return 0


# ============================================================
# GENETIC VARIANT INPUT
# ============================================================

def variant_input(label, key):

    value = st.selectbox(
        label,
        [
            "No known variant",
            "Variant present"
        ],
        key=key
    )

    if value == "Variant present":
        return 1

    return 0


# ============================================================
# CALCULATE FEATURE CONTRIBUTIONS
# ============================================================

def calculate_patient_contributions(
    input_data,
    original_probability
):

    contributions = []

    original_array = np.array(
        input_data,
        dtype=float
    )

    for i, feature in enumerate(EXPECTED_FEATURES):

        modified_array = original_array.copy()

        reference_value = get_reference_value(
            feature
        )

        modified_array[i] = reference_value

        modified_scaled = scaler.transform(
            modified_array.reshape(1, -1)
        )

        modified_prediction = model.predict(
            modified_scaled,
            verbose=0
        )[0][0]

        contribution = (
            original_probability
            - float(modified_prediction)
        )

        contributions.append({
            "feature": feature,
            "display_name":
                FEATURE_DISPLAY_NAMES.get(
                    feature,
                    feature
                ),
            "patient_value":
                original_array[i],
            "contribution":
                contribution
        })

    return contributions


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("👩 Maternal Information")

col1, col2, col3 = st.columns(3)

with col1:

    maternal_age = st.number_input(
        "Maternal Age (years)",
        min_value=15.0,
        max_value=60.0,
        value=28.0
    )

with col2:

    maternal_bmi = st.number_input(
        "Maternal BMI",
        min_value=10.0,
        max_value=60.0,
        value=23.0
    )

with col3:

    maternal_sbp = st.number_input(
        "Maternal Systolic BP",
        min_value=70.0,
        max_value=220.0,
        value=120.0
    )


col1, col2, col3 = st.columns(3)

with col1:

    maternal_dbp = st.number_input(
        "Maternal Diastolic BP",
        min_value=40.0,
        max_value=140.0,
        value=80.0
    )

with col2:

    fasting_glucose = st.number_input(
        "Maternal Fasting Glucose (mg/dL)",
        min_value=40.0,
        max_value=400.0,
        value=90.0
    )

with col3:

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


st.divider()


# ============================================================
# FETAL INFORMATION
# ============================================================

st.header("👶 Fetal Information")

col1, col2, col3 = st.columns(3)

with col1:

    fetal_heart_rate = st.number_input(
        "Fetal Heart Rate (bpm)",
        min_value=80.0,
        max_value=220.0,
        value=140.0
    )

with col2:

    fetal_movement = st.number_input(
        "Fetal Movement (per hour)",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

with col3:

    growth_percentile = st.number_input(
        "Ultrasound Growth Percentile",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )


estimated_weight = st.number_input(
    "Estimated Fetal Weight (g)",
    min_value=200.0,
    max_value=6000.0,
    value=3000.0
)


st.divider()


# ============================================================
# MATERNAL / FAMILY HISTORY
# ============================================================

st.header("🧬 Family and Medical History")

col1, col2 = st.columns(2)

with col1:

    family_history = yes_no_input(
        "Family History of Diabetes",
        "family_history"
    )

    consanguinity = yes_no_input(
        "Consanguinity",
        "consanguinity"
    )

with col2:

    previous_gdm = yes_no_input(
        "Previous Gestational Diabetes",
        "previous_gdm"
    )

    autoimmune_history = yes_no_input(
        "Maternal Autoimmune History",
        "autoimmune_history"
    )


st.divider()


# ============================================================
# GENETIC INFORMATION
# ============================================================

st.header("🧬 Genetic Information")

col1, col2, col3 = st.columns(3)

with col1:

    kcnj11_variant = variant_input(
        "KCNJ11 Variant",
        "kcnj11_variant"
    )

    abcc8_variant = variant_input(
        "ABCC8 Variant",
        "abcc8_variant"
    )

    ins_variant = variant_input(
        "INS Variant",
        "ins_variant"
    )

with col2:

    chr6q24 = variant_input(
        "Chromosome 6q24 Abnormality",
        "chr6q24"
    )

    gck_variant = variant_input(
        "GCK Variant",
        "gck_variant"
    )

    hnf1b_variant = variant_input(
        "HNF1B Variant",
        "hnf1b_variant"
    )

with col3:

    gata6_variant = variant_input(
        "GATA6 Variant",
        "gata6_variant"
    )

    glis3_variant = variant_input(
        "GLIS3 Variant",
        "glis3_variant"
    )


st.divider()


# ============================================================
# GENE EXPRESSION
# ============================================================

st.header("🔬 Molecular / Gene Expression Information")

st.info(
    "Enter gene-expression values if molecular information "
    "is available. Otherwise, the default values can be used."
)

col1, col2, col3 = st.columns(3)

with col1:

    ins_expr = st.number_input(
        "INS Expression",
        value=1.0
    )

    pdx1_expr = st.number_input(
        "PDX1 Expression",
        value=1.0
    )

    nkx6_1_expr = st.number_input(
        "NKX6-1 Expression",
        value=1.0
    )

with col2:

    mafa_expr = st.number_input(
        "MAFA Expression",
        value=1.0
    )

    gck_expr = st.number_input(
        "GCK Expression",
        value=1.0
    )

    slc2a2_expr = st.number_input(
        "SLC2A2 Expression",
        value=1.0
    )

with col3:

    abcc8_expr = st.number_input(
        "ABCC8 Expression",
        value=1.0
    )

    kcnj11_expr = st.number_input(
        "KCNJ11 Expression",
        value=1.0
    )

    neurod1_expr = st.number_input(
        "NEUROD1 Expression",
        value=1.0
    )


hnf1b_expr = st.number_input(
    "HNF1B Expression",
    value=1.0
)


st.divider()


# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = [

    maternal_age,

    maternal_bmi,

    maternal_sbp,

    maternal_dbp,

    fasting_glucose,

    hba1c,

    gestational_age,

    fetal_heart_rate,

    fetal_movement,

    growth_percentile,

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


# ============================================================
# VERIFY INPUT LENGTH
# ============================================================

if len(input_data) != 33:

    st.error(
        f"❌ Input error: {len(input_data)} features found. "
        "Expected 33."
    )

    st.stop()


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.header("🔮 AI Twin Prediction")

predict_button = st.button(
    "🔮 Predict Neonatal Diabetes Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        input_array = np.array(
            input_data,
            dtype=float
        ).reshape(1, -1)

        # Scale input
        input_scaled = scaler.transform(
            input_array
        )

        # Model prediction
        prediction = model.predict(
            input_scaled,
            verbose=0
        )[0][0]

        probability = float(prediction)

        risk_percentage = probability * 100


        # ====================================================
        # RISK CLASSIFICATION
        # ====================================================

        if probability >= 0.70:

            risk_level = "HIGH RISK"

        elif probability >= 0.40:

            risk_level = "MODERATE RISK"

        else:

            risk_level = "LOWER RISK"


        # ====================================================
        # MAIN RESULT
        # ====================================================

        st.subheader("🎯 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Risk",
                f"{risk_percentage:.2f}%"
            )

        with col2:

            st.metric(
                "Risk Level",
                risk_level
            )

        with col3:

            st.metric(
                "Decision Threshold",
                "70%"
            )


        # ====================================================
        # RESULT MESSAGE
        # ====================================================

        if probability >= 0.70:

            st.error(
                "⚠️ The AI Twin estimates a higher predicted "
                "risk based on the information entered."
            )

        elif probability >= 0.40:

            st.warning(
                "⚠️ The AI Twin estimates an intermediate "
                "predicted risk based on the information entered."
            )

        else:

            st.success(
                "✅ The AI Twin estimates a lower predicted "
                "risk based on the information entered."
            )


        st.divider()


        # ====================================================
        # OVERALL PREDICTION
        # ====================================================

        st.header("📊 Overall Prediction")

        if probability >= 0.70:

            st.markdown(
                """
                ### 🔴 Higher Predicted Risk

                The AI Twin has produced a risk probability above
                the configured 70% threshold.

                This means the current combination of maternal,
                fetal, genetic and molecular inputs is associated
                with a higher model-predicted risk.
                """
            )

        elif probability >= 0.40:

            st.markdown(
                """
                ### 🟠 Intermediate Predicted Risk

                The AI Twin has produced a risk probability between
                the lower-risk and higher-risk ranges.

                Additional clinical assessment and appropriate
                follow-up may be considered.
                """
            )

        else:

            st.markdown(
                """
                ### 🟢 Lower Predicted Risk

                The AI Twin has produced a risk probability below
                the configured 70% threshold.

                Based on the entered information, the model does
                not classify the case as higher predicted risk.
                """
            )


        # ====================================================
        # RISK INTERPRETATION
        # ====================================================

        st.header("🧠 Risk Interpretation")

        st.write(
            f"""
            The AI Twin generated a predicted probability of
            **{risk_percentage:.2f}%**.

            The configured decision threshold is **70%**.

            Therefore, the current prediction is classified as
            **{risk_level}**.
            """
        )


        # ====================================================
        # FEATURE CONTRIBUTIONS
        # ====================================================

        contributions = calculate_patient_contributions(
            input_data,
            probability
        )


        # Sort features by absolute influence

        contributions_sorted = sorted(
            contributions,
            key=lambda x: abs(x["contribution"]),
            reverse=True
        )


        top_features = contributions_sorted[:5]


        # ====================================================
        # WHY DID AI TWIN GIVE THIS RESULT?
        # ====================================================

        st.header("🔍 Why did the AI Twin give this result?")

        if probability >= 0.70:

            st.write(
                """
                The model produced a higher predicted risk because
                the overall combination of the entered features
                pushed the neural network's output above the
                configured risk threshold.
                """
            )

        elif probability >= 0.40:

            st.write(
                """
                The model produced an intermediate predicted risk
                because the combined input pattern produced a
                probability between the lower and higher risk
                ranges.
                """
            )

        else:

            st.write(
                """
                The model produced a lower predicted risk because
                the overall combination of the entered features
                kept the neural network's output below the
                configured risk threshold.
                """
            )


        # ====================================================
        # MAIN FACTORS
        # ====================================================

        st.subheader(
            "🔎 Main Factors Influencing the Prediction"
        )

        for item in top_features:

            feature_name = item["display_name"]

            contribution = item["contribution"]


            if contribution > 0:

                st.write(
                    f"🔺 **{feature_name}** "
                    f"increased the model's predicted risk."
                )

            elif contribution < 0:

                st.write(
                    f"🔻 **{feature_name}** "
                    f"decreased the model's predicted risk."
                )

            else:

                st.write(
                    f"⚪ **{feature_name}** "
                    f"had little influence on this prediction."
                )


        # ====================================================
        # OVERALL EXPLANATION
        # ====================================================

        st.subheader("📝 Overall Explanation")

        positive_features = [
            item["display_name"]
            for item in contributions
            if item["contribution"] > 0
        ]

        negative_features = [
            item["display_name"]
            for item in contributions
            if item["contribution"] < 0
        ]


        if probability >= 0.70:

            if positive_features:

                st.write(
                    "The higher prediction was mainly influenced "
                    "by the combination of features that increased "
                    "the model's predicted risk."
                )

                st.write(
                    "**Examples of risk-increasing factors:** "
                    + ", ".join(
                        positive_features[:3]
                    )
                )

            else:

                st.write(
                    "The model produced a higher prediction from "
                    "the combined feature pattern."
                )


        elif probability >= 0.40:

            st.write(
                "The prediction is intermediate because the "
                "different input features produced a mixed effect "
                "on the model output."
            )

            if positive_features:

                st.write(
                    "**Factors increasing model risk:** "
                    + ", ".join(
                        positive_features[:3]
                    )
                )

            if negative_features:

                st.write(
                    "**Factors decreasing model risk:** "
                    + ", ".join(
                        negative_features[:3]
                    )
                )


        else:

            if negative_features:

                st.write(
                    "The lower prediction was supported by the "
                    "combination of features that decreased the "
                    "model's predicted risk."
                )

                st.write(
                    "**Examples of risk-reducing factors:** "
                    + ", ".join(
                        negative_features[:3]
                    )
                )

            else:

                st.write(
                    "The model produced a lower prediction based "
                    "on the overall combination of entered features."
                )


        # ====================================================
        # WHAT SHOULD HAPPEN NEXT?
        # ====================================================

        st.header("🩺 What Should Happen Next?")


        if probability >= 0.70:

            st.warning(
                """
                ### Higher predicted risk

                The model indicates that this case deserves
                closer clinical attention.

                Possible next steps include:

                • Review the maternal and fetal information.

                • Consider appropriate genetic evaluation when
                  clinically indicated.

                • Discuss the result with an obstetrician,
                  pediatrician, endocrinologist or genetic
                  specialist.

                • Continue appropriate prenatal monitoring.

                • Do not use this AI prediction alone to make
                  a medical diagnosis or treatment decision.
                """
            )


        elif probability >= 0.40:

            st.info(
                """
                ### Intermediate predicted risk

                The model suggests that additional attention may
                be useful.

                Possible next steps include:

                • Review the entered clinical information.

                • Continue routine clinical monitoring.

                • Consider further evaluation if there are
                  additional clinical or family-history concerns.

                • Discuss any concerns with the appropriate
                  healthcare professional.

                • Do not use the AI prediction as a diagnosis.
                """
            )


        else:

            st.success(
                """
                ### Lower predicted risk

                The AI Twin does not classify the current input
                as higher predicted risk.

                Possible next steps include:

                • Continue appropriate prenatal care.

                • Continue routine maternal and fetal monitoring.

                • Discuss any new symptoms or clinical concerns
                  with a healthcare professional.

                • Remember that a lower model prediction does not
                  completely rule out disease.
                """
            )


        # ====================================================
        # VISUAL PREDICTION GRAPH
        # ====================================================

        st.header("📈 Prediction Visualization")

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.bar(
            ["Predicted Risk"],
            [risk_percentage]
        )

        ax.axhline(
            y=70,
            linestyle="--",
            label="70% Threshold"
        )

        ax.set_ylim(
            0,
            100
        )

        ax.set_ylabel(
            "Risk Probability (%)"
        )

        ax.set_title(
            "AI Twin Neonatal Diabetes Risk Prediction"
        )

        ax.legend()

        st.pyplot(fig)

        plt.close(fig)


        # ====================================================
        # DIGITAL TWIN PROFILE
        # ====================================================

        st.header("👤 Digital Twin Profile")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("👩 Maternal Profile")

            st.write(
                f"**Age:** {maternal_age} years"
            )

            st.write(
                f"**BMI:** {maternal_bmi}"
            )

            st.write(
                f"**Systolic BP:** {maternal_sbp} mmHg"
            )

            st.write(
                f"**Diastolic BP:** {maternal_dbp} mmHg"
            )

            st.write(
                f"**Fasting Glucose:** "
                f"{fasting_glucose} mg/dL"
            )

            st.write(
                f"**HbA1c:** {hba1c}%"
            )

        with col2:

            st.subheader("👶 Fetal Profile")

            st.write(
                f"**Gestational Age:** "
                f"{gestational_age} weeks"
            )

            st.write(
                f"**Heart Rate:** "
                f"{fetal_heart_rate} bpm"
            )

            st.write(
                f"**Fetal Movement:** "
                f"{fetal_movement}/hour"
            )

            st.write(
                f"**Growth Percentile:** "
                f"{growth_percentile}"
            )

            st.write(
                f"**Estimated Weight:** "
                f"{estimated_weight} g"
            )


        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🧬 Genetic Profile")

            genetic_status = []

            if kcnj11_variant:
                genetic_status.append("KCNJ11")

            if abcc8_variant:
                genetic_status.append("ABCC8")

            if ins_variant:
                genetic_status.append("INS")

            if chr6q24:
                genetic_status.append("Chr6q24")

            if gck_variant:
                genetic_status.append("GCK")

            if hnf1b_variant:
                genetic_status.append("HNF1B")

            if gata6_variant:
                genetic_status.append("GATA6")

            if glis3_variant:
                genetic_status.append("GLIS3")


            if genetic_status:

                st.write(
                    "**Variants reported:** "
                    + ", ".join(
                        genetic_status
                    )
                )

            else:

                st.write(
                    "No selected genetic variants."
                )


        with col2:

            st.subheader("🔬 Molecular Profile")

            st.write(
                "Gene-expression features included:"
            )

            st.write(
                "INS, PDX1, NKX6-1, MAFA, GCK, "
                "SLC2A2, ABCC8, KCNJ11, "
                "NEUROD1 and HNF1B"
            )


        # ====================================================
        # LIMITATION
        # ====================================================

        st.divider()

        st.warning(
            """
            ⚠️ **Important Research Limitation**

            This is a research and educational prototype.

            The current model was developed using a synthetic
            demonstration dataset and has not been clinically
            validated.

            The displayed percentage is a **model-predicted risk
            probability**, not clinical accuracy.

            This system must not be used as a replacement for
            professional medical diagnosis, genetic testing,
            prenatal care or treatment decisions.
            """
        )


        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        with st.expander("🔧 Technical Details"):

            st.write(
                "**Model:** Neural Network"
            )

            st.write(
                "**Input Features:** 33"
            )

            st.write(
                "**Feature Groups:** "
                "Maternal + Fetal + Genetic + Molecular"
            )

            st.write(
                "**Scaler:** Saved training-data scaler"
            )

            st.write(
                "**Explanation Method:** "
                "One-feature-at-a-time reference replacement"
            )

            st.write(
                "**Decision Threshold:** 70%"
            )

            st.write(
                "**Target:** Neonatal Diabetes Risk"
            )


        # ====================================================
        # FOOTER
        # ====================================================

        st.divider()

        st.caption(
            "Neonatal Diabetes AI Twin | "
            "Deep Learning Research Project"
        )


    except Exception as e:

        st.error(
            "❌ An error occurred while generating "
            "the prediction."
        )

        st.exception(e)
