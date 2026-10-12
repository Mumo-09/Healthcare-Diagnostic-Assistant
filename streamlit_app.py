
import streamlit as st
import uuid

from modules.agent import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner import TreatmentPlanner

st.set_page_config(
    page_title="Intelligent Healthcare Diagnostic Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Intelligent Healthcare Diagnostic Assistant")
st.write("AI-powered healthcare diagnostic prototype")

st.warning(
    "Educational prototype only. This application does not provide "
    "confirmed medical diagnoses or replace professional medical care."
)

st.header("Patient Information")

with st.form("patient_form"):
    patient_id = st.text_input(
        "Patient ID",
        value=f"P-{uuid.uuid4().hex[:6].upper()}"
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            value=20
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=30.0,
            max_value=45.0,
            value=37.0,
            step=0.1
        )

        heart_rate = st.number_input(
            "Heart rate (beats/min)",
            min_value=20,
            max_value=250,
            value=80
        )

    with col2:
        blood_pressure = st.text_input(
            "Blood pressure",
            placeholder="e.g. 120/80"
        )

    st.subheader("Symptoms")

    symptoms_options = [
    "fever",
    "high_fever",
    "cough",
    "fatigue",
    "loss_of_smell",
    "headache",
    "difficulty_breathing",
    "shortness_of_breath",
    "ear_pain",
    "chest_pain",
    "sore_throat",
    "runny_nose",
    "nausea",
    "vomiting",
    "diarrhea",
    "abdominal_pain",
    "dizziness",
    "joint_pain",
    "muscle_pain",
    "rash",
    "itching",
    "night_sweats",
    "weight_loss",
    "stiff_neck",
    "light_sensitivity",
    "frequent_urination",
    "excessive_thirst",
    "blurred_vision",
    "sweating",
    "back_pain",
    "toothache",
    "ear_discharge",
    "chills",
    "weakness",
    "loss_of_appetite",
    "constipation",
    "bloating",
    "difficulty_swallowing",
    "palpitations",
    "fainting",
    "confusion",
    "seizures",
    "swollen_lymph_nodes",
    "eye_pain",
    "red_eyes",
    "skin_pain",
    "frequent_urination",
    "painful_urination"
]
    selected_symptoms = st.multiselect(
        "Select all reported symptoms",
        options=symptoms_options,
        format_func=lambda symptom: symptom.replace("_", " ").title()
    )

    custom_symptoms_text = st.text_input(
        "Other symptoms",
        placeholder="Enter additional symptoms separated by commas"
    )

    custom_symptoms = [
        symptom.strip().lower().replace(" ", "_")
        for symptom in custom_symptoms_text.split(",")
        if symptom.strip()
    ]

    symptoms = list(dict.fromkeys(selected_symptoms + custom_symptoms))

    if symptoms:
        st.write(
            "**Symptoms recorded:**",
            ", ".join(
                symptom.replace("_", " ").title()
                for symptom in symptoms
            )
        )

    submitted = st.form_submit_button(
        "Analyze Symptoms",
        type="primary"
    )


if submitted:
    if not symptoms:
        st.error("Please select at least one symptom.")

    else:
        try:
            patient = PatientPercept(
                patient_id=patient_id.strip() or "Unknown",
                symptoms=symptoms,
                age=int(age),
                temperature=float(temperature),
                heart_rate=int(heart_rate),
                blood_pressure=blood_pressure.strip() or "Not provided"
            )

            agent = HealthcareDiagnosticAgent()

            agent.register_module(
                "Knowledge Base",
                MedicalKnowledgeBase()
            )

            agent.register_module(
                "Bayesian Network",
                SimpleBayesianDiagnostics()
            )
            
            agent.register_module(
                "Machine Learning",
                MLDiagnosticClassifier()
            )
            
            
            agent.register_module(
                "Neural Network",
                NeuralDiagnosticModel()
            )

            agent.register_module(
                "Fuzzy Logic", 
                FuzzySeverityAssessor()
            ) 
            
            agent.register_module(
                "Treatment Planner",
                TreatmentPlanner()
            )
            results = agent.run(patient)
                
            st.divider()
            st.header("Analysis Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Urgency", results["urgency"])

            with col2:
                st.metric(
                    "Confidence",
                    f'{results["confidence"]:.1%}'
                )

            with col3:
                st.metric("Next Action", results["next_action"])

            st.subheader("Diagnostic Summary")

            st.write(
                results["diagnosis"].replace("_", " ").title()
            )

            st.subheader("Recommendations")

            for recommendation in results["recommendations"]:
                st.write(f"- {recommendation}")

            with st.expander("Patient Details"):
                st.write("Patient ID:", results["patient_id"])
                st.write("Age:", patient.age)
                st.write("Temperature:", patient.temperature, "°C")
                st.write("Heart rate:", patient.heart_rate, "bpm")
                st.write("Blood pressure:", patient.blood_pressure)

                st.write(
                    "Symptoms:",
                    ", ".join(
                        symptom.replace("_", " ").title()
                        for symptom in results["symptoms"]
                    )
                )

                st.write("Timestamp:", results["timestamp"])
            st.subheader("Individual AI Module Results")

            module_results = agent.think()

            for module_name, module_result in module_results.items():
                st.write(f"### {module_name}")
                st.json(module_result)
            st.info(
                "This is an educational prototype. Its results are not "
                "clinically validated and must not replace assessment "
                "by a qualified healthcare professional."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")
            st.caption(
                "Check the terminal for the full error message."
            )