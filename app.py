import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Skin Cancer Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        color: #0c2d48;  /* Darker text for better contrast */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #664d03;  /* Dark amber text */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #198754;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        color: #842029;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        model = joblib.load('skin_cancer_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("Model files not found. Please run the training notebook first.")
        return None, None, None

# Load model
model, scaler, label_encoders = load_model()

# Main title
st.markdown('<h1 class="main-header">🔬 Skin Cancer Detection System</h1>', unsafe_allow_html=True)

# Information about the system
st.markdown("""
<div class="info-box">
    <h3>📋 About This System</h3>
    <p>This AI-powered system helps in the early detection of skin cancer by analyzing various dermatological features. 
    The model has been trained on clinical data and uses machine learning to provide risk assessments.</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h3>⚠️ Important Disclaimer</h3>
    <p><strong>This system is for educational and informational purposes only.</strong> 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with a qualified healthcare provider for any health concerns.</p>
</div>
""", unsafe_allow_html=True)

if model is not None:
    # Sidebar for input
    st.sidebar.header("📝 Patient Information")
    
    # Create input fields
    with st.sidebar.form("patient_form"):
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        st.subheader("Skin Characteristics")
        skin_type = st.selectbox("Skin Type", [
            "Type I", "Type II", "Type III", "Type IV", "Type V", "Type VI"
        ])
        
        sun_exposure = st.selectbox("Sun Exposure", ["Low", "Moderate", "High"])
        
        st.subheader("Medical History")
        family_history = st.selectbox("Family History of Skin Cancer", ["No", "Yes"])
        
        st.subheader("Mole Characteristics")
        mole_count = st.number_input("Number of Moles", min_value=0, max_value=20, value=3)
        diameter_mm = st.number_input("Diameter (mm)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        
        st.subheader("Symptoms")
        itchiness = st.selectbox("Itchiness", ["No", "Yes"])
        bleeding = st.selectbox("Bleeding", ["No", "Yes"])
        
        st.subheader("ABCDE Criteria")
        asymmetry = st.selectbox("Asymmetry", ["No", "Yes"])
        border_irregularity = st.selectbox("Border Irregularity", ["No", "Yes"])
        color_variation = st.selectbox("Color Variation", ["No", "Yes"])
        evolution = st.selectbox("Evolution (Changes over time)", ["No", "Yes"])
        
        submit_button = st.form_submit_button("🔍 Analyze Risk")
    
    # Main content area
    if submit_button:
        # Prepare input data
        input_data = {
            'Age': age,
            'Gender': gender,
            'Skin_Type': skin_type,
            'Sun_Exposure': sun_exposure,
            'Family_History': family_history,
            'Mole_Count': mole_count,
            'Itchiness': itchiness,
            'Bleeding': bleeding,
            'Asymmetry': asymmetry,
            'Border_Irregularity': border_irregularity,
            'Color_Variation': color_variation,
            'Diameter_mm': diameter_mm,
            'Evolution': evolution
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        input_encoded = input_df.copy()
        
        for column in input_encoded.select_dtypes(include=['object']).columns:
            if column in label_encoders:
                # Handle unseen categories
                try:
                    input_encoded[column] = label_encoders[column].transform(input_encoded[column])
                except ValueError:
                    # If category not seen during training, use most frequent category
                    input_encoded[column] = 0
        
        # Scale the features
        input_scaled = scaler.transform(input_encoded)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h2 class="sub-header">📊 Risk Assessment</h2>', unsafe_allow_html=True)
            
            # Risk level
            risk_score = prediction_proba[1] * 100
            
            if prediction == 1:
                st.markdown(f"""
                <div class="danger-box">
                    <h3>⚠️ HIGH RISK</h3>
                    <p>The analysis indicates a <strong>HIGH RISK</strong> for skin cancer.</p>
                    <p><strong>Risk Score: {risk_score:.1f}%</strong></p>
                    <p>Please consult with a dermatologist immediately for professional evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ LOW RISK</h3>
                    <p>The analysis indicates a <strong>LOW RISK</strong> for skin cancer.</p>
                    <p><strong>Risk Score: {risk_score:.1f}%</strong></p>
                    <p>Continue regular skin monitoring and maintain good sun protection habits.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['Low Risk', 'High Risk'],
                    y=[prediction_proba[0] * 100, prediction_proba[1] * 100],
                    marker_color=['green', 'red']
                )
            ])
            
            fig.update_layout(
                title="Risk Assessment Probability",
                xaxis_title="Risk Level",
                yaxis_title="Probability (%)",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h2 class="sub-header">📋 Input Summary</h2>', unsafe_allow_html=True)
            
            # Display input summary
            summary_data = {
                'Parameter': list(input_data.keys()),
                'Value': list(input_data.values())
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Risk factors analysis
            st.markdown('<h3 class="sub-header">🔍 Risk Factor Analysis</h3>', unsafe_allow_html=True)
            
            risk_factors = []
            
            if age > 50:
                risk_factors.append("Age > 50 years")
            if family_history == "Yes":
                risk_factors.append("Family history of skin cancer")
            if sun_exposure == "High":
                risk_factors.append("High sun exposure")
            if mole_count > 5:
                risk_factors.append("Multiple moles")
            if diameter_mm > 6:
                risk_factors.append("Large diameter mole")
            if asymmetry == "Yes":
                risk_factors.append("Asymmetric mole")
            if border_irregularity == "Yes":
                risk_factors.append("Irregular borders")
            if color_variation == "Yes":
                risk_factors.append("Color variation")
            if evolution == "Yes":
                risk_factors.append("Changing mole")
            if bleeding == "Yes":
                risk_factors.append("Bleeding")
            if itchiness == "Yes":
                risk_factors.append("Itchiness")
            
            if risk_factors:
                st.markdown("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
            else:
                st.markdown("**No significant risk factors identified.**")
        
        # Recommendations
        st.markdown('<h2 class="sub-header">💡 Recommendations</h2>', unsafe_allow_html=True)
        
        if prediction == 1:
            recommendations = [
                "Schedule an appointment with a dermatologist immediately",
                "Avoid sun exposure and use high SPF sunscreen",
                "Monitor the mole for any changes",
                "Take photos of the mole to track changes",
                "Avoid picking or scratching the area",
                "Consider a full-body skin examination"
            ]
        else:
            recommendations = [
                "Continue regular self-examination of skin",
                "Use sunscreen with SPF 30+ daily",
                "Avoid excessive sun exposure, especially during peak hours",
                "Schedule routine dermatology check-ups",
                "Monitor any changes in existing moles",
                "Maintain a healthy lifestyle"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    else:
        # Show initial information
        st.markdown('<h2 class="sub-header">🚀 How to Use This System</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1. Enter Information
            Fill out the patient information form in the sidebar with accurate details about:
            - Basic demographics
            - Skin characteristics
            - Medical history
            - Mole properties
            """)
        
        with col2:
            st.markdown("""
            ### 2. Submit for Analysis
            Click the "Analyze Risk" button to get an AI-powered assessment based on:
            - Machine learning algorithms
            - Clinical data patterns
            - Risk factor analysis
            """)
        
        with col3:
            st.markdown("""
            ### 3. Review Results
            Get comprehensive results including:
            - Risk assessment score
            - Probability visualization
            - Personalized recommendations
            - Risk factor analysis
            """)
        
        # ABCDE criteria explanation
        st.markdown('<h2 class="sub-header">📚 Understanding the Input Criteria</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3 style="margin-top: 0;">📊 Features</h3>
                <ul>
                    <li><strong>Age</strong>: 18–90 years</li>
                    <li><strong>Gender</strong>: Male/Female</li>
                    <li><strong>Skin Type</strong>: Fitzpatrick I-VI</li>
                    <li><strong>Sun Exposure</strong>: Low/Moderate/High</li>
                    <li><strong>Family History</strong>: Yes/No</li>
                    <li><strong>Mole Count</strong>: Number of moles</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
                <h3 style="margin-top: 0;">🩺 Clinical Signs</h3>
                <ul>
                    <li><strong>Itchiness</strong>: Yes/No</li>
                    <li><strong>Bleeding</strong>: Yes/No</li>
                    <li><strong>Asymmetry</strong>: Yes/No</li>
                    <li><strong>Border Irregularity</strong>: Yes/No</li>
                    <li><strong>Color Variation</strong>: Yes/No</li>
                    <li><strong>Diameter</strong>: mm</li>
                    <li><strong>Evolution</strong>: Yes/No</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        
st.markdown("""
<div class="info-box">
    <h3 style="margin-top: 0;">🧴 Fitzpatrick Skin Types</h3>
    <p>The Fitzpatrick scale classifies skin response to sun exposure:</p>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
        <div>
            <p><strong>Type I</strong>: Always burns, never tans</p>
            <p><strong>Type II</strong>: Burns easily, tans minimally</p>
            <p><strong>Type III</strong>: Sometimes burns, gradually tans</p>
        </div>
        <div>
            <p><strong>Type IV</strong>: Rarely burns, tans well</p>
            <p><strong>Type V</strong>: Very rarely burns, tans easily</p>
            <p><strong>Type VI</strong>: Never burns, deeply pigmented</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Additional helper functions for the app
def create_gauge_chart(value, title):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    return fig


# Healthcare provider information
with st.expander("🏥 For Healthcare Providers"):
    st.markdown("""
    ### Clinical Integration Notes
    
    **Intended Use**: This tool is designed to assist healthcare providers in preliminary risk assessment and patient education.
    
    **Limitations**:
    - Not a diagnostic tool
    - Requires clinical validation
    - Should be used alongside professional examination
    - May not detect all types of skin cancer
    
    **Recommendations for Clinical Use**:
    1. Use as a screening support tool
    2. Always perform physical examination
    3. Consider patient history and risk factors
    4. Follow up with appropriate diagnostic procedures
    5. Document findings according to medical standards
    
    **Integration with EHR Systems**:
    - Can be integrated with electronic health records
    - Supports batch processing for population screening
    - Provides structured output for documentation
    """)

# Data privacy and security
with st.expander("🔒 Privacy and Security"):
    st.markdown("""
    ### Data Handling
    
    - **No Data Storage**: This application does not store any patient data
    - **Session-Based**: All inputs are processed in real-time and discarded
    - **Local Processing**: All computations happen locally
    - **HIPAA Considerations**: Ensure compliance with local healthcare regulations
    
    ### Security Best Practices
    - Deploy on secure servers with SSL/TLS
    - Implement user authentication if required
    - Regular security audits and updates
    - Compliance with healthcare data protection standards
    """)



# Contact and support information
st.markdown("---")
st.markdown("""
### 📞 Support and Contact

For medical questions, or feedback:
- Medical Questions: Consult with qualified healthcare providers
- Feedback: Reach out to us on our socials

**Emergency**: If you have urgent medical concerns, contact your healthcare provider immediately or call emergency services.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #ffffff; margin-top: 2rem;">
    <p>🔬 Skin Cancer Detection System | Built with ⚒️ by Himanshu and Vansh</p>
    <p><em>For educational purposes only. Not a substitute for professional medical advice.</em></p>
</div>
""", unsafe_allow_html=True)