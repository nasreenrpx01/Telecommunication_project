import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# CSS for background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://www.pi.exchange/hubfs/Blog%20images/Headers%20%28compressed%29/Predicting%20Customer%20Churn%20and%20Next%20Best%20Offer%20in%20Telecommunications.jpg");
    background-size: cover;
    background-position: center;
}
</style>
'''

# Apply CSS for background
st.markdown(page_bg_img, unsafe_allow_html=True)

# Application title
st.title("Telecommunication Customer Churn Prediction")
st.write("Predict whether a customer will churn based on their usage and plan details.")


# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')  # Replace with your actual model file name

model = load_model()


# Allow user to upload a data file
st.sidebar.header("Upload Data File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['xlsx'])

def load_sample_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    else:
        st.warning("Please upload a valid data file.")
        st.stop()
# Load data based on uploaded file
sample_data = load_sample_data(uploaded_file)

st.sidebar.subheader("Input Parameters") 

# Define encoders outside user input for consistency
state_encoder = LabelEncoder()
voice_plan_encoder = LabelEncoder()
intl_plan_encoder = LabelEncoder()

# Updated states list with CA as the selected state
states = ['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI',
       'IA', 'MT', 'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC',
       'NE', 'WY', 'HI', 'IL', 'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR',
       'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM', 'NV', 'DC',
       'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND']

state_encoder.fit(states)
voice_plan_encoder.fit(['Yes', 'No'])
intl_plan_encoder.fit(['Yes', 'No'])

# User inputs for prediction
def user_input_features():
    area_code = st.sidebar.selectbox('Area Code', ['415', '408', '510'])  # Area code updated
    state = st.sidebar.selectbox('State', states)
    voice_plan = st.sidebar.selectbox('Voice Plan', ['Yes', 'No'])
    intl_plan = st.sidebar.selectbox('International Plan', ['Yes', 'No'])
    account_length = st.sidebar.number_input('Account Length', 1, 243, 100)
    intl_mins = st.sidebar.number_input('International Minutes', 0, 20, 0)
    intl_calls = st.sidebar.number_input('International Calls', 0, 20, 0)
    day_mins = st.sidebar.number_input('Day Minutes', 0, 351, 0)
    day_calls = st.sidebar.number_input('Day Calls', 0, 165, 0)
    eve_mins = st.sidebar.number_input('Evening Minutes', 0, 363, 0)
    eve_calls = st.sidebar.number_input('Evening Calls', 0, 170, 0)
    night_mins = st.sidebar.number_input('Night Minutes', 0, 395, 0)
    night_calls = st.sidebar.number_input('Night Calls', 0, 175, 0)
    customer_calls = st.sidebar.number_input('Customer Calls', 0, 9, 0)
    
    # Encode inputs for model prediction
    state_encoded = state_encoder.transform([state])[0]
    voice_plan_encoded = voice_plan_encoder.transform([voice_plan])[0]
    intl_plan_encoded = intl_plan_encoder.transform([intl_plan])[0]
    area_code_encoded = {'415': 0, '408': 1, '510': 2}.get(area_code, 0)
    
    # Prepare the data dictionary with required columns (using human-readable values)
    data = {
        'state': state,  # Keep the original state for preview
        'area.code': area_code,  # Keep original area code for preview
        'account.length': account_length,
        'voice.plan': voice_plan,  # Keep the original values for preview
        'intl.plan': intl_plan,  # Keep the original values for preview
        'intl.mins': intl_mins,
        'intl.calls': intl_calls,
        'day.mins': day_mins,
        'day.calls': day_calls,
        'eve.mins': eve_mins,
        'eve.calls': eve_calls,
        'night.mins': night_mins,
        'night.calls': night_calls,
        'customer.calls': customer_calls
    }

    # Prepare the input DataFrame for model prediction
    input_df = pd.DataFrame([{
        'state': state_encoded,
        'area.code': area_code_encoded,
        'account.length': account_length,
        'voice.plan': voice_plan_encoded,
        'intl.plan': intl_plan_encoded,
        'intl.mins': intl_mins,
        'intl.calls': intl_calls,
        'day.mins': day_mins,
        'day.calls': day_calls,
        'eve.mins': eve_mins,
        'eve.calls': eve_calls,
        'night.mins': night_mins,
        'night.calls': night_calls,
        'customer.calls': customer_calls
    }])

    return input_df, data

# Get the input DataFrame and the parameter data for display
input_df, input_data = user_input_features()

# Display input parameters on the sidebar
st.subheader("Preview of Input Parameters")
st.write(input_data)

# Display the DataFrame to verify the input
st.write("### Input Dataframe Preview")
st.write(input_df)

# Prediction logic
if st.button("Predict Churn"):
    try:
        # Make the prediction using the encoded input
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.error("The customer is likely to churn.")
        else:
            st.success("The customer is unlikely to churn.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("Developed by Nasreen Fatima")
