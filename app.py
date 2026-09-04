import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#Load the train model
model=tf.keras.models.load_model('model.h5')

##load the encoder and scaler

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)


with open('one_hot_enc.pkl','rb') as file:
    one_hot_enc=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


##streamlit app
st.title('Customer churm prediction')

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
