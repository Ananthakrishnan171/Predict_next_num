import streamlit as st
import numpy as np
import pickle
from keras.models import model_from_json

# 🎯 Load Model from .pkl
with open('model.pkl', 'rb') as file:
    model_structure, model_weights = pickle.load(file)

model = model_from_json(model_structure)
model.set_weights(model_weights)
model.compile(optimizer='adam', loss='mse')

# 🌐 Streamlit App UI
st.set_page_config(page_title="Next Number Prediction", page_icon="🔢", layout="centered")

st.markdown(
    """
    <div style="text-align:center;">
        <h1 style='color:#3366ff;'>🔢 RNN Next Number Predictor</h1>
        <p style='font-size:18px;'>Enter 3 consecutive numbers, and I will predict the next number for you!</p>
    </div>
    """, unsafe_allow_html=True
)

# 📥 User Inputs with nice layout
col1, col2, col3 = st.columns(3)

with col1:
    n1 = st.number_input('Number 1', step=1, format="%d")
with col2:
    n2 = st.number_input('Number 2', step=1, format="%d")
with col3:
    n3 = st.number_input('Number 3', step=1, format="%d")

# 📤 Predict Button
if st.button("Predict Next Number 🚀"):
    sample_input = np.array([[[n1], [n2], [n3]]])
    prediction = model.predict(sample_input, verbose=0)
    st.success(f"✅ Predicted Next Number: **{float(prediction[0][0]):.2f}**")

# 🦶 Footer
st.markdown(
    """
    <hr>
    <div style='text-align:center; color:gray; font-size:12px;'>Powered by Streamlit & Keras</div>
    """,
    unsafe_allow_html=True
)
