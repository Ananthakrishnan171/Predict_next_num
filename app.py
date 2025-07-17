import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache_resource
def load_rnn_model():
    model = load_model("Rnn_sequence_model.h5")
    return model

model = load_rnn_model()

st.title("Predict Next Number in a Sequence (RNN Model)")

st.header("Input 3 Sequential Numbers")
col1, col2, col3 = st.columns(3)
num1 = col1.number_input("Number 1", value=1)
num2 = col2.number_input("Number 2", value=2)
num3 = col3.number_input("Number 3", value=3)

# Prepare input for prediction
input_data = np.array([[[num1], [num2], [num3]]], dtype=np.float32)
prediction = model.predict(input_data, verbose=0)
predicted_value = prediction[0][0]

st.success(f"Predicted Next Number: **{predicted_value:.2f}**")

st.header("Visualization on 1 to 100 Sequence (Training Set)")

# Prepare sequence data for visualization
sequence = np.array([i for i in range(1, 101)])
window_size = 3

X = []
y = []

for i in range(len(sequence) - window_size):
    X.append(sequence[i:i + window_size])
    y.append(sequence[i + window_size])

X = np.array(X).reshape((len(X), window_size, 1))
y = np.array(y)

# Predict on whole 1-100 sequence
y_pred = model.predict(X, verbose=0).flatten()

# Plotting
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(window_size, 100), y, label="Actual", linewidth=2)
ax.plot(range(window_size, 100), y_pred, label="Predicted", linestyle="--", linewidth=2)
ax.set_xlabel("Time Step")
ax.set_ylabel("Value")
ax.set_title("RNN Sequence Prediction (1 to 100)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
