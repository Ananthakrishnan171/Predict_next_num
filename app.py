import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Sequence Next Number Prediction", page_icon="🔢", layout="centered")

st.title("🔢 Predict the Next Number in a Sequence")
st.write("Provide 3 consecutive numbers. The model will predict the next one.")

model = load_model("Rnn_sequence_model.h5")

# Input columns
col1, col2, col3 = st.columns(3)
n1 = col1.number_input("First number", value=1, step=1)
n2 = col2.number_input("Second number", value=2, step=1)
n3 = col3.number_input("Third number", value=3, step=1)

if st.button("Predict"):
    input_seq = np.array([[n1, n2, n3]]).reshape((1, 3, 1))
    predicted = model.predict(input_seq, verbose=0)
    st.success(f"👉 Predicted Next Number: **{predicted[0][0]:.2f}**")

# Plotting the sequence prediction as in your original notebook
sequence = np.array([i for i in range(1, 101)])
window_size = 3
X = []
y = []

for i in range(len(sequence) - window_size):
    X.append(sequence[i:i + window_size])
    y.append(sequence[i + window_size])

X = np.array(X).reshape((len(X), window_size, 1))
y = np.array(y)

y_pred = model.predict(X, verbose=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(window_size, 100), y, label='Actual')
ax.plot(range(window_size, 100), y_pred.flatten(), linestyle='--', label='Predicted')
ax.set_title("RNN Sequence Prediction (1-100)")
ax.set_xlabel("Time Step")
ax.set_ylabel("Value")
ax.grid(True)
ax.legend()

st.pyplot(fig)
