import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import SimpleRNN, Dense

# Prepare data
sequence = np.array([i for i in range(1, 101)])
window_size = 3
X = []
y = []

for i in range(len(sequence) - window_size):
    X.append(sequence[i:i + window_size])
    y.append(sequence[i + window_size])

X = np.array(X).reshape((len(X), window_size, 1))
y = np.array(y)

# Build or Load model
@st.cache_resource
def get_model():
    try:
        model = load_model("Rnn_sequence_model.h5")
    except:
        model = Sequential()
        model.add(SimpleRNN(50, activation='relu', input_shape=(window_size, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=500, verbose=0)
        model.save("rnn_sequence_model.h5")
    return model

model = get_model()

# Streamlit UI
st.title("Next Number Prediction using RNN")
st.write("Enter any **3 consecutive numbers**:")

col1, col2, col3 = st.columns(3)
num1 = col1.number_input('Number 1', value=1)
num2 = col2.number_input('Number 2', value=2)
num3 = col3.number_input('Number 3', value=3)

input_data = np.array([[num1, num2, num3]]).reshape((1, window_size, 1))
predicted_value = model.predict(input_data, verbose=0)[0][0]

st.success(f"Predicted Next Number: **{predicted_value:.2f}**")

# Show plot
st.header("Visualization (1 to 100 Prediction)")
y_pred = model.predict(X, verbose=0).flatten()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(window_size, 100), y, label='Actual', linewidth=2)
ax.plot(range(window_size, 100), y_pred, label='Predicted', linestyle='--', linewidth=2)
ax.set_xlabel("Time Step")
ax.set_ylabel("Value")
ax.set_title("RNN Sequence Prediction (1 to 100)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
