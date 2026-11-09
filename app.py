import torch
import torch.nn as nn
import streamlit as st

class ANN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

    def forward(self, x):
        return self.model(x)

import torch
import joblib
import pandas as pd
import __main__



model = torch.load(
    "Full_Model.pth",
    map_location="cpu",
    weights_only=False
)

model.eval()

scaler = joblib.load("scaler.pkl")

st.title('POWER PLANT PREDICTIOR')


T = st.number_input("Ambient Temperature (AT): ")
V = st.number_input("Exhaust Vacuum (V): ")
AP = st.number_input("Ambient Pressure (AP): ")
RH = st.number_input("Relative Humidity (RH): ")

if st.button("PREDICT"):
    features = pd.DataFrame(
    [[T, V, AP, RH]],
    columns=["AT", "V", "AP", "RH"]
    )

    scaled_features = scaler.transform(features)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)



    with torch.no_grad():
        prediction = model(input_tensor)
        st.write(f"Predicted Energy Output: {prediction.item():.2f} MW")
# print(f"\nPredicted Energy Output: {prediction.item():.2f} MW")


