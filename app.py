import streamlit as st
import pickle
import pandas as pd

st.title("💎 Diamond Price Predictor")

with open("diamond_knn_model.pkl", "rb") as f:
    model = pickle.load(f)

carat = st.number_input("Carat", 0.2, 5.0)
depth = st.number_input("Depth")
table = st.number_input("Table")
x = st.number_input("X")
y = st.number_input("Y")
z = st.number_input("Z")
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity = st.selectbox("Clarity", ["IF","VVS1","VVS2","VS1","VS2","SI1","SI2","I1"])

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "carat": carat, "depth": depth, "table": table,
        "x": x, "y": y, "z": z,
        "cut": cut, "color": color, "clarity": clarity
    }])

    price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ${price:.2f}")
