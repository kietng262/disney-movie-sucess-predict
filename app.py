import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    with open('disney_model_with_director.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🎬 Disney Movie Success Predictor")

st.markdown("Predict whether a new Disney movie is likely to succeed based on early production information.")

# User Inputs
title = st.text_input("Movie Title", "Untitled Project")
runtime = st.slider("Runtime (minutes)", 60, 180, 100)
genres = st.multiselect("Genre(s)", 
                        ['Action', 'Adventure', 'Animation', 'Comedy', 'Drama', 'Fantasy', 'Family', 'Sci-Fi'],
                        default=['Adventure'])
director = st.text_input("Director Name", "Jon Favreau")
release_season = st.selectbox("Release Season", ['Spring', 'Summer', 'Fall', 'Winter'])
is_sequel = st.checkbox("Is this a sequel?", value=False)
is_franchise = st.checkbox("Part of a known franchise?", value=False)

# Format input
def prepare_input():
    return pd.DataFrame([{
        'runtime': runtime,
        'genres': ", ".join(genres),
        'director': director,
        'release_season': release_season,
        'is_sequel': int(is_sequel),
        'is_franchise': int(is_franchise)
    }])

# Predict
if st.button("🎯 Predict"):
    input_df = prepare_input()
    
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]  # Probability of success
        
        label = "✅ Likely Success" if prediction == 1 else "❌ Likely Failure"
        st.subheader(f"Prediction: {label}")
        st.metric(label="Success Probability", value=f"{proba*100:.2f}%")
        
        with st.expander("📊 View Input Data"):
            st.write(input_df)
    except Exception as e:
        st.error(f"Prediction failed: {e}")