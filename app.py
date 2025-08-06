# app.py
import streamlit as st
import pandas as pd
import pickle

# --- Load The Model ---
# Use cache to avoid reloading the model on each interaction
@st.cache_resource
def load_model():
    with open('disney_model_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_model()

# --- User Interface ---
st.set_page_config(page_title="Disney Success Predictor", page_icon="🎬", layout="wide")

st.title("🎬 Disney Movie Success Predictor")
st.write("Enter the movie's details to predict its likelihood of success.")

# Create columns for the layout
col1, col2 = st.columns(2)

with col1:
    st.header("Basic Information")
    runtime_minutes = st.slider("Runtime (minutes)", 50, 200, 90)
    genre_count = st.number_input("Number of Genres", 1, 10, 3)
    log_votes = st.slider("Log of Vote Count", 5.0, 15.0, 10.0, 0.1)
    avg_director_success_rate = st.slider("Director's Average Success Rate", 0.0, 1.0, 0.5, 0.01)

with col2:
    st.header("Other Factors")
    is_adult = st.selectbox("Is it an adult (18+) film?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    sequel_flag = st.selectbox("Is it a sequel?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    franchise_flag = st.selectbox("Is it part of a franchise?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# --- Handle Input and Prediction ---
if st.button("Predict Now!", type="primary"):
    # Create a DataFrame from the user's input.
    # The structure of this DataFrame MUST EXACTLY MATCH the structure of X_train.
    
    # Based on the notebook, the model needs many one-hot encoded columns.
    # For this example, we will create a simplified input. In a real-world scenario,
    # you would collect inputs for 'genres', 'release_season', etc.,
    # and process them to create the correctly formatted DataFrame.
    
    # Simplified example:
    input_data = {
        'runtime_minutes': [runtime_minutes],
        'genre_count': [genre_count],
        'is_adult': [is_adult],
        'sequel_flag': [sequel_flag],
        'log_votes': [log_votes],
        'franchise_flag': [franchise_flag],
        'avg_director_success_rate': [avg_director_success_rate],
        # Add dummy columns for the one-hot features the pipeline expects.
        # You would get this list from your saved pipeline.
        'Action': [0], 'Adventure': [0], 'Animation': [0], 'Biography': [0], 'Comedy': [1], 'Crime': [0],
        'Documentary': [0], 'Drama': [0], 'Family': [0], 'Fantasy': [0], 'Film-Noir': [0], 'Game-Show': [0],
        'History': [0], 'Horror': [0], 'Music': [0], 'Musical': [0], 'Mystery': [0], 'News': [0], 'Reality-TV': [0],
        'Romance': [0], 'Sci-Fi': [0], 'Short': [0], 'Sport': [0], 'Talk-Show': [0], 'Thriller': [0], 'War': [0], 'Western': [0],
        'season_Fall': [1], 'season_Spring': [0], 'season_Summer': [0], 'season_Winter': [0],
        'decade_1930': [0], 'decade_1940': [0], 'decade_1950': [0], 'decade_1960': [0], 'decade_1970': [0],
        'decade_1980': [0], 'decade_1990': [1], 'decade_2000': [0], 'decade_2010': [0], 'decade_2020': [0],
        'window_Holiday': [0], 'window_Other': [1], 'window_Summer': [0]
    }

    input_df = pd.DataFrame(input_data)
    
    # Make prediction
    prediction = pipeline.predict(input_df)[0]
    prediction_proba = pipeline.predict_proba(input_df)[0]

    # Display the result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"🎬 The movie is likely to be a SUCCESS!")
        st.write(f"Probability of Success: {prediction_proba[1]*100:.2f}%")
    else:
        st.error(f"😔 The movie is likely to FAIL.")
        st.write(f"Probability of Success: {prediction_proba[1]*100:.2f}%")
    
    st.progress(prediction_proba[1])