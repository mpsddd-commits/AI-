# app/main.py

import streamlit as st
import pandas as pd
import numpy as np
from src import data_preprocessing, train
import os
from create_pie_chart import create_earth_composition_pie_chart

def main():
    """
    Main function to run the data analysis pipeline as a Streamlit app.
    """
    st.title("Data Analysis and Visualization App")

    st.header("1. Data Analysis Pipeline")

    if st.button("Run Analysis"):
        st.write("Running the data analysis pipeline...")

        # Create a dummy CSV file for demonstration
        data_dir = 'app/data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        data_path = os.path.join(data_dir, 'sample_data.csv')

        # Create a sample dataframe
        data = {
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        }
        df = pd.DataFrame(data)
        
        # Save the dataframe to a csv file
        df.to_csv(data_path, index=False)
        st.write(f"Sample data created at `{data_path}`")
        st.dataframe(df.head())

        df_loaded = data_preprocessing.load_data(data_path)
        df_processed = data_preprocessing.preprocess_data(df_loaded)
        
        st.write("Data preprocessed (filled missing values).")

        if 'target' in df_processed.columns:
            st.write("Training model...")
            clf, score = train.train_model(df_processed)
            st.write(f"Model Accuracy: `{score}`")

        else:
            st.write("Target column not found. Skipping model training.")

    st.header("2. Earth's Composition Pie Chart")
    
    # Check if the image exists
    pie_chart_path = 'app/earth_composition_pie_chart.png'
    if not os.path.exists(pie_chart_path):
        create_earth_composition_pie_chart()

    st.image(pie_chart_path, caption="Elemental Composition of Earth's Crust by Mass")


if __name__ == "__main__":
    main()
