import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.utils.file_utils import TempDir

# Load Diabetes dataset
diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

# Set page layout
st.set_page_config(
    page_title="Diabetes MLOps Project",
    page_icon=":pill:",
    layout="wide",
)

# Add logo to the top right corner
logo_path = "diabetes.jpg"
st.image(logo_path, width=150)

# Sidebar for user input parameters
st.sidebar.header('User Input Parameters')

# Page selection
pages = ["Home", "Visualization", "Modeling"]
page = st.sidebar.selectbox("Select Page:", pages)

# Main app content
st.title("Diabetes MLOps Project")

# Home Page
if page == "Home":
    st.write("Welcome to the Diabetes Prediction App. Use the sidebar to navigate to different pages.")

# Visualization Page
elif page == "Visualization":
    st.title("Visualization Page")
    st.write("Explore visualizations of the Diabetes dataset.")

    # Create DataFrame
    df_diabetes = pd.DataFrame(data=X_diabetes, columns=[f'feature_{i}' for i in range(X_diabetes.shape[1])])
    df_diabetes['target'] = y_diabetes

    
    # Histogram
    st.subheader("Histogram:")
    histogram_fig = px.histogram(df_diabetes, x='target', color='target', marginal='rug')
    st.plotly_chart(histogram_fig)

    # Boxplot
    st.subheader("Box Plot:")
    boxplot_fig = px.box(df_diabetes, y='target', points='all', title='Target Boxplot')
    st.plotly_chart(boxplot_fig)

    # Bar plot
    st.subheader("Bar Plot:")
    barplot_fig = px.bar(df_diabetes, x='target', title='Target distribution')
    st.plotly_chart(barplot_fig)

# Modeling Page
elif page == "Modeling":
    st.title("Modeling Page")
    st.write("Train a RandomForestRegressor on the Diabetes dataset.")

    # Sidebar - User input features
    st.sidebar.header('User Input Parameters')

    # Model training
    st.subheader("Train Model:")
    with st.form("train_model_form_diabetes"):
        split_ratio_diabetes = st.slider("Train/Test Split Ratio", 0.1, 0.9, 0.8, 0.05)
        submit_button_diabetes = st.form_submit_button(label="Train Model")

    if submit_button_diabetes:
        # Split the data
        X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(
            X_diabetes, y_diabetes, test_size=split_ratio_diabetes, random_state=42
        )

        # Initialize and train the model
        model_diabetes = RandomForestRegressor()
        model_diabetes.fit(X_train_diabetes, y_train_diabetes)

        # Make predictions
        y_pred_diabetes = model_diabetes.predict(X_test_diabetes)

        # Log model and metrics to MLflow
        with mlflow.start_run():
            mlflow.log_param("split_ratio_diabetes", split_ratio_diabetes)
            mlflow.log_metric("mean_squared_error_diabetes", mean_squared_error(y_test_diabetes, y_pred_diabetes))
            mlflow.sklearn.log_model(model_diabetes, "model_diabetes")

        # Download artifacts
        if mlflow.active_run() is not None:
            run_id_model_diabetes = mlflow.active_run().info.run_id
            with TempDir(chdr=True) as tmp_model_diabetes:
                try:
                    download_artifacts(run_id_model_diabetes, "model_diabetes", tmp_model_diabetes.path())
                    st.success(f"Artifacts downloaded successfully to {tmp_model_diabetes.path()}")
                except MlflowException as e:
                    st.error(f"Failed to download artifacts: {e}")
