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


    # Bar plot
    st.subheader("Bar Plot:")
    barplot_fig = px.bar(df_diabetes, x='target', title='Target distribution')
    st.plotly_chart(barplot_fig)

# Modeling Page
elif page == "Modeling":
    st.title("Modeling Page")
    st.write("Train models on the Diabetes dataset.")

    # Sidebar - User input features
    st.sidebar.header('User Input Features')

    # Model training
    st.subheader("Train Models:")

    models = {
        "Random Forest": (RandomForestRegressor(), "rf"),
        "K-Nearest Neighbors (KNN)": (KNeighborsRegressor(), "knn"),
        "Logistic Regression": (LogisticRegression(), "logreg"),
        "Support Vector Machine (SVM)": (SVR(), "svm"),
        "Pipeline (StandardScaler + RandomForestRegressor)": (
            Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor())]), "pipeline"),
    }

    for model_name, (model, model_prefix) in models.items():
        with st.form(f"{model_prefix}_model_form"):
            test_size = st.slider(f"Test Size ({model_name})", 0.1, 0.5, 0.2, 0.05)
            submit_button = st.form_submit_button(label=f"Train {model_name} Model")

        if submit_button:
            # Split the data
            X_train_model, X_test_model, y_train_model, y_test_model = train_test_split(
                X_diabetes, y_diabetes, test_size=test_size, random_state=42
            )

            # Train the model
            model.fit(X_train_model, y_train_model)

            # Make predictions
            y_pred_model = model.predict(X_test_model)

            # Calculate mean squared error
            mse_model = mean_squared_error(y_test_model, y_pred_model)

            # Log model and metrics to MLflow
            with mlflow.start_run():
                mlflow.log_param(f"test_size_{model_prefix}", test_size)
                mlflow.log_metric(f"mean_squared_error_{model_prefix}", mse_model)
                mlflow.sklearn.log_model(model, f"model_{model_prefix}")

            # Download artifacts
            if mlflow.active_run() is not None:
                run_id_model = mlflow.active_run().info.run_id
                with TempDir(chdr=True) as tmp_model:
                    try:
                        download_artifacts(run_id_model, f"model_{model_prefix}", tmp_model.path())
                        st.success(f"Artifacts downloaded successfully to {tmp_model.path()}")
                    except MlflowException as e:
                        st.error(f"Failed to download artifacts: {e}")

