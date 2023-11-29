import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Load Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

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
    df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    # Scatter plot
    st.subheader("Scatter Plot:")
    scatter_fig = px.scatter_matrix(df, dimensions=diabetes.feature_names, color='target')
    st.plotly_chart(scatter_fig)

    # Histogram
    st.subheader("Histogram:")
    histogram_fig = px.histogram(df, x='target', color='target', marginal='rug')
    st.plotly_chart(histogram_fig)

    # Heatmap
    st.subheader("Heatmap:")
    heatmap_fig = px.imshow(df.corr(), color_continuous_scale='viridis', labels=dict(color='Correlation'))
    st.plotly_chart(heatmap_fig)

    # Count plot
    st.subheader("Count Plot:")
    countplot_fig = px.histogram(df, x='target', color='target', marginal='rug')
    st.plotly_chart(countplot_fig)

    # Bar plot
    st.subheader("Bar Plot:")
    barplot_fig = px.bar(df, x='target', title='Target distribution')
    st.plotly_chart(barplot_fig)


# Modeling Page
elif page == "Modeling":
    st.title("Modeling Page")
    st.write("Train a RandomForestRegressor on the Diabetes dataset.")

    # Sidebar - User input features
    st.sidebar.header('User Input Parameters')

    # Model training
    st.subheader("Train Model:")
    with st.form("train_model_form"):
        split_ratio = st.slider("Train/Test Split Ratio", 0.1, 0.9, 0.8, 0.05)
        submit_button = st.form_submit_button(label="Train Model")

    if submit_button:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

        # Initialize and train the model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Log model and metrics to MLflow
        with mlflow.start_run():
            mlflow.log_param("split_ratio", split_ratio)
            mlflow.log_metric("mean_squared_error", mean_squared_error(y_test, y_pred))
            mlflow.sklearn.log_model(model, "model")
