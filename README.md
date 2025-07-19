#    diabete_Mlops

This Streamlit application coordinates diabetes information examination with MLOps utilizing MLflow. It has three pages: **Home**, **Visualization**, and **Modeling**.

- **Home**: Prologue to the application.
  
- **Visualization**: Showcases histograms and bar plots for the diabetes dataset utilizing Plotly.
  
- **Modeling**: Permits clients to prepare different relapse models (Irregular Woodland, KNN, Calculated Relapse, SVR, and a Pipeline with StandardScaler). Clients can choose the test size and train models, which are then assessed utilizing mean squared mistake. Models and measurements are logged with MLflow, and relics can be downloaded if accessible.

The application incorporates highlights for exploring between pages, transferring logos, and imagining or preparing models on the diabetes dataset.

streamlit run app.py


mlflow ui
