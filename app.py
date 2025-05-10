import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load clustering models
scaler = StandardScaler()  # Create new StandardScaler instance
pca = joblib.load("models/pca.pkl")

# Load scalers for LSTM
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Forecasting model options
model_options = {
    "Original XGBoost": "models/xg_model.pkl",
    "Original Linear Regression": "models/linear_regression_model.pkl",
    "Original Random Forest": "models/rf_model.pkl",
    "Improved XGBoost": "models/improved_xgboost.joblib",
    "Improved Linear Regression": "models/improved_linear_regression.joblib",
    "Improved Random Forest": "models/improved_random_forest.joblib",
    "LSTM (Neural Net)": "models/lstm_model.h5"
}

# Define features used for forecasting
forecast_features = ['temperature', 'humidity', 'windSpeed', 'pressure',
                     'hour', 'day_of_week', 'month', 'kmeans_cluster']

# Load dataset
df = pd.read_csv("data/final.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df["date"] = df["datetime"].dt.date

# Streamlit UI
st.title("Electricity Demand Forecasting & Clustering")

# Add k value slider
k_value = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=4, step=1)

city = st.selectbox("Select City", df["city"].unique())
start_date = st.date_input("Start Date", df["date"].min())
end_date = st.date_input("End Date", df["date"].max())
selected_model = st.selectbox("Select Forecasting Model", list(model_options.keys()))

if start_date > end_date:
    st.warning("Start date must be before end date.")
else:
    filtered_df = df[(df["city"] == city) &
                     (df["date"] >= start_date) &
                     (df["date"] <= end_date)].copy()
    
    if filtered_df.empty:
        st.warning("No data found.")
    else:
        # Prepare data for clustering
        clustering_features = [  # same as in training
            'temperature', 'apparentTemperature', 'dewPoint', 'humidity', 'pressure',
            'windSpeed', 'windGust', 'windBearing', 'cloudCover', 'uvIndex', 'visibility',
            'precipIntensity', 'precipProbability', 'ozone', 'demand', 'demand_zscore',
            'demand_anomaly', 'demand_anomaly_iqr', 'anomaly_iso',
            'hour', 'day_of_week', 'month'
        ]
        X_clust = filtered_df[clustering_features]
        X_scaled = scaler.fit_transform(X_clust)
        X_pca = pca.transform(X_scaled)
        
        # Create and fit KMeans with selected k value
        kmeans = KMeans(n_clusters=k_value, random_state=42)
        filtered_df["kmeans_cluster"] = kmeans.fit_predict(X_pca)

        # Clustering plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                        hue=filtered_df["kmeans_cluster"], palette="tab10", ax=ax1)
        ax1.set_title(f"PCA Clustering (k={k_value})")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig1)

        # Forecasting
        X_forecast = filtered_df[forecast_features]
        y_true = filtered_df["demand"]

        if selected_model == "LSTM (Neural Net)":
            model = load_model(model_options[selected_model])
            
            # Get the expected input shape from the model
            expected_features = model.input_shape[2]
            look_back = model.input_shape[1]
            
            # Prepare data for LSTM - using only demand as the model expects
            y_lstm = filtered_df[['demand']].values
            
            # Scale the data
            y_scaled = scaler_y.fit_transform(y_lstm)
            
            # Create sequences with look-back window
            X_seq = []
            for i in range(len(y_scaled) - look_back):
                X_seq.append(y_scaled[i:i + look_back])
            X_seq = np.array(X_seq)
            
            # Print shapes for debugging
            st.write(f"X_seq shape: {X_seq.shape}")
            st.write(f"Expected input shape: {model.input_shape}")
            
            # Make predictions
            y_pred_scaled = model.predict(X_seq)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            
            # Adjust y_true and datetime to match predictions
            y_true = filtered_df['demand'].values[look_back:]
            pred_dates = filtered_df["datetime"].values[look_back:]
        else:
            try:
                # Load the appropriate model
                if "Improved" in selected_model:
                    # For improved models, we need to handle the preprocessing pipeline
                    model = joblib.load(model_options[selected_model])
                    # The model is a pipeline that includes preprocessing
                    y_pred = model.predict(filtered_df)
                else:
                    if selected_model == "Original XGBoost":
                        # Create new XGBoost model with same parameters as training
                        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                        # Load the model parameters
                        saved_model = joblib.load(model_options[selected_model])
                        model.load_model(saved_model.get_booster().save_raw())
                    else:
                        model = joblib.load(model_options[selected_model])
                    y_pred = model.predict(X_forecast)
                
                y_true = filtered_df["demand"].values
                pred_dates = filtered_df["datetime"].values
                
                # Add model performance metrics
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                
                st.write(f"Model Performance Metrics:")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"R² Score: {r2:.4f}")
                
            except Exception as e:
                st.error(f"Error loading {selected_model} model: {str(e)}")
                st.info("Please ensure you have the correct version of XGBoost installed.")
                st.stop()

        # Forecast plot
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(pred_dates, y_true, label="Actual")
        ax2.plot(pred_dates, y_pred, label="Predicted", linestyle="--")
        ax2.set_title(f"Electricity Demand Forecast ({selected_model})")
        ax2.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

        # Add scatter plot of actual vs predicted
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(y_true, y_pred, alpha=0.5)
        ax3.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax3.set_xlabel('Actual Values')
        ax3.set_ylabel('Predicted Values')
        ax3.set_title(f'Actual vs Predicted Values ({selected_model})')
        plt.tight_layout()
        st.pyplot(fig3)

        # Documentation
        with st.expander("ℹ️ Help & Documentation"):
            st.markdown(f"""
            - **Clustering**: PCA + KMeans (k={k_value}) based on weather/time/anomaly features.
            - **Forecasting**: Using **{selected_model}** on weather, time & cluster features.
            - Improved models include enhanced preprocessing and hyperparameter tuning.
            - LSTM input is reshaped to meet model expectations.
            """)
