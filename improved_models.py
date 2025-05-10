import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_and_preprocess_data():
    # Load the data
    print("Loading data from Final.csv...")
    df = pd.read_csv('Data/Final.csv')
    
    # Convert boolean columns to int
    bool_cols = ['demand_anomaly', 'demand_anomaly_iqr', 'anomaly_iso']
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    # Define feature groups
    categorical_cols = ['company', 'region', 'city', 'summary', 'icon', 'precipType', 'season', 'day_of_week']
    numerical_cols = [
        'precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature', 'dewPoint', 'humidity',
        'pressure', 'windSpeed', 'windGust', 'windBearing', 'cloudCover', 'uvIndex', 'visibility', 'ozone',
        'hour', 'month', 'week', 'demand_zscore', 'kmeans_cluster'
    ] + bool_cols
    # Columns to drop
    drop_cols = ['date', 'time', 'datetime']
    
    # Prepare features and target
    X = df[categorical_cols + numerical_cols]
    y = df['demand']
    
    # Create preprocessing pipelines for both categorical and numerical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor

def plot_predictions_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.tight_layout()
    plt.show()

def create_comparison_df(y_true, y_pred, model_name):
    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Difference': y_true - y_pred,
        'Abs_Difference': np.abs(y_true - y_pred)
    })
    
    # Calculate statistics
    stats = {
        'Model': model_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Mean Difference': df['Difference'].mean(),
        'Std Difference': df['Difference'].std(),
        'Min Difference': df['Difference'].min(),
        'Max Difference': df['Difference'].max()
    }
    
    return df, stats

def train_improved_models(X_train, X_test, y_train, y_test, preprocessor, categorical_cols):
    # 1. Improved Linear Regression with Ridge
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=1.0))
    ])
    
    print("\nTraining Linear Regression model...")
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    
    # 2. Improved XGBoost with hyperparameter tuning
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 1
    }
    
    print("\nTraining XGBoost model...")
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', xgb.XGBRegressor(**xgb_params, random_state=42))
    ])
    
    xgb_pipeline.fit(X_train, y_train)
    y_pred_xg = xgb_pipeline.predict(X_test)
    mae_xg = mean_absolute_error(y_test, y_pred_xg)
    rmse_xg = np.sqrt(mean_squared_error(y_test, y_pred_xg))
    
    # 3. Improved Random Forest with hyperparameter tuning
    rf_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True
    }
    
    print("\nTraining Random Forest model...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor(**rf_params, random_state=42))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    # Print results
    print("\nModel Performance Metrics:")
    print(f'Enhanced Linear Regression - MAE: {mae_lr:.4f}, RMSE: {rmse_lr:.4f}')
    print(f'Enhanced XGBoost - MAE: {mae_xg:.4f}, RMSE: {rmse_xg:.4f}')
    print(f'Enhanced Random Forest - MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}')
    
    # Get feature names after preprocessing
    feature_names = (preprocessor.named_transformers_['cat']
                    .get_feature_names_out(categorical_cols))
    feature_names = np.concatenate([feature_names, numerical_cols])
    
    # Feature importance for XGBoost and Random Forest
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_pipeline.named_steps['xgb'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_pipeline.named_steps['rf'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Important Features (XGBoost):")
    print(xgb_importance.head())
    print("\nTop 5 Important Features (Random Forest):")
    print(rf_importance.head())
    
    # Create comparison DataFrames and statistics for each model
    comparison_dfs = {}
    all_stats = []
    
    # Linear Regression
    lr_df, lr_stats = create_comparison_df(y_test, y_pred_lr, 'Linear Regression')
    comparison_dfs['Linear Regression'] = lr_df
    all_stats.append(lr_stats)
    
    # XGBoost
    xgb_df, xgb_stats = create_comparison_df(y_test, y_pred_xg, 'XGBoost')
    comparison_dfs['XGBoost'] = xgb_df
    all_stats.append(xgb_stats)
    
    # Random Forest
    rf_df, rf_stats = create_comparison_df(y_test, y_pred_rf, 'Random Forest')
    comparison_dfs['XGBoost'] = rf_df
    all_stats.append(rf_stats)
    
    # Print detailed statistics
    print("\nDetailed Model Statistics:")
    stats_df = pd.DataFrame(all_stats)
    print(stats_df.to_string(index=False))
    
    # Plot predictions vs actual for each model
    print("\nGenerating prediction plots...")
    plot_predictions_vs_actual(y_test, y_pred_lr, 'Linear Regression')
    plot_predictions_vs_actual(y_test, y_pred_xg, 'XGBoost')
    plot_predictions_vs_actual(y_test, y_pred_rf, 'Random Forest')
    
    # Print sample predictions
    print("\nSample Predictions (First 5 rows):")
    for model_name, df in comparison_dfs.items():
        print(f"\n{model_name} Predictions:")
        print(df.head().to_string())
    
    return {
        'linear_regression': {'model': lr_pipeline, 'predictions': y_pred_lr, 'comparison_df': lr_df},
        'xgboost': {'model': xgb_pipeline, 'predictions': y_pred_xg, 'comparison_df': xgb_df},
        'random_forest': {'model': rf_pipeline, 'predictions': y_pred_rf, 'comparison_df': rf_df},
        'statistics': stats_df
    }

def save_models(results):
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save each model
    model_files = {
        'linear_regression': 'improved_linear_regression.joblib',
        'xgboost': 'improved_xgboost.joblib',
        'random_forest': 'improved_random_forest.joblib'
    }
    
    for model_name, file_name in model_files.items():
        model_path = os.path.join('models', file_name)
        joblib.dump(results[model_name]['model'], model_path)
        print(f"Saved {model_name} model to {model_path}")
    
    # Save statistics
    stats_path = os.path.join('models', 'improved_model_statistics.csv')
    results['statistics'].to_csv(stats_path, index=False)
    print(f"Saved model statistics to {stats_path}")

if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()
    
    # Get categorical columns for feature names
    categorical_cols = ['company', 'region', 'city', 'summary', 'icon', 'precipType', 'season', 'day_of_week']
    numerical_cols = [
        'precipIntensity', 'precipProbability', 'temperature', 'apparentTemperature', 'dewPoint', 'humidity',
        'pressure', 'windSpeed', 'windGust', 'windBearing', 'cloudCover', 'uvIndex', 'visibility', 'ozone',
        'hour', 'month', 'week', 'demand_zscore', 'kmeans_cluster'
    ] + ['demand_anomaly', 'demand_anomaly_iqr', 'anomaly_iso']
    
    # Train the models
    results = train_improved_models(X_train, X_test, y_train, y_test, preprocessor, categorical_cols)
    
    # Save the models
    save_models(results) 