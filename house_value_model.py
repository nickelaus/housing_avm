import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb # <--- IMPORT XGBOOST
import optuna # <--- IMPORT OPTUNA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # <-- Add this line BEFORE importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import datetime # Keep for current_year if any other age-like feature might be added later, or remove if truly unused.
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog
import os
from sklearn.compose import ColumnTransformer # <-- Import ColumnTransformer


def load_data(file_path, sheet_name="EnhancedMasterList"):
    """Loads data from the specified Excel sheet."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, target_column, feature_columns=None):
    """Basic preprocessing: selects features, target, and handles missing values."""
    if df is None:
        return None, None, None

    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in DataFrame.")
        return None, None, None

    if feature_columns is None:
        potential_features = df.select_dtypes(include=['number']).columns.tolist()
        # Ensure target_column is not accidentally included in features
        feature_columns = [col for col in potential_features if col != target_column]
        if not feature_columns:
            print("Error: No numeric feature columns found or specified.")
            return None, None, None
        print(f"Auto-selected numeric feature columns: {feature_columns}")
    
    # Remove houses older than 1.5 years ago (days_since_sold > 545)
    if 'Days_Since_Sold' in df.columns:
        initial_shape = df.shape
        df = df[df['Days_Since_Sold'] <= 500]
        print(f"Filtered out houses sold more than 1.5 years ago (Days_Since_Sold > 545). Rows before: {initial_shape[0]}, after: {df.shape[0]}")

    # Remove houses from specified cities
    cities_to_remove = [] # enter cities to remove here i.e. ['Etna', 'Alpine', 'Star Valley Ranch']
    city_col_candidates = [col for col in df.columns if 'city' in col.lower()]
    if city_col_candidates:
        city_col = city_col_candidates[0]
        before_city_filter = df.shape[0]
        df = df[~df[city_col].isin(cities_to_remove)]
        print(f"Filtered out houses from cities {cities_to_remove} using column '{city_col}'. Rows before: {before_city_filter}, after: {df.shape[0]}")
    else:
        print("Warning: No city column found for filtering cities.")
    # Ensure specified feature_columns exist in the DataFrame
    actual_features_in_df = [col for col in feature_columns if col in df.columns]
    missing_specified_features = [col for col in feature_columns if col not in df.columns]
    if missing_specified_features:
        print(f"Warning: The following specified features are not in the DataFrame and will be ignored: {missing_specified_features}")
    
    if not actual_features_in_df:
        print(f"Error: None of the specified or auto-selected features are present in the DataFrame. Available columns: {df.columns.tolist()}")
        return None, None, None

    X = df[actual_features_in_df].copy()
    y = df[target_column].copy()

    # Handle missing values
    for col in X.columns:
        if X[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col].fillna(X[col].mean(), inplace=True)
                print(f"Filled missing values in numeric column '{col}' with its mean.")
            else: # Attempt to fill non-numeric with mode
                try:
                    mode_val = X[col].mode()[0]
                    X[col].fillna(mode_val, inplace=True)
                    print(f"Filled missing values in non-numeric column '{col}' with mode '{mode_val}'.")
                except IndexError: # If mode cannot be found (e.g., all NaNs or multiple modes and no first)
                    print(f"Warning: Could not find a unique mode for non-numeric column '{col}'. It might contain many NaNs or be problematic.")
                    # Optionally, drop column or fill with a constant placeholder if critical
                    # X[col].fillna("Unknown", inplace=True) # Example placeholder


    if y.isnull().any():
        print(f"Warning: Target column '{target_column}' has {y.isnull().sum()} missing values. Rows with missing target will be dropped.")
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        if X.empty:
            print("Error: No data remaining after dropping NaNs in target variable.")
            return None, None, None

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    return X, y, actual_features_in_df # Return the list of features actually used

def correlation_feature_selection(X, y, threshold=0.5):
    # Calculate correlation matrix
    corr = pd.DataFrame(X).corr()
    
    # Get correlation with target variable
    target_corr = X.corrwith(pd.Series(y)).abs()
    
    # Select features above threshold
    selected_features = target_corr[target_corr > threshold].index.tolist()
    
    return selected_features

def analyze_feature_importance(X, y, feature_names):
    """Analyzes and prints feature importance using Random Forest."""
    if X is None or y is None or X.size == 0 or y.size == 0:
        print("Input data (X or y) is None or empty.")
        return

    print("\n--- Feature Importance Analysis (Random Forest) ---")
    # Using RandomForestRegressor for feature importance
    # n_estimators can be tuned, 100 is a common default
    # random_state for reproducibility
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    try:
        model_rf.fit(X, y)
    except Exception as e:
        print(f"Error fitting RandomForestRegressor for feature importance: {e}")
        print("Please ensure your features (X) are numeric and do not contain NaNs or Infs.")
        return

    importances_rf = model_rf.feature_importances_
    feature_importance_df_rf = pd.DataFrame({'feature': feature_names, 'importance': importances_rf})
    feature_importance_df_rf = feature_importance_df_rf.sort_values(by='importance', ascending=False)

    print("Random Forest Feature Importances:")
    print(feature_importance_df_rf)

    # plt.figure(figsize=(12, 7))
    # sns.barplot(x='importance', y='feature', data=feature_importance_df_rf.head(10)) # Display top 10
    # plt.title('Top 10 Feature Importances (Random Forest)')
    # plt.tight_layout()
    # plt.show()

    # XGBoost Feature Importance (if XGBoost is available and data is suitable)
    print("\n--- Feature Importance Analysis (XGBoost) ---")
    try:
        # model_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)                      
        model_xgb = create_xgb_model()
                                        
        #n_estimators': 340, 'learning_rate': 0.13902497447581666, 'max_depth': 6, 'min_child_weight': 2, 'subsample': 0.713186137275456, 'colsample_bytree': 0.7480581151233089, 'alpha': 0.752419230564225
        model_xgb.fit(X, y)
        importances_xgb = model_xgb.feature_importances_
        feature_importance_df_xgb = pd.DataFrame({'feature': feature_names, 'importance': importances_xgb})
        feature_importance_df_xgb = feature_importance_df_xgb.sort_values(by='importance', ascending=False)
        
        # print("XGBoost Feature Importances:")
        # print(feature_importance_df_xgb)

        # plt.figure(figsize=(12, 7))
        # sns.barplot(x='importance', y='feature', data=feature_importance_df_xgb.head(10))
        # plt.title('Top 10 Feature Importances (XGBoost)')
        # plt.tight_layout()
        # plt.show()
        return feature_importance_df_rf, feature_importance_df_xgb # Return both
    except Exception as e:
        print(f"Could not generate XGBoost feature importances: {e}")
        return feature_importance_df_rf, None # Return RF importances and None for XGB


def train_evaluate_model_and_predict(X_train_scaled, X_test_scaled, y_train_log, y_test_log, model_type='linear_regression', tune_hyperparameters_xgb=False, tune_hyperparameters_ridge=False, use_optuna_for_xgb=True, n_optuna_trials=100, ridge_alpha=1.0):


    model = None
    best_params_xgb = None 
    grid_search_object = None # To hold GridSearchCV object if used

    if model_type == 'ridge':
        if tune_hyperparameters_ridge:  # Add this parameter to your function
            print("Tuning Ridge hyperparameters...")
            ridge = Ridge(random_state=42)
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
            }
            grid_search = GridSearchCV(
                ridge, 
                param_grid, 
                cv=5, 
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train_log)
            model = grid_search.best_estimator_
            best_params_ridge = grid_search.best_params_
            print(f"Best Ridge parameters: {best_params_ridge}")
            needs_explicit_fit = False
        else:
            model = Ridge(alpha=ridge_alpha, random_state=42)
    elif model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'xgboost':
        if tune_hyperparameters_xgb:
            if use_optuna_for_xgb:
                print("\n--- Tuning XGBoost Hyperparameters with Optuna ---")
                
                def objective(trial):
                    # ... (objective function as you have it) ...
                    params = {
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        # 'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                        # 'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                        # 'max_depth': trial.suggest_int('max_depth', 3, 10),
                        # 'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
                        # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
                        # 'gamma': trial.suggest_float('gamma', 0, 0.5, step=0.05),
                        # 'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                        # 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),

                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.15, log=True),
                        'max_depth': trial.suggest_int('max_depth', 2, 10),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 12),
                        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
                        'alpha': trial.suggest_float('alpha', 0.001, .95, log=True),
                        # 'reg_alpha': trial.suggest_float('reg_alpha', .001, .95, log=True),
                        # 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    xgb_model_optuna = xgb.XGBRegressor(**params)
                    score = cross_val_score(xgb_model_optuna, X_train_scaled, y_train_log, cv=3, scoring='r2', n_jobs=-1).mean()
                    return score

                study = optuna.create_study(direction='maximize', study_name='xgb_regression_tuning')
                try:
                    study.optimize(objective, n_trials=n_optuna_trials, timeout=600) 
                    best_params_xgb = study.best_params
                    # print(f"Best XGBoost parameters found by Optuna: {best_params_xgb}")
                    # Print as a class-style attribute assignment for easy copy-paste
                    print("Best XGBoost parameters found by Optuna:")
                    for k, v in best_params_xgb.items():
                        print(f"{k} = {repr(v)},")
                    model = xgb.XGBRegressor(objective='reg:squarederror', **best_params_xgb, random_state=42, n_jobs=-1)
                except Exception as e:
                    print(f"Error during Optuna XGBoost hyperparameter tuning: {e}")
                    print("Falling back to default XGBoost parameters.")
                    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

            else: # Use GridSearchCV
                print("\n--- Tuning XGBoost Hyperparameters with GridSearchCV ---")
                # ... (your GridSearchCV param_grid and setup) ...
                param_grid = {
                    'n_estimators': [100, 200, 300], # Number of boosting rounds
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 4, 5],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.01, 0.1],
                    'reg_lambda': [1, 1.5, 2] # Default is 1, 0 means no L2
                }
                xgb_model_to_tune = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
                grid_search_object = GridSearchCV(estimator=xgb_model_to_tune, param_grid=param_grid,
                                           cv=3, scoring='r2', verbose=1, n_jobs=-1)
                try:
                    grid_search_object.fit(X_train_scaled, y_train_log)
                    best_params_xgb = grid_search_object.best_params_
                    print(f"Best XGBoost parameters found by GridSearchCV: {best_params_xgb}")
                    model = grid_search_object.best_estimator_ # This model is already fitted
                except Exception as e:
                    print(f"Error during GridSearchCV XGBoost hyperparameter tuning: {e}")
                    print("Falling back to default XGBoost parameters.")
                    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        else: # Use default XGBoost parameters if not tuning

            # ~90% accuracy
            model = create_xgb_model()
            
    else:
        print(f"Unsupported model type: {model_type}")
        return None

    # --- Model Fitting Logic ---
    model_description = f"{model_type.replace('_', ' ').title()}"
    needs_explicit_fit = True

    if model_type == 'xgboost' and tune_hyperparameters_xgb:
        if use_optuna_for_xgb:
            model_description += " (Optuna tuned)"
            # Model from Optuna (with best_params or default after error) always needs explicit fitting here.
        else: # GridSearchCV
            model_description += " (GridSearch tuned)"
            if grid_search_object and hasattr(grid_search_object, 'best_estimator_') and model is grid_search_object.best_estimator_:
                print(f"Using pre-fitted model from GridSearchCV for {model_description}.")
                needs_explicit_fit = False
            elif not (grid_search_object and hasattr(grid_search_object, 'best_estimator_')):
                 model_description += " (GridSearch failed, using default and fitting)"
    elif model_type == 'xgboost':
        model_description += " (default params)"
    
    print(f"\n--- Preparing {model_description} Model ---")

    if needs_explicit_fit:
        if model is not None:
            print(f"Fitting the {model_description} model on X_train, y_train...")
            try:
                model.fit(X_train_scaled, y_train_log)
            except Exception as e:
                print(f"Error explicitly fitting {model_description} model: {e}")
                return None
        else:
            print(f"Model object for {model_description} is None before explicit fit. Tuning/instantiation might have failed.")
            return None
            
    print("Model training complete.")

    # --- Model Evaluation ---
    if model is None: # Should not happen if logic above is correct, but as a safeguard
        print("Model is None before evaluation. Cannot proceed.")
        return None

    print("\n--- Model Evaluation ---")
    y_pred_train_log = model.predict(X_train_scaled)
    y_pred_test_log = model.predict(X_test_scaled)

    y_pred_train = np.expm1(y_pred_train_log) # Reverse the log transform

    # Also get the original scale y_test for evaluation
    y_pred_test = np.expm1(y_pred_test_log)

    y_test = np.expm1(y_test_log) # Reverse the log transform for test set
    y_train = np.expm1(y_train_log) # Reverse the log transform for train set
    
    # # Scatter plot of actual vs. predicted values
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test, y_pred_test, alpha=0.5)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    # plt.xlabel('Actual Price')
    # plt.ylabel('Predicted Price')
    # plt.title('Actual vs. Predicted Prices: ' + model_description)
    # plt.show()
    
    # 1. Calculate the prediction errors (residuals)
    residuals = y_test - y_pred_test

    # 2. Calculate the standard deviation of the residuals
    std_error = np.std(residuals)

    # 3. Define the confidence level and find the z-score (1.96 for 95%)
    confidence_level = 0.95
    z_score = 1.96

    # 4. Calculate the upper and lower bounds of the prediction interval
    lower_bound = y_pred_test - z_score * std_error
    upper_bound = y_pred_test + z_score * std_error

    # 5. Generate the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # Sort values for a clean line plot of the confidence band
    sort_indices = np.argsort(y_test)
    y_test_sorted = y_test.iloc[sort_indices]
    y_pred_sorted = y_pred_test[sort_indices]
    lower_bound_sorted = lower_bound[sort_indices]
    upper_bound_sorted = upper_bound[sort_indices]

    # Scatter plot of individual predictions
    plt.scatter(y_test, y_pred_test, color='royalblue', alpha=0.6, label=f'Model Predictions: {model_description}')

    # Line plot for the ideal fit
    plt.plot(y_test_sorted, y_test_sorted, color='red', linestyle='--', linewidth=2, label='Ideal Fit (Actual = Predicted)')

    # Shaded confidence interval
    plt.fill_between(
        y_test_sorted,
        lower_bound_sorted,
        upper_bound_sorted,
        color='gray',
        alpha=0.3,
        label=f'{int(confidence_level*100)}% Prediction Interval'
    )

    plt.title(f'Model Predictions with Confidence Interval: {model_description}', fontsize=16)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.legend()
    plt.tight_layout()
    # Create an 'images' directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')

    if os.path.exists(f'images/model_predictions_with_confidence_interval_{model_type}.png'):
        os.remove(f'images/model_predictions_with_confidence_interval_{model_type}.png')
    # # Save the figure with high resolution and tight bounding box
    plt.savefig(f'images/model_predictions_with_confidence_interval_{model_type}.png', dpi=300, bbox_inches='tight')    
    # plt.show()
    plt.close()

    print("Training Set Performance:")
    print(f"  MAE: {mean_absolute_error(y_train_log, y_pred_train_log):.4f}")
    print(f"  MSE: {mean_squared_error(y_train_log, y_pred_train_log):.4f}")
    print(f"  R-squared: {r2_score(y_train_log, y_pred_train_log):.4f}")

    print("\nTest Set Performance:")
    print(f"  MAE: {mean_absolute_error(y_test_log, y_pred_test_log):.4f}")
    print(f"  MSE: {mean_squared_error(y_test_log, y_pred_test_log):.4f}")
    print(f"  R-squared: {r2_score(y_test_log, y_pred_test_log):.4f}")

    # Evaluate
    train_score = model.score(X_train_scaled, y_train_log)
    test_score = model.score(X_test_scaled, y_test_log)
    print(f"Training R² score: {train_score:.3f}")
    print(f"Test R² score: {test_score:.3f}")    

    # predict 
    model_for_prediction = model# trained_xgb_model_optuna 
    if model_for_prediction and actual_training_feature_names:
        print(f"\nUsing {type(model_for_prediction).__name__} for example prediction.")
        new_property_data = {#ppppppredict
            'SQUARE FEET': 2112,
            'BEDS': 3,
            'BATHS': 2,
            # 'LOT SIZE': 43999,
            'LotSize_Log': np.log1p(43560*1),  # Use log1p if your model was trained on log-transformed Lot Size
            'AGE': 48,  # Provide 'AGE' directly
            'Basement_Level': 3,
            'Garage_Spaces':0,
            'STORAGE_STRUCTURE': 1,
            'Recently_Renovated': 1,

            'Days_Since_Sold': 1,  # Example value, adjust as needed
             #'Month_Sold': 10,  # Example value, adjust as needed
            # Ensure all keys here match names in actual_training_feature_names
        }
        new_data_df = pd.DataFrame([new_property_data])
        # Reorder columns to match the training data order, which the preprocessor expects
        new_data_df = new_data_df[actual_training_feature_names]

        new_data_preprocessed = preprocessor.transform(new_data_df)

        # --- DEBUGGING BLOCK: Add this to inspect the scaled data ---
        print("\n--- Debugging New Property Prediction ---")
        processed_feature_names = preprocessor.get_feature_names_out()
        print("Feature Names (post-processing):", processed_feature_names)
        print("Scaled Input Values for Model:", new_data_preprocessed)
        print("-------------------------------------\n")
        # --- END DEBUGGING BLOCK ---


        # Verify all features needed for training are in new_property_data
        missing_keys_for_prediction = [f for f in actual_training_feature_names if f not in new_property_data]
        if missing_keys_for_prediction:
            print(f"Error: Missing keys in new_property_data for prediction: {missing_keys_for_prediction}")
        else:
            unseen_predicted_price_log = predict_new_property_price(
                model_for_prediction,
                new_data_preprocessed
            )
            if unseen_predicted_price_log is not None:
                unseen_predicted_price = np.expm1(unseen_predicted_price_log)  # Reverse the log transform
                print(f"----> Final {model_description} Predicted Price for new property: ${unseen_predicted_price:,.2f}")
                # 3. Calculate the confidence interval for this prediction.
                #    We use the same standard error (std_error) and z_score (1.96)
                #    calculated from the model's performance on the entire test set.
                
                non_zero_mask = y_test != 0
                percentage_errors = (y_test[non_zero_mask] - y_pred_test[non_zero_mask]) / y_test[non_zero_mask]

                # Calculate the standard deviation of the percentage errors
                std_percentage_error = np.std(percentage_errors)

                # The margin of error is now a percentage of the prediction itself
                margin_of_error_relative = unseen_predicted_price * z_score * std_percentage_error                
                
                # margin_of_error = z_score * std_error


                unseen_lower_bound = unseen_predicted_price - margin_of_error_relative
                unseen_upper_bound = unseen_predicted_price + margin_of_error_relative

                # 4. Create the visualization.
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(8, 6))

                # Use errorbar to show the point and the confidence interval
                ax.errorbar(
                    x=[1], # A single categorical position on the x-axis
                    y=[unseen_predicted_price],
                    yerr=[[margin_of_error_relative]], # yerr is the margin of error for lower and upper
                    fmt='o', # Format for the point marker
                    color='darkorange',
                    ecolor='royalblue',
                    elinewidth=3,
                    capsize=10,
                    markersize=12,
                    markeredgecolor='black'
                )

                # Add text annotations for clarity
                ax.text(1, unseen_predicted_price, f' Predicted: ${unseen_predicted_price:,.0f} ', ha='left', va='center', fontsize=12, color='black', weight='bold')
                ax.text(1.05, unseen_upper_bound, f' Upper Bound: ${unseen_upper_bound:,.0f} ', ha='left', va='center', fontsize=11, color='darkgreen')
                ax.text(1.05, unseen_lower_bound, f' Lower Bound: ${unseen_lower_bound:,.0f} ', ha='left', va='center', fontsize=11, color='darkred')

                # Formatting the plot
                ax.set_xticks([1])
                ax.set_xticklabels(['Your Property Prediction'], fontsize=12)
                ax.set_ylabel('Predicted Price ($)', fontsize=12)
                ax.set_title(f'Prediction and 95% Confidence Interval for Unseen Property ({model_description})', fontsize=16)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                ax.grid(axis='x') # Keep grid on y-axis only
                plt.tight_layout()
                # plt.show()
                # remove the previous plot if it exists:
                if os.path.exists(f'images/prediction_with_confidence_interval_{model_type}.png'):
                    os.remove(f'images/prediction_with_confidence_interval_{model_type}.png')

                plt.savefig(f'images/prediction_with_confidence_interval_{model_type}.png', dpi=300, bbox_inches='tight')
                # If you want to close the plot to free memory
                plt.close()



    else:
        print("Skipping prediction example as the selected model or feature names are not available.")    





    return model


# This function should only need the model and the scaled data array.
def predict_new_property_price(model, data_scaled):
    """
    Predicts property prices using a model trained on scaled data.
    Handles both single (returning a float) and batch predictions.
    """
    predictions = model.predict(data_scaled)

       

    # If only one prediction was made, return it as a single number (scalar).
    if predictions.shape[0] == 1:
        return predictions[0]
    else:
        return predictions


def create_xgb_model():
    """
    Creates and returns an XGBRegressor model with default parameters
    """
    model_xgb = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,

            n_estimators = 730,
            learning_rate = 0.00805564518843674,
            max_depth = 2,
            min_child_weight = 3,
            subsample = 0.32297763282302394,
            colsample_bytree = 0.9485964201968369,
            alpha = 0.07098441209183254,
    )
    # {'n_estimators': 890, 'learning_rate': 0.12263389146867545, 'max_depth': 8, 'min_child_weight': 1, 'subsample': 0.6222002928887224, 'colsample_bytree': 0.32716083424759446, 'alpha': 0.8286567665837326}
    return model_xgb

if __name__ == "__main__":
    
    # Set up file dialog to select CSV file from data folder
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Set initial directory to 'data' folder if it exists
    data_folder = os.path.join(os.getcwd(), 'data')
    initial_dir = data_folder if os.path.exists(data_folder) else os.getcwd()

    # Open file dialog to select CSV file
    INPUT_EXCEL_PATH = filedialog.askopenfilename(
        title="Select enriched, cleaned CSV file",
        initialdir=initial_dir,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    # Check if the selected file is an output file (enriched cleaned)
    if 'enriched' not in INPUT_EXCEL_PATH.lower() or 'cleaned' not in INPUT_EXCEL_PATH.lower():
        print("Warning: Selected file does not appear to be an output file.")
        import tkinter.messagebox as messagebox
        
        # Show a dialog asking if they want to continue
        result = messagebox.askyesno(
            "File Selection Warning", 
            "The selected file does not appear to be an output file.\n\n"
            "Expected filename to contain 'enriched_cleaned'.\n\n"
            "Do you want to continue anyway?"
        )
        
        if not result:
            print("Exiting...")
            exit()
    # Check if a file was selected
    if not INPUT_EXCEL_PATH:
        print("No file selected. Exiting...")
        exit()

    print(f"Selected file: {INPUT_EXCEL_PATH}")
    TARGET_COLUMN = 'PRICE'
    # Define the features your model will be trained on.
    # 'AGE' is now expected to be a column directly in your Excel sheet.
    
    FEATURE_COLUMNS = [
        # 'SQUARE FEET', 'BEDS', 'BATHS', 'LOT SIZE', 'AGE', # 'AGE' from spreadsheet
        'SQUARE FEET', 'BEDS', 'BATHS', 'LotSize_Log', 'AGE', # 'AGE' from spreadsheet
        'Basement_Level', 'Garage_Spaces','STORAGE_STRUCTURE', 'Recently_Renovated',
        'Days_Since_Sold', 
        #'Month_Sold',

        # 'Days_Since_Sold','Month_Sold'
        #simplifying 
        # 'SQUARE FEET', 'BATHS', 'LOT SIZE', 'AGE', # 'AGE' from spreadsheet
        # 'Combined_Garage_Score',
        # Add any other relevant feature column names from your Excel file
        # Add any other relevant feature column names from your Excel file
    ]

    # df = load_data(INPUT_EXCEL_PATH)
    df = pd.read_csv(INPUT_EXCEL_PATH)


    if 'PRICE' in df.columns:
        price_threshold = df['PRICE'].quantile(0.99)
        # price_threshold = 1200000
        original_count = len(df)
        df = df[df['PRICE'] <= price_threshold].copy()
        print(f"Filtered out {original_count - len(df)} properties with values above {price_threshold:,.2f}.")
    
    df['LotSize_Log'] = np.log1p(df['LOT SIZE'])
    # exit(0) # Exit early for now, remove this line to continue with the rest of the script
    if df is not None:
        print("\nDataFrame columns upon loading:", df.columns.tolist())
        print("\nDataFrame info:")
        df.info()
        
        # Pass a copy of FEATURE_COLUMNS to avoid modification if preprocess_data changes it internally
        X, y, actual_training_feature_names = preprocess_data(df.copy(), TARGET_COLUMN, list(FEATURE_COLUMNS))
        
        if X is not None and y is not None and not X.empty and not y.empty and actual_training_feature_names:
            print(f"\nActual features used for training: {actual_training_feature_names}")

            "Trains a regression model and evaluates it."
            if X is None or y is None or X.empty or y.empty:
                print("Cannot train model with empty data.")
                exit(1)

            # 1. Define which columns to scale and which to pass through
            numerical_features = [col for col in actual_training_feature_names if col != 'Recently_Renovated'or col != 'STORAGE_STRUCTURE']
            passthrough_features = ['Recently_Renovated', 'STORAGE_STRUCTURE']

            # 2. Create the preprocessor object
            # This will apply StandardScaler to numerical features and do nothing to the passthrough features.
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('pass', 'passthrough', passthrough_features)
                ],
                remainder='drop' # This ensures only the columns you specified are kept
            )

            y_log = np.log1p(y)
            X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

            # # Then scale using only training data
            # scaler = StandardScaler()
            # # X_train_scaled = scaler.fit_transform(X_train)
            # X_train_scaled = pd.DataFrame(
            # scaler.fit_transform(X_train),
            # columns=X_train.columns,
            # index=X_train.index
            # )
            # # Use same scaler on test data (only transform, not fit)
            # # X_test_scaled = scaler.transform(X_test)    
            # X_test_scaled = pd.DataFrame(
            # scaler.transform(X_test),
            # columns=X_test.columns,
            # index=X_test.index
            # )

            # 3. Fit the preprocessor and transform the training/test data
            # This replaces your old StandardScaler logic
            X_train_preprocessed = preprocessor.fit_transform(X_train)
            X_test_preprocessed = preprocessor.transform(X_test)
            
            processed_feature_names = preprocessor.get_feature_names_out()

            # Analyze feature importance (now potentially returns two dataframes)
            rf_importances, xgb_importances = analyze_feature_importance(X_train_preprocessed, y_train_log, processed_feature_names)

            trained_lr_model = train_evaluate_model_and_predict(X_train_preprocessed, X_test_preprocessed, y_train_log, y_test_log, model_type='linear_regression')
            trained_rf_model = train_evaluate_model_and_predict(X_train_preprocessed, X_test_preprocessed, y_train_log, y_test_log, model_type='random_forest')
            trained_ridge_linear_model = train_evaluate_model_and_predict(
                X_train_preprocessed, X_test_preprocessed, y_train_log, y_test_log,
                model_type='ridge', 
                tune_hyperparameters_ridge=False, 
                ridge_alpha=1.0
                )
            # --- XGBoost Training with Optuna ---
            # Set tune_hyperparameters_xgb to True and use_optuna_for_xgb to True
            trained_xgb_model_optuna = train_evaluate_model_and_predict(
                X_train_preprocessed, X_test_preprocessed, y_train_log, y_test_log, 
                model_type='xgboost', 
                tune_hyperparameters_xgb=False, 
                use_optuna_for_xgb=True,
                n_optuna_trials=500 # Adjust number of trials as needed
            )

            # You can also try GridSearchCV for comparison if you want:
            # trained_xgb_model_grid = train_and_evaluate_model(
            #     X, y, 
            #     model_type='xgboost', 
            #     tune_hyperparameters_xgb=True, 
            #     use_optuna_for_xgb=False 
            # )

            # --- Example Prediction ---
            # You can choose which model to use for prediction, e.g., trained_xgb_model
            # For this example, let's stick with Random Forest, but you can change it.
           
 
        else:
            print("Exiting due to issues in data loading, preprocessing, or no valid features for training.")

# # =============================================================================
# #  Batch Prediction for Lot Size Analysis
# # =============================================================================

# # 1. Define the base characteristics of the property (without the lot size)
# property_base_data = {
#     'SQUARE FEET': 2112,
#     'BEDS': 3,
#     'BATHS': 2,
#     'AGE': 48,
#     'Basement_Level': 2,
#     'Garage_Spaces': 0,
#     'STORAGE_STRUCTURE': 1,
#     'Recently_Renovated': 1,
#     # 'LotSize_Log' will be added in the loop
# }

# # 2. Define the different lot sizes you want to test
# lot_sizes_to_test = [
#     20000,
# 25000,
# 30000,
# 38000,
# 43560,
# 43750,
# 43980,
# 43999,
# 44000,
# 45000,
# 50000,
# 60000,
# 70000,
# 75000,
# 80000,
# 100000,
# ]

# # 3. Create a list to store the results
# price_predictions = []

# print("\n--- Analyzing Price Change with Lot Size ---")

# # 4. Loop through each lot size and predict the price
# for size in lot_sizes_to_test:
#     # Create a copy of the base data for the current iteration
#     current_property_data = property_base_data.copy()
    
#     # Add the log-transformed lot size for the current iteration
#     current_property_data['LotSize_Log'] = np.log1p(size)
    
#     # Create a single-row DataFrame
#     new_data_df = pd.DataFrame([current_property_data])
    
#     # Ensure column order matches the training data
#     new_data_df = new_data_df[actual_training_feature_names]
    
#     # Use the fitted preprocessor to transform the data
#     new_data_preprocessed = preprocessor.transform(new_data_df)
    
#     # Predict the price using your function
#     predicted_price = predict_new_property_price(
#         model_for_prediction,
#         new_data_preprocessed
#     )
    
#     # Store the results
#     price_predictions.append({'Lot Size': size, 'Predicted Price': predicted_price})

# # 5. Print the final results in a clean table
# print("Prediction Results:")
# for result in price_predictions:
#     print(f"Lot Size: {result['Lot Size']:>7,} sqft -> Predicted Price: ${result['Predicted Price']:,.2f}")