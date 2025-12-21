import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

url = "https://github.com/AngelFelixV/RegresionCarros_Angel/blob/main/car_price_prediction%20(1).csv"
df_raw = pd.read_csv(url)

df_raw['Category'].value_counts()

df_raw.isnull().sum()

df_raw.isna().sum()

df_raw.info()

def contador_de_datos(df, columnas, titles=None, max_cols=2):
    n = len(columnas)
    rows = (n + max_cols - 1) // max_cols

    plt.figure(figsize=(5 * max_cols, 4 * rows))

    for i, col in enumerate(columnas):
        plt.subplot(rows, max_cols, i + 1)
        sns.countplot(data=df, x=col, palette='pastel', order=df[col].value_counts().index)
        plt.title(titles[i] if titles else col)
        plt.xticks(rotation=45)
        plt.xlabel("")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.suptitle("Distributions of Applicants", fontsize=16, y=1.02)
    plt.show()

columnas=['Prod. year','Category','Leather interior','Fuel type','Cylinders','Gear box type','Drive wheels','Color','Doors','Airbags']
contador_de_datos(df_raw, columnas)

df_raw.drop(columns=['ID'],inplace=True)

df_raw['Levy'] = df_raw['Levy'].astype(str)
df_raw['Levy']=df_raw['Levy'].str.replace('-','0')

df_raw['Levy']=df_raw['Levy'].astype(float)

df_raw.head()

df_raw['Mileage'] = df_raw['Mileage'].astype(str)
df_raw['Mileage'] = df_raw['Mileage'].str.replace(' km','')

df_raw['Mileage'] = df_raw['Mileage'].astype(float)

df_raw['Doors'] = df_raw['Doors'].str.extract(r'(\d+)').astype(int)

df_raw

df_raw['Turbo_status'] = np.where(df_raw['Engine volume'].str.contains('Turbo', case=False, na=False), 'Turbo', 'No turbo')
df_raw['Engine volume'] = df_raw['Engine volume'].str.extract(r'(\d+\.?\d*)')[0].astype(float)

df_raw

plt.boxplot(df_raw['Price'])
plt.title('Boxplot de Price')
plt.ylabel('Valor')
plt.show()

plt.boxplot(df_raw['Levy'])
plt.title('Boxplot de Levy')
plt.ylabel('Valor')
plt.show()

plt.boxplot(df_raw['Mileage'])
plt.title('Boxplot de Mileage')
plt.ylabel('Valor')
plt.show()

def quitar_outliers(df, variable):
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5*IQR
    limite_superior = Q3 + 1.5*IQR

    df_filtrado = df[(df[variable] >= limite_inferior) & (df[variable] <= limite_superior)]

    return df_filtrado

def quitar_outliers_std(df, variable):
    mean_val = df[variable].mean()
    std_dev = df[variable].std()

    # Define bounds for outliers (e.g., 3 standard deviations from the mean)
    limite_inferior = mean_val - 3 * std_dev
    limite_superior = mean_val + 3 * std_dev

    df_filtrado = df[(df[variable] >= limite_inferior) & (df[variable] <= limite_superior)]

    return df_filtrado

# Apply the new outlier removal function to df_raw
df_raw_std_filtered = df_raw.copy()
df_raw_std_filtered = quitar_outliers_std(df_raw_std_filtered, 'Price')
df_raw_std_filtered = quitar_outliers_std(df_raw_std_filtered, 'Levy')
df_raw_std_filtered = quitar_outliers_std(df_raw_std_filtered, 'Mileage')

print(f"Original DataFrame shape: {df_raw.shape}")
print(f"Filtered DataFrame shape (mean/std method): {df_raw_std_filtered.shape}")
display(df_raw_std_filtered.head())

quitar_outliers(df_raw, 'Price')
quitar_outliers(df_raw, 'Levy')
quitar_outliers(df_raw, 'Mileage')

plt.boxplot(df_raw_std_filtered['Price'])
plt.title('Boxplot de Price')
plt.ylabel('Valor')
plt.show()

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Create a copy to avoid modifying the original filtered DataFrame directly
df_for_encoding = df_raw_std_filtered[['Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color','Turbo_status']].copy()

# Convert 'Prod. year' to string before fitting the encoder
df_for_encoding['Prod. year'] = df_for_encoding['Prod. year'].astype(str)

X_encoded = encoder.fit_transform(df_for_encoding)
encoded_cols = encoder.get_feature_names_out(['Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color','Turbo_status'])
X_encoded = pd.DataFrame(X_encoded, columns=encoded_cols, index=df_raw_std_filtered.index)

df = pd.concat([df_raw_std_filtered.drop(columns=['Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color','Turbo_status']), X_encoded], axis=1)

df

from sklearn.preprocessing import StandardScaler
X = df.drop(columns='Price')
y = df['Price']
scaler = StandardScaler()
X_nor = scaler.fit_transform(X)

y_scaler = StandardScaler()
y_nor = y_scaler.fit_transform(y.values.reshape(-1,1))

X_final = pd.DataFrame(X_nor, columns = X.columns)
y_final = pd.DataFrame(y_nor, columns = ['Price'])

X_final

y_final

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

regresion_multi_var = linear_model.LinearRegression()
regresion_multi_var.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error
y_pred = regresion_multi_var.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)

r2 = regresion_multi_var.score(X_test,y_test)
print('coeficiente de indeterminación R2: {r2}'.format(r2 = r2))

df = pd.concat([df_raw_std_filtered.drop(columns=['Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color','Turbo_status']), X_encoded], axis=1)

df

from sklearn.preprocessing import StandardScaler
X = df.drop(columns='Price')
y = df['Price']
scaler = StandardScaler()
X_nor = scaler.fit_transform(X)

y_scaler = StandardScaler()
y_nor = y_scaler.fit_transform(y.values.reshape(-1,1))

X_final = pd.DataFrame(X_nor, columns = X.columns)
y_final = pd.DataFrame(y_nor, columns = ['Price'])

X_final

y_final

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

regresion_multi_var = linear_model.LinearRegression()
regresion_multi_var.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error
y_pred = regresion_multi_var.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)

r2 = regresion_multi_var.score(X_test,y_test)
print('coeficiente de indeterminación R2: {r2}'.format(r2 = r2))

def predict_new_car_price(car_data):
    """
    Predicts the price of a car based on its features using the trained linear regression model.

    Args:
        car_data (dict): A dictionary containing the features of the car.
                         Example: {'Levy': '1399', 'Manufacturer': 'LEXUS', 'Model': 'RX 450',
                                   'Prod. year': 2010, 'Category': 'Jeep', 'Leather interior': 'Yes',
                                   'Fuel type': 'Hybrid', 'Engine volume': '3.5', 'Mileage': '186005 km',
                                   'Cylinders': 6.0, 'Gear box type': 'Automatic', 'Drive wheels': '4x4',
                                   'Doors': '04-May', 'Wheel': 'Left wheel', 'Color': 'Silver', 'Airbags': 12}

    Returns:
        float: The predicted price of the car.
    """

    # Convert input dictionary to a pandas DataFrame (single row)
    input_df = pd.DataFrame([car_data])

    # Apply the same preprocessing steps as the training data

    # 1. Handle 'Levy' column
    input_df['Levy'] = input_df['Levy'].astype(str)
    input_df['Levy'] = input_df['Levy'].str.replace('-', '0')
    input_df['Levy'] = input_df['Levy'].astype(float)

    # 2. Handle 'Mileage' column
    input_df['Mileage'] = input_df['Mileage'].astype(str)
    input_df['Mileage'] = input_df['Mileage'].str.replace(' km', '')
    input_df['Mileage'] = input_df['Mileage'].astype(float)

    # 3. Handle 'Doors' column
    # Ensure the string extraction for 'Doors' is robust
    input_df['Doors'] = input_df['Doors'].str.extract(r'(\d+)').astype(int)

    # 4. Handle 'Engine volume' and create 'Turbo_status'
    input_df['Turbo_status'] = np.where(input_df['Engine volume'].str.contains('Turbo', case=False, na=False), 'Turbo', 'No turbo')
    input_df['Engine volume'] = input_df['Engine volume'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

    # 5. One-hot encode categorical features using the fitted encoder
    categorical_cols = ['Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color', 'Turbo_status']
    # For 'Prod. year', ensure it's treated as string for encoding if it was during fit
    input_df['Prod. year'] = input_df['Prod. year'].astype(str)

    # Set handle_unknown='error' to raise a ValueError for unknown categories
    encoded_features = encoder.transform(input_df[categorical_cols])
    encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols), index=input_df.index)

    # 6. Select numerical features and concatenate with encoded features
    numerical_cols = ['Levy', 'Engine volume', 'Mileage', 'Cylinders', 'Doors', 'Airbags']
    processed_input_df = pd.concat([input_df[numerical_cols], encoded_features_df], axis=1)

    # 7. Reindex to match the training data columns exactly and fill missing with 0
    #    This handles cases where a new car might have a category not seen during training
    #    or if the order of columns is different.
    processed_input_df = processed_input_df.reindex(columns=X.columns, fill_value=0)

    # 8. Scale all features using the fitted scaler
    scaled_features = scaler.transform(processed_input_df)
    scaled_features_df = pd.DataFrame(scaled_features, columns=X.columns, index=processed_input_df.index)

    # 9. Make prediction using the trained model
    scaled_prediction = regresion_multi_var.predict(scaled_features_df)

    # 10. Inverse transform the scaled prediction to get the original price scale
    original_price = y_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0][0]

    return original_price

sample_car_with_unknown_manufacturer = {
    'Levy': '1399', 'Manufacturer': 'UNKNOWN_MANUFACTURER', 'Model': 'RX 450', 'Prod. year': 2010,
    'Category': 'Jeep', 'Leather interior': 'Yes', 'Fuel type': 'Hybrid', 'Engine volume': '3.5',
    'Mileage': '186005 km', 'Cylinders': 6.0, 'Gear box type': 'Automatic', 'Drive wheels': '4x4',
    'Doors': '04-May', 'Wheel': 'Left wheel', 'Color': 'Silver', 'Airbags': 12
}

try:
    predicted_price = predict_new_car_price(sample_car_with_unknown_manufacturer)
    print(f"Predicted Price: {predicted_price}")
except ValueError as e:
    print("Error")

numerical_cols_for_stats = ['Levy', 'Engine volume', 'Mileage', 'Cylinders', 'Doors', 'Airbags']
categorical_cols_for_stats = ['Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color', 'Turbo_status']

# Calculate means for numerical columns
mean_values = df_raw_std_filtered[numerical_cols_for_stats].mean().to_dict()

# Calculate modes for categorical columns
mode_values = {}
for col in categorical_cols_for_stats:
    mode_values[col] = df_raw_std_filtered[col].mode()[0] # Get the first mode if there are multiple

# Combine all statistics into one dictionary
global_default_values = {**mean_values, **mode_values}

print("Global Default Values:")
for k, v in global_default_values.items():
    print(f"{k}: {v}")

model_specific_defaults = {}
unique_models = df_raw_std_filtered['Model'].unique()

for model in unique_models:
    # Filter the DataFrame for the current model
    model_df = df_raw_std_filtered[df_raw_std_filtered['Model'] == model]

    # Calculate means for numerical columns
    model_mean_values = model_df[numerical_cols_for_stats].mean().to_dict()

    # Calculate modes for categorical columns
    model_mode_values = {}
    for col in categorical_cols_for_stats:
        # Ensure 'Prod. year' is treated as string for mode calculation if it was during encoding
        if col == 'Prod. year':
            model_mode_values[col] = model_df[col].astype(str).mode()[0]
        else:
            model_mode_values[col] = model_df[col].mode()[0]

    # Combine numerical and categorical stats for the current model
    model_specific_defaults[model] = {**model_mean_values, **model_mode_values}

# Display a few examples from the created dictionary to verify
print("Model-Specific Default Values (first 3 models):")
for i, (model_name, defaults) in enumerate(model_specific_defaults.items()):
    if i >= 3: # Limit output to first 3 for brevity
        break
    print(f"\nModel: {model_name}")
    for k, v in defaults.items():
        print(f"  {k}: {v}")

from scipy.stats import t

# Calculate degrees of freedom (df)
n = X_train.shape[0]  # Number of samples
p = X_train.shape[1]  # Number of features
df = n - p - 1

# Calculate the critical t-value for a 95% prediction interval (alpha = 0.05, two-tailed, so 0.975 percentile)
t_critical = t.ppf(0.975, df)

print(f"Degrees of freedom (df): {df}")
print(f"Critical t-value for 95% prediction interval: {t_critical}")

y_train_pred = regresion_multi_var.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

print(f"RMSE on training data: {rmse_train}")

def predict_new_car_price(car_data):
    """
    Predicts the price of a car based on its features using the trained linear regression model.
    Returns the point prediction and its 95% prediction interval.

    Args:
        car_data (dict): A dictionary containing the features of the car.
                         Example: {'Levy': '1399', 'Manufacturer': 'LEXUS', 'Model': 'RX 450',
                                   'Prod. year': 2010, 'Category': 'Jeep', 'Leather interior': 'Yes',
                                   'Fuel type': 'Hybrid', 'Engine volume': '3.5', 'Mileage': '186005 km',
                                   'Cylinders': 6.0, 'Gear box type': 'Automatic', 'Drive wheels': '4x4',
                                   'Doors': '04-May', 'Wheel': 'Left wheel', 'Color': 'Silver', 'Airbags': 12}

    Returns:
        tuple: A tuple containing the predicted price, lower bound of 95% PI, and upper bound of 95% PI.
    """

    # Convert input dictionary to a pandas DataFrame (single row)
    input_df = pd.DataFrame([car_data])

    # Apply the same preprocessing steps as the training data

    # 1. Handle 'Levy' column
    input_df['Levy'] = input_df['Levy'].astype(str)
    input_df['Levy'] = input_df['Levy'].str.replace('-', '0')
    input_df['Levy'] = input_df['Levy'].astype(float)

    # 2. Handle 'Mileage' column
    input_df['Mileage'] = input_df['Mileage'].astype(str)
    input_df['Mileage'] = input_df['Mileage'].str.replace(' km', '')
    input_df['Mileage'] = input_df['Mileage'].astype(float)

    # 3. Handle 'Doors' column
    input_df['Doors'] = input_df['Doors'].str.extract(r'(\d+)')[0].astype(int)

    # 4. Handle 'Engine volume' and create 'Turbo_status'
    input_df['Turbo_status'] = np.where(input_df['Engine volume'].astype(str).str.contains('Turbo', case=False, na=False), 'Turbo', 'No turbo')
    input_df['Engine volume'] = input_df['Engine volume'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

    # 5. One-hot encode categorical features using the fitted encoder
    categorical_cols = ['Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color', 'Turbo_status']
    input_df['Prod. year'] = input_df['Prod. year'].astype(str)

    encoded_features = encoder.transform(input_df[categorical_cols])
    encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols), index=input_df.index)

    # 6. Select numerical features and concatenate with encoded features
    numerical_cols = ['Levy', 'Engine volume', 'Mileage', 'Cylinders', 'Doors', 'Airbags']
    processed_input_df = pd.concat([input_df[numerical_cols], encoded_features_df], axis=1)

    # 7. Reindex to match the training data columns exactly and fill missing with 0
    processed_input_df = processed_input_df.reindex(columns=X.columns, fill_value=0)

    # 8. Scale all features using the fitted scaler
    scaled_features = scaler.transform(processed_input_df)
    scaled_features_df = pd.DataFrame(scaled_features, columns=X.columns, index=processed_input_df.index)

    # 9. Make prediction using the trained model (scaled output)
    scaled_prediction = regresion_multi_var.predict(scaled_features_df)

    # 10. Inverse transform the scaled prediction to get the original price scale
    original_price = y_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0][0]

    # 11. Calculate the prediction interval
    rmse_original_scale = rmse_train * y_scaler.scale_[0]

    margin_of_error = t_critical * rmse_original_scale
    lower_bound = original_price - margin_of_error
    upper_bound = original_price + margin_of_error

    # Ensure prices are not negative
    original_price = max(0, original_price)
    lower_bound = max(0, lower_bound)
    upper_bound = max(0, upper_bound)

    return original_price, lower_bound, upper_bound

print("The predict_new_car_price function has been updated to cap predicted prices at zero.")

