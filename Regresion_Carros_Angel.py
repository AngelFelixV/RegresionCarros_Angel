# =========================
# COMPLETE ONE-CELL PIPELINE
# =========================

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# LOAD DATA
# -------------------------
url = "https://raw.githubusercontent.com/AngelFelixV/RegresionCarros_Angel/main/car_price_prediction%20(1).csv"
df = pd.read_csv(url)

# -------------------------
# BASIC CLEANING
# -------------------------
df.drop(columns=['ID'], inplace=True)

df['Levy'] = df['Levy'].astype(str).str.replace('-', '0').astype(float)
df['Mileage'] = df['Mileage'].astype(str).str.replace(' km', '').astype(float)
df['Doors'] = df['Doors'].astype(str).str.extract(r'(\d+)').astype(int)

df['Turbo_status'] = np.where(
    df['Engine volume'].astype(str).str.contains('Turbo', case=False, na=False),
    'Turbo', 'No turbo'
)
df['Engine volume'] = df['Engine volume'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

# -------------------------
# OUTLIER REMOVAL (STD)
# -------------------------
for col in ['Price', 'Levy', 'Mileage']:
    m, s = df[col].mean(), df[col].std()
    df = df[(df[col] >= m - 3*s) & (df[col] <= m + 3*s)]

# -------------------------
# ENCODING
# -------------------------
categorical_cols = [
    'Manufacturer','Model','Prod. year','Category',
    'Leather interior','Fuel type','Gear box type',
    'Drive wheels','Wheel','Color','Turbo_status'
]

df['Prod. year'] = df['Prod. year'].astype(str)

encoder = OneHotEncoder(
    drop='first',
    sparse_output=False,
    handle_unknown='ignore'
)

X_cat = encoder.fit_transform(df[categorical_cols])
X_cat = pd.DataFrame(
    X_cat,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=df.index
)

numerical_cols = ['Levy','Engine volume','Mileage','Cylinders','Doors','Airbags']
X = pd.concat([df[numerical_cols], X_cat], axis=1)
y = df['Price']

# -------------------------
# SCALING
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1,1))

# -------------------------
# TRAIN MODEL
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
print(f"RMSE (scaled): {rmse:.4f}")

# -------------------------
# SMART DEFAULTS
# -------------------------
global_defaults = {}
for col in numerical_cols:
    global_defaults[col] = df[col].mean()
for col in categorical_cols:
    global_defaults[col] = df[col].mode()[0]

model_defaults = {}
for m in df['Model'].unique():
    d = df[df['Model'] == m]
    defaults = {}
    for col in numerical_cols:
        defaults[col] = d[col].mean()
    for col in categorical_cols:
        defaults[col] = d[col].mode()[0]
    model_defaults[m] = defaults

# -------------------------
# BUILD FULL INPUT
# -------------------------
def build_complete_car_input(user_input):
    base = model_defaults.get(user_input['Model'], global_defaults).copy()
    base.update(user_input)
    base['Mileage'] = f"{int(base['Mileage'])} km"
    base['Engine volume'] = str(base['Engine volume'])
    base['Doors'] = str(int(base['Doors']))
    return base

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_price(user_input):

    car = build_complete_car_input(user_input)
    df_in = pd.DataFrame([car])

    df_in['Levy'] = df_in['Levy'].astype(float)
    df_in['Mileage'] = df_in['Mileage'].str.replace(' km','').astype(float)
    df_in['Doors'] = df_in['Doors'].astype(str).str.extract(r'(\d+)').astype(int)

    df_in['Turbo_status'] = np.where(
        df_in['Engine volume'].str.contains('Turbo', case=False, na=False),
        'Turbo','No turbo'
    )
    df_in['Engine volume'] = df_in['Engine volume'].str.extract(r'(\d+\.?\d*)').astype(float)

    df_in['Prod. year'] = df_in['Prod. year'].astype(str)

    X_cat_in = encoder.transform(df_in[categorical_cols])
    X_cat_in = pd.DataFrame(
        X_cat_in,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    X_num_in = df_in[numerical_cols]
    X_final = pd.concat([X_num_in, X_cat_in], axis=1)
    X_final = X_final.reindex(columns=X.columns, fill_value=0)

    X_scaled_in = scaler.transform(X_final)
    y_scaled_pred = model.predict(X_scaled_in)

    price = y_scaler.inverse_transform(y_scaled_pred.reshape(-1,1))[0][0]
    return max(0, price)

# -------------------------
# EXAMPLE (USER INPUT ONLY)
# -------------------------
user_input = {
    'Manufacturer': 'LEXUS',
    'Model': 'RX 450',
    'Category': 'Jeep',
    'Fuel type': 'Hybrid',
    'Gear box type': 'Automatic',
    'Color': 'Silver'
}

print(f"Predicted Price: ${predict_price(user_input):,.0f}")

