# app.py
# Streamlit app using RandomForestRegressor + color selector

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Car Price Prediction App (Random Forest)")
st.markdown("Predict car prices using a **Random Forest Regressor** and explore the effect of **car color** interactively.")

@st.cache_data

def load_data():
    url = "https://raw.githubusercontent.com/AngelFelixV/RegresionCarros_Angel/main/car_price_prediction%20(1).csv"
    df = pd.read_csv(url)
    df = df.drop(columns=['ID'])

    df['Levy'] = df['Levy'].astype(str).str.replace('-', '0').astype(float)
    df['Mileage'] = df['Mileage'].astype(str).str.replace(' km', '').astype(float)
    df['Doors'] = df['Doors'].str.extract(r'(\d+)').astype(int)

    df['Turbo_status'] = np.where(
        df['Engine volume'].astype(str).str.contains('Turbo', case=False, na=False),
        'Turbo', 'No turbo'
    )
    df['Engine volume'] = df['Engine volume'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

    return df


df = load_data()

st.sidebar.header("🎛 Model Settings")
n_estimators = st.sidebar.slider("Number of trees", 100, 600, 300, step=50)
max_depth = st.sidebar.slider("Max depth", 5, 40, 20)

categorical_cols = [
    'Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior',
    'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color', 'Turbo_status'
]

numerical_cols = ['Levy', 'Engine volume', 'Mileage', 'Cylinders', 'Doors', 'Airbags']

df['Prod. year'] = df['Prod. year'].astype(str)

encoder = OneHotEncoder(drop='first', sparse_output=False)
X_cat = encoder.fit_transform(df[categorical_cols])
X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols))

X = pd.concat([df[numerical_cols].reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("RMSE", f"{rmse:,.0f}")
col2.metric("R²", f"{r2:.3f}")

st.divider()

st.subheader("🔮 Predict a New Car Price")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    manufacturer = c1.selectbox("Manufacturer", sorted(df['Manufacturer'].unique()))
    model_name = c2.selectbox("Model", sorted(df['Model'].unique()))
    prod_year = c3.selectbox("Production year", sorted(df['Prod. year'].unique()))

    category = c1.selectbox("Category", sorted(df['Category'].unique()))
    fuel = c2.selectbox("Fuel type", sorted(df['Fuel type'].unique()))
    gearbox = c3.selectbox("Gear box", sorted(df['Gear box type'].unique()))

    drive = c1.selectbox("Drive wheels", sorted(df['Drive wheels'].unique()))
    wheel = c2.selectbox("Wheel", sorted(df['Wheel'].unique()))

    # 🎨 Color selector (key requirement)
    color = c3.selectbox("Car color", sorted(df['Color'].unique()))

    leather = c1.selectbox("Leather interior", ['Yes', 'No'])
    turbo = c2.selectbox("Turbo", ['Turbo', 'No turbo'])

    levy = c3.number_input("Levy", value=1000.0)
    engine = c1.number_input("Engine volume", value=2.0)
    mileage = c2.number_input("Mileage", value=150000.0)
    cylinders = c3.number_input("Cylinders", value=4.0)
    doors = c1.number_input("Doors", value=4)
    airbags = c2.number_input("Airbags", value=6)

    submitted = st.form_submit_button("Predict price")

if submitted:
    input_dict = {
        'Manufacturer': manufacturer,
        'Model': model_name,
        'Prod. year': prod_year,
        'Category': category,
        'Leather interior': leather,
        'Fuel type': fuel,
        'Gear box type': gearbox,
        'Drive wheels': drive,
        'Wheel': wheel,
        'Color': color,
        'Turbo_status': turbo
    }

    input_cat = encoder.transform(pd.DataFrame([input_dict]))
    input_cat = pd.DataFrame(input_cat, columns=encoder.get_feature_names_out())

    input_num = pd.DataFrame([[levy, engine, mileage, cylinders, doors, airbags]], columns=numerical_cols)

    X_input = pd.concat([input_num, input_cat], axis=1)
    X_input = X_input.reindex(columns=X.columns, fill_value=0)

    price = model.predict(X_input)[0]

    st.success(f"💰 Estimated price: **${price:,.0f}**")

    st.caption("Random Forest models naturally capture non-linear effects like color, fuel type, and interactions.")

