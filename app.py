import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# ======================
# LOAD & PREPROCESS DATA
# ======================
@st.cache_data
def load_and_train():
    url = "https://raw.githubusercontent.com/AngelFelixV/RegresionCarros_Angel/main/car_price_prediction%20(1).csv"
    df = pd.read_csv(url)

    df.drop(columns=['ID'], inplace=True)
    df['Levy'] = df['Levy'].astype(str).str.replace('-', '0').astype(float)
    df['Mileage'] = df['Mileage'].astype(str).str.replace(' km', '').astype(float)
    df['Doors'] = df['Doors'].astype(str).str.extract(r'(\d+)').astype(int)

    df['Turbo_status'] = np.where(
        df['Engine volume'].astype(str).str.contains('Turbo', case=False, na=False),
        'Turbo', 'No turbo'
    )
    df['Engine volume'] = df['Engine volume'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)

    for col in ['Price', 'Levy', 'Mileage']:
        m, s = df[col].mean(), df[col].std()
        df = df[(df[col] >= m - 3*s) & (df[col] <= m + 3*s)]

    categorical_cols = [
        'Manufacturer','Model','Prod. year','Category',
        'Leather interior','Fuel type','Gear box type',
        'Drive wheels','Wheel','Color','Turbo_status'
    ]

    numerical_cols = ['Levy','Engine volume','Mileage','Cylinders','Doors','Airbags']
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

    X = pd.concat([df[numerical_cols], X_cat], axis=1)
    y = df['Price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1,1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Defaults
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

    return df, model, encoder, scaler, y_scaler, X.columns, categorical_cols, numerical_cols, model_defaults, global_defaults

df, model, encoder, scaler, y_scaler, X_cols, cat_cols, num_cols, model_defaults, global_defaults = load_and_train()

# ======================
# UI
# ======================
st.title("🚗 Car Price Prediction App")

col1, col2 = st.columns(2)

with col1:
    manufacturer = st.selectbox(
        "Manufacturer",
        sorted(df['Manufacturer'].unique())
    )

    models = sorted(df[df['Manufacturer'] == manufacturer]['Model'].unique())
    model_sel = st.selectbox("Model", models)

    fuel = st.selectbox("Fuel Type", sorted(df['Fuel type'].unique()))
    gear = st.selectbox("Gear Box Type", sorted(df['Gear box type'].unique()))

with col2:
    st.markdown("### 🚙 Category")
    categories = sorted(df['Category'].unique())
    selected_category = st.radio("", categories)

    st.markdown("### 🎨 Color")
    colors = sorted(df['Color'].unique())

    cols = st.columns(6)
    selected_color = colors[0]
    for i, c in enumerate(colors):
        with cols[i % 6]:
            if st.button(c):
                selected_color = c
            st.markdown(
                f"""
                <div style="
                    background-color:{c.lower().replace(' ','')};
                    width:40px;
                    height:40px;
                    border-radius:6px;
                    border:2px solid #555;
                    margin:auto;">
                </div>
                """,
                unsafe_allow_html=True
            )

# ======================
# PREDICTION
# ======================
def predict_price(user_input):
    base = model_defaults.get(user_input['Model'], global_defaults).copy()
    base.update(user_input)

    base['Mileage'] = f"{int(base['Mileage'])} km"
    base['Engine volume'] = str(base['Engine volume'])
    base['Doors'] = str(int(base['Doors']))

    df_in = pd.DataFrame([base])
    df_in['Levy'] = df_in['Levy'].astype(float)
    df_in['Mileage'] = df_in['Mileage'].str.replace(' km','').astype(float)
    df_in['Doors'] = df_in['Doors'].str.extract(r'(\d+)').astype(int)

    df_in['Turbo_status'] = np.where(
        df_in['Engine volume'].str.contains('Turbo', case=False, na=False),
        'Turbo','No turbo'
    )
    df_in['Engine volume'] = df_in['Engine volume'].str.extract(r'(\d+\.?\d*)').astype(float)
    df_in['Prod. year'] = df_in['Prod. year'].astype(str)

    X_cat = encoder.transform(df_in[cat_cols])
    X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols))
    X_num = df_in[num_cols]

    X_final = pd.concat([X_num, X_cat], axis=1)
    X_final = X_final.reindex(columns=X_cols, fill_value=0)

    X_scaled = scaler.transform(X_final)
    y_scaled = model.predict(X_scaled)

    return max(0, y_scaler.inverse_transform(y_scaled.reshape(-1,1))[0][0])

# ======================
# BUTTON
# ======================
if st.button("💰 Predict Price"):
    user_input = {
        'Manufacturer': manufacturer,
        'Model': model_sel,
        'Category': selected_category,
        'Fuel type': fuel,
        'Gear box type': gear,
        'Color': selected_color
    }

    price = predict_price(user_input)

    st.success(f"Estimated Price: ${price:,.0f}")
