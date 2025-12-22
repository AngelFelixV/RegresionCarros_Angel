import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# ======================
# COLOR MAP
# ======================
COLOR_MAP = {
    "Black": "#000000",
    "White": "#E6E6E6",
    "Silver": "#BFC1C2",
    "Grey": "#7A7A7A",
    "Gray": "#7A7A7A",
    "Blue": "#1f77b4",
    "Red": "#d62728",
    "Carmelian red": "#8B0000",
    "Green": "#2ca02c",
    "Yellow": "#E6C200",
    "Golden": "#C9A227",
    "Orange": "#FF8C00",
    "Brown": "#8B4513",
    "Light beige": "#D8CFC4",
    "Pink": "#FFB6C1",
    "Purple": "#7B3F99",
}

# ======================
# CAR ICON
# ======================
def car_icon(color_hex, category):
    shape = {
        "Sedan": "border-radius:20px 20px 10px 10px;",
        "Jeep": "border-radius:8px;",
        "Hatchback": "border-radius:16px;",
        "Coupe": "border-radius:25px;",
        "Universal": "border-radius:10px;",
    }.get(category, "border-radius:12px;")

    return f"""
    <div style="
        background:{color_hex};
        width:100px;
        height:45px;
        {shape}
        box-shadow:0 4px 10px rgba(0,0,0,0.6);
        margin:auto;
    "></div>
    <p style="text-align:center;color:#ccc">{category}</p>
    """

# ======================
# LOAD & TRAIN
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
        df['Engine volume'].astype(str).str.contains('Turbo', case=False),
        'Turbo', 'No turbo'
    )
    df['Engine volume'] = df['Engine volume'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    df['Prod. year'] = df['Prod. year'].astype(str)

    cat_cols = [
        'Manufacturer','Model','Prod. year','Category',
        'Leather interior','Fuel type','Gear box type',
        'Drive wheels','Wheel','Color','Turbo_status'
    ]
    num_cols = ['Levy','Engine volume','Mileage','Cylinders','Doors','Airbags']

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(df[cat_cols])
    X = pd.concat(
        [df[num_cols].reset_index(drop=True),
         pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols))],
        axis=1
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(df['Price'].values.reshape(-1,1))

    X_train, _, y_train, _ = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    defaults = df[num_cols].mean().to_dict()
    defaults.update(df[cat_cols].mode().iloc[0].to_dict())

    return df, model, encoder, scaler, y_scaler, X.columns, cat_cols, num_cols, defaults

df, model, encoder, scaler, y_scaler, X_cols, cat_cols, num_cols, defaults = load_and_train()

# ======================
# UI
# ======================
st.title("🚗 Car Price Prediction App")

c1, c2 = st.columns(2)

with c1:
    manufacturer = st.selectbox("Manufacturer", sorted(df['Manufacturer'].unique()))
    model_sel = st.selectbox("Model", sorted(df[df['Manufacturer'] == manufacturer]['Model'].unique()))
    fuel = st.selectbox("Fuel Type", sorted(df['Fuel type'].unique()))
    gear = st.selectbox("Gear Box Type", sorted(df['Gear box type'].unique()))

with c2:
    selected_category = st.radio("Category", sorted(df['Category'].unique()))
    colors = sorted(df['Color'].unique())
    cols = st.columns(6)

   st.markdown("### 🎨 Color")

colors = sorted(df['Color'].dropna().unique())
cols = st.columns(6)

for i, c in enumerate(colors):
    hex_color = COLOR_MAP.get(c, "#999999")

    with cols[i % 6]:
        if st.button(" ", key=f"color_{c}"):
            st.session_state.selected_color = c

        st.markdown(
            f"""
            <div style="
                background-color:{hex_color};
                width:42px;
                height:42px;
                border-radius:8px;
                border:{'3px solid black' if st.session_state.selected_color == c else '1px solid #555'};
                margin:auto;
            "></div>
            <p style="text-align:center;font-size:12px">{c}</p>
            """,
            unsafe_allow_html=True
        )


# ======================
# PREVIEW
# ======================
st.markdown("### 🚘 Your Selected Car")
st.markdown(
    car_icon(COLOR_MAP.get(selected_color, "#999999"), selected_category),
    unsafe_allow_html=True
)

# ======================
# PREDICT
# ======================
def predict_price(user_input):
    df_in = pd.DataFrame([user_input])
    X_cat = encoder.transform(df_in[cat_cols])
    X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols))
    X_final = pd.concat([df_in[num_cols], X_cat], axis=1)
    X_final = X_final.reindex(columns=X_cols, fill_value=0)
    y_scaled = model.predict(scaler.transform(X_final))
    return max(0, y_scaler.inverse_transform(y_scaled.reshape(-1,1))[0][0])

if st.button("💰 Predict Price"):
    user_input = defaults.copy()
    user_input.update({
        'Manufacturer': manufacturer,
        'Model': model_sel,
        'Category': selected_category,
        'Fuel type': fuel,
        'Gear box type': gear,
        'Color': selected_color
    })

    price = predict_price(user_input)
    st.success(f"Estimated Price: ${price:,.0f}")

