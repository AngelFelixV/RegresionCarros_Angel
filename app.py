import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# ======================
# COLOR MAP (SAFE)
# ======================
COLOR_MAP = {
    "Black": "#000000",
    "White": "#FFFFFF",
    "Silver": "#C0C0C0",
    "Grey": "#808080",
    "Gray": "#808080",
    "Blue": "#1f77b4",
    "Red": "#d62728",
    "Carmelian Red": "#960018",
    "Green": "#2ca02c",
    "Yellow": "#FFD700",
    "Golden": "#FFD700",
    "Orange": "#FFA500",
    "Brown": "#8B4513",
    "Beige": "#F5F5DC",
    "Pink": "#FFC0CB",
    "Purple": "#800080",
}

# ======================
# CAR ICON
# ======================
def car_icon(color_hex, category):
    category = str(category).lower()
    if "jeep" in category or "suv" in category:
        emoji = "🚙"
    elif "truck" in category or "pickup" in category:
        emoji = "🚚"
    elif "bus" in category:
        emoji = "🚌"
    else:
        emoji = "🚗"

    return f"""
    <div style="font-size:100px; text-align:center;
                filter: drop-shadow(0 0 6px {color_hex});">
        {emoji}
    </div>
    """

# ======================
# LOAD & TRAIN
# ======================
@st.cache_data
def load_and_train():
    url = "https://raw.githubusercontent.com/AngelFelixV/RegresionCarros_Angel/main/car_price_prediction%20(1).csv"
    df = pd.read_csv(url)

    df.drop(columns=["ID"], inplace=True)
    df["Color"] = df["Color"].astype(str).str.strip().str.title()

    df["Levy"] = df["Levy"].astype(str).str.replace("-", "0").astype(float)
    df["Mileage"] = df["Mileage"].astype(str).str.replace(" km", "").astype(float)
    df["Doors"] = df["Doors"].astype(str).str.extract(r"(\d+)").astype(int)

    df["Turbo_status"] = np.where(
        df["Engine volume"].astype(str).str.contains("Turbo", case=False),
        "Turbo", "No turbo"
    )
    df["Engine volume"] = df["Engine volume"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)

    for col in ["Price", "Levy", "Mileage"]:
        m, s = df[col].mean(), df[col].std()
        df = df[(df[col] >= m - 3*s) & (df[col] <= m + 3*s)]

    cat_cols = [
        "Manufacturer","Model","Prod. year","Category",
        "Leather interior","Fuel type","Gear box type",
        "Drive wheels","Wheel","Color","Turbo_status"
    ]
    num_cols = ["Levy","Engine volume","Mileage","Cylinders","Doors","Airbags"]

    df["Prod. year"] = df["Prod. year"].astype(str)

    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[cat_cols])
    X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols))

    X = pd.concat([df[num_cols].reset_index(drop=True), X_cat], axis=1)
    y = df["Price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1,1))

    X_train, _, y_train, _ = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    defaults = {}
    for c in num_cols:
        defaults[c] = df[c].mean()
    for c in cat_cols:
        defaults[c] = df[c].mode()[0]

    return df, model, encoder, scaler, y_scaler, X.columns, cat_cols, num_cols, defaults

df, model, encoder, scaler, y_scaler, X_cols, cat_cols, num_cols, defaults = load_and_train()

# ======================
# SESSION STATE INIT
# ======================
if "selected_color" not in st.session_state:
    st.session_state["selected_color"] = df["Color"].mode()[0]

# ======================
# UI
# ======================
st.title("🚗 Car Price Prediction App")

c1, c2 = st.columns(2)

with c1:
    manufacturer = st.selectbox("Manufacturer", sorted(df["Manufacturer"].unique()))
    models = sorted(df[df["Manufacturer"] == manufacturer]["Model"].unique())
    model_sel = st.selectbox("Model", models)
    fuel = st.selectbox("Fuel Type", sorted(df["Fuel type"].unique()))
    gear = st.selectbox("Gear Box Type", sorted(df["Gear box type"].unique()))

with c2:
    st.markdown("### 🚙 Category")
    selected_category = st.radio("", sorted(df["Category"].unique()))

# ======================
# COLOR GRID
# ======================
st.markdown("### 🎨 Color")
colors = sorted(df["Color"].dropna().unique())
cols = st.columns(6)

for i, c in enumerate(colors):
    hex_color = COLOR_MAP.get(c, "#999999")

    with cols[i % 6]:
        if st.button(" ", key=f"color_{c}"):
            st.session_state["selected_color"] = c

        st.markdown(
            f"""
            <div style="
                background-color:{hex_color};
                width:42px;
                height:42px;
                border-radius:8px;
                border:{'3px solid black' if st.session_state["selected_color"] == c else '1px solid #555'};
                margin:auto;
            "></div>
            <p style="text-align:center;font-size:12px">{c}</p>
            """,
            unsafe_allow_html=True
        )

# ======================
# CAR PREVIEW
# ======================
st.markdown(
    car_icon(
        COLOR_MAP.get(st.session_state["selected_color"], "#999999"),
        selected_category
    ),
    unsafe_allow_html=True
)

# ======================
# PREDICTION
# ======================
def predict_price(user):
    base = defaults.copy()
    base.update(user)

    df_in = pd.DataFrame([base])
    df_in["Prod. year"] = df_in["Prod. year"].astype(str)

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
        "Manufacturer": manufacturer,
        "Model": model_sel,
        "Category": selected_category,
        "Fuel type": fuel,
        "Gear box type": gear,
        "Color": st.session_state["selected_color"]
    }

    price = predict_price(user_input)
    st.success(f"Estimated Price: ${price:,.0f}"
