import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ======================
# Cargar objetos
# ======================
modelo = joblib.load("modelos_auto.pkl")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("y_scaler.pkl")
encoder = joblib.load("encoder.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

# ======================
# Interfaz
# ======================
st.title("🚗 Predicción de Precio de Autos")

st.write("Introduce las características del auto:")

# ----------------------
# Variables numéricas
# ----------------------
levy = st.number_input("Levy", min_value=0.0, value=500.0)
mileage = st.number_input("Kilometraje", min_value=0.0, value=50000.0)
engine_volume = st.number_input("Volumen del motor (L)", min_value=0.5, max_value=8.0, value=2.0)
doors = st.selectbox("Número de puertas", [2, 3, 4, 5])
cylinders = st.number_input("Cilindros", min_value=2, max_value=16, value=4)
airbags = st.number_input("Número de airbags", min_value=0, max_value=20, value=6)

# ----------------------
# Variables categóricas
# ----------------------
manufacturer = st.selectbox("Marca", encoder.categories_[0])
model = st.selectbox("Modelo", encoder.categories_[1])
prod_year = st.selectbox("Año de producción", encoder.categories_[2])
category = st.selectbox("Categoría", encoder.categories_[3])
leather = st.selectbox("Interior de cuero", encoder.categories_[4])
fuel = st.selectbox("Tipo de combustible", encoder.categories_[5])
gear = st.selectbox("Tipo de transmisión", encoder.categories_[6])
drive = st.selectbox("Tracción", encoder.categories_[7])
wheel = st.selectbox("Volante", encoder.categories_[8])
color = st.selectbox("Color", encoder.categories_[9])
turbo = st.selectbox("Turbo", encoder.categories_[10])

def plot_colors_grid(colors, title="Available Car Colors", ncols=5, figsize=(10, 6)):
    """
    Plots a grid of color swatches with their names.

    Args:
        colors (list): A list of color names (strings).
        title (str): Title of the plot.
        ncols (int): Number of columns in the grid.
        figsize (tuple): Figure size.
    """
    nrows = int(np.ceil(len(colors) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, color_name in enumerate(colors):
        if i >= len(axes): # Prevent index out of bounds for axes
            break

        ax = axes[i]
        try:
            # Try to use the color name directly if matplotlib recognizes it
            color_val = color_name.lower().replace(' ', '')
            if color_val == 'gray': # Gray is 'grey' in some contexts, ensure consistency
                color_val = 'grey'
            if color_val == 'darkblue': # common variation
                color_val = 'midnightblue'
            if color_val == 'silver':
                color_val = '#C0C0C0' # Hex code for silver
            if color_val == 'golden':
                color_val = '#FFD700' # Hex code for golden
            if color_val == 'beig':
                color_val = '#F5F5DC' # Hex code for beige
            if color_val == 'orange':
                color_val = '#FFA500'

            # Create a Rectangle patch to give it a defined shape
            patch = mpatches.Rectangle((0, 0), 1, 1, facecolor=color_val, edgecolor='black', linewidth=1)
        except ValueError:
            # If not a recognized color name, use a default (e.g., grey) and print a warning
            print(f"Warning: '{color_name}' not a standard matplotlib color. Using grey.")
            patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='grey', edgecolor='black', linewidth=1)

        ax.add_patch(patch)
        ax.set_title(color_name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off') # Hide axes ticks and labels

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Get unique colors from the DataFrame
# Ensure df_raw is loaded. If not, the previous %run Regresion_Carros_Angel.py should have loaded it.
if 'df_raw' in locals():
    unique_colors = df_raw['Color'].unique().tolist()
    plot_colors_grid(unique_colors)
else:
    print("df_raw DataFrame not found. Please ensure the data loading script has run.")

# ======================
# Predicción
# ======================
if st.button("Predecir precio"):
    
    # ----------------------
    # Parte numérica
    # ----------------------
    X_base = pd.DataFrame(
    np.zeros((1, len(columnas_modelo))),
    columns=columnas_modelo
)

# Numéricas
X_base.loc[0, 'Levy'] = levy
X_base.loc[0, 'Mileage'] = mileage
X_base.loc[0, 'Engine volume'] = engine_volume
X_base.loc[0, 'Doors'] = doors
X_base.loc[0, 'Cylinders'] = cylinders
X_base.loc[0, 'Airbags'] = airbags

# Categóricas
X_cat = pd.DataFrame([[
    manufacturer, model, prod_year, category,
    leather, fuel, gear, drive, wheel, color, turbo
]], columns=encoder.feature_names_in_)

X_cat_encoded = encoder.transform(X_cat)
X_cat_encoded = pd.DataFrame(
    X_cat_encoded,
    columns=encoder.get_feature_names_out(),
    index=X_base.index
)

X_base[X_cat_encoded.columns] = X_cat_encoded

# Escalado
X_scaled = scaler_X.transform(X_base)

# Predicción
y_pred_scaled = modelo.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

st.success(f"💰 Precio estimado: ${y_pred[0][0]:,.2f}")
