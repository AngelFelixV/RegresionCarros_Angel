# RegresionCarros_Angel
# Regresión Carros - Predicción de Precios de Autos Semi-nuevos

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)](https://regresioncarrosangel-dwdr7sjni26n6rnf332k5v.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

---

## Descripción del Proyecto

**RegresionCarros_Angel** es una aplicación web basada en **machine learning** que predice el precio de autos semi-nuevos utilizando un modelo de **regresión lineal**. La aplicación integra un pipeline completo de procesamiento de datos, entrenamiento del modelo y una interfaz interactiva construida con **Streamlit** para facilitar predicciones en tiempo real.

### Características Principales

- Predicción Inteligente: Modelo entrenado con datos reales de mercado
- Interfaz Visual: Aplicación web interactiva y amigable
- Manejo Robusto de Datos: Limpieza, encoding y normalización automática
- Escalabilidad: Preparado para nuevos datos y características
- Información Detallada: Validación y métricas de desempeño incluidas

---

## Pipeline de Machine Learning

### 1. Carga de Datos
- Fuente: Dataset CSV con más de 3,000 registros de autos
- Variables: 15+ características incluyendo fabricante, modelo, año, combustible, etc.

### 2. Limpieza y Preprocesamiento
```python
# Eliminación de columnas innecesarias
df.drop(columns=['ID'], inplace=True)

# Conversión y limpieza de datos:
- Levy: Eliminación de valores faltantes ("-" → 0)
- Mileage: Extracción de valor numérico ("120000 km" → 120000)
- Doors: Extracción de número de puertas ("4-5" → 4)
- Engine volume: Separación de turbo y volumen
```

### 3. Feature Engineering
- Nueva Variable: Turbo_status (Turbo / No turbo)
- Variables Numéricas: Levy, Engine volume, Mileage, Cylinders, Doors, Airbags
- Variables Categóricas: Manufacturer, Model, Prod. year, Category, Leather interior, Fuel type, Gear box type, Drive wheels, Wheel, Color, Turbo_status

### 4. Tratamiento de Outliers
Eliminación de valores atípicos usando el método de desviación estándar (3σ):
```python
for col in ['Price', 'Levy', 'Mileage']:
    m, s = df[col].mean(), df[col].std()
    df = df[(df[col] >= m - 3*s) & (df[col] <= m + 3*s)]
```

### 5. Encoding de Variables Categóricas
- Método: OneHotEncoder con `drop='first'` para evitar multicolinealidad
- Salida: 40+ características binarizadas

### 6. Normalización
```python
# Variables independientes (X)
X_scaled = StandardScaler().fit_transform(X)

# Variable objetivo (y)
y_scaled = StandardScaler().fit_transform(y)
```

### 7. Entrenamiento del Modelo
- Algoritmo: Linear Regression
- Split: 80% entrenamiento, 20% prueba
- Métrica: RMSE en escala normalizada

### 8. Predicción
El modelo utiliza valores por defecto inteligentes (media/moda por modelo) para características faltantes.

---

## Estructura del Proyecto

```
RegresionCarros_Angel/
├── app.py                              # Aplicación Streamlit
├── Regresion_Carros_Angel.py          # Pipeline completo de ML
├── requirements.txt                    # Dependencias
├── car_price_prediction.csv           # Dataset de entrenamiento
│
├── Modelos Entrenados (PKL):
├── linear_regression_model.pkl        # Modelo Linear Regression
├── x_scaler.pkl                       # StandardScaler para features
├── y_scaler_target.pkl                # StandardScaler para precio
├── onehot_encoder.pkl                 # OneHotEncoder
└── model_feature_columns.pkl          # Nombres de columnas
```

---

## Instalación y Configuración

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes)

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/AngelFelixV/RegresionCarros_Angel.git
cd RegresionCarros_Angel
```

### Paso 2: Crear Entorno Virtual (Recomendado)
```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

### Paso 3: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Ejecutar la Aplicación
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

---

## Cómo Usar la Aplicación

### Interfaz Principal

1. Selecciona Fabricante
   - Elige entre 34+ marcas disponibles

2. Selecciona Modelo
   - Los modelos se filtran automáticamente según el fabricante

3. Elige Categoría
   - Sedán, SUV, Jeep, Truck, etc.

4. Especifica Características
   - Tipo de Combustible (Gasolina, Diésel, Híbrido, etc.)
   - Tipo de Transmisión (Manual, Automática)
   - Color del Vehículo (con vista previa visual)

5. Obtén la Predicción
   - Haz clic en "Predict Price"
   - Recibe el precio estimado en USD

### Ejemplo de Entrada
```python
user_input = {
    'Manufacturer': 'LEXUS',
    'Model': 'RX 450',
    'Category': 'Jeep',
    'Fuel type': 'Hybrid',
    'Gear box type': 'Automatic',
    'Color': 'Silver'
}
# Resultado: Precio estimado
```

---

## Uso Programático

### Ejecutar el Pipeline Completo
```bash
python Regresion_Carros_Angel.py
```

### Integrar en tu Código
```python
from Regresion_Carros_Angel import predict_price

# Hacer predicción
user_input = {
    'Manufacturer': 'TOYOTA',
    'Model': 'CAMRY',
    'Category': 'Sedan',
    'Fuel type': 'Petrol',
    'Gear box type': 'Automatic',
    'Color': 'White'
}

price = predict_price(user_input)
print(f"Precio Estimado: ${price:,.0f}")
```

---

## Desempeño del Modelo

| Métrica | Valor |
|---------|-------|
| **Algoritmo** | Linear Regression |
| **Dataset** | 2,900+ registros |
| **Features** | 40+ (después de encoding) |
| **Train/Test Split** | 80/20 |
| **Normalización** | StandardScaler |
| **Manejo de Outliers** | 3σ method |

**Nota**: Los valores de RMSE se calculan en escala normalizada. Para métricas finales, consultar logs de entrenamiento.

---

## Dependencias Principales

| Paquete | Versión | Propósito |
|---------|---------|----------|
| **streamlit** | Latest | Interfaz web |
| **pandas** | Latest | Manipulación de datos |
| **numpy** | Latest | Operaciones numéricas |
| **scikit-learn** | Latest | Machine Learning |
| **matplotlib** | Latest | Visualización |
| **seaborn** | Latest | Visualización estadística |

Para la lista completa, ver `requirements.txt`

---

## Características Avanzadas

### Validación de Colores
- Mapa personalizado de 16+ colores con preview visual
- Selector interactivo con vista previa en tiempo real

### Categorización Automática
- Diferentes categorías (Sedán, SUV, Truck)
- Interfaz intuitiva y amigable

### Defaults Inteligentes
- Se utilizan valores promedio por modelo
- Fallback a valores globales si es necesario
- Asegura predicciones realistas incluso con entrada parcial

---

## Autores

- Angel Felix V - Colaborador
- Francisco Javier Sanchez Acosta - Colaborador
- Andrei Mendoza Sánchez - Colaborador

---

## Tecnologías Utilizadas

- Backend: Python 3.8+
- Frontend: Streamlit
- ML Framework: scikit-learn
- Análisis: pandas, NumPy
- Visualización: Matplotlib, Seaborn, Plotly

---

## Licencia

Este proyecto es de uso libre para fines educativos y comerciales.

---

## Deployment

La aplicación está disponible en:  
**[Car Price Prediction App](https://regresioncarrosangel-dwdr7sjni26n6rnf332k5v.streamlit.app/)**

Desplegado en Streamlit Cloud para acceso gratuito y en tiempo real.


## Agradecimientos

- Dataset de Kaggle
- Comunidad de scikit-learn
- Streamlit por la excelente framework

---

**Última actualización**: Abril 2026  
**Estado**: Activo y en mantenimiento
