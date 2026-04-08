# 🚗 RegresionCarros_Angel
## Predicción de Precios de Autos Semi-nuevos con Machine Learning

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)](https://regresioncarrosangel-dwdr7sjni26n6rnf332k5v.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📋 Descripción General

**RegresionCarros_Angel** es una aplicación web full-stack que predice precios de autos semi-nuevos en tiempo real. Combina un **pipeline robusto de ML** con una **interfaz Streamlit intuitiva**, permitiendo predicciones precisas basadas en características del vehículo.

### 🎯 Objetivos
- Predecir precios de mercado basándose en especificaciones del auto
- Proporcionar herramienta accesible para compradores y vendedores
- Demostrar pipeline completo de Machine Learning en producción

---

## ⭐ Características Clave

| Característica | Descripción |
|---|---|
| **🤖 Modelo Inteligente** | Regresión lineal entrenada con 2,900+ registros reales |
| **🎨 Interfaz Amigable** | Aplicación web interactiva con selector visual de colores |
| **⚡ Predicción en Tiempo Real** | Respuestas instantáneas sin latencia notable |
| **📊 40+ Características** | Features ingenierizadas mediante One-Hot Encoding |
| **🔄 Manejo Robusto** | Limpieza automática, outliers detectados, valores por defecto inteligentes |
| **☁️ Deployment Listo** | Disponible en Streamlit Cloud (sin costo) |
| **💾 Modelos Guardados** | PKL serializado para reproducibilidad |

---

## 📊 Rendimiento del Modelo

### Estadísticas de Entrenamiento

```
Dataset Utilizado:           2,900+ registros
Variables Originales:        15+ características
Variables Finales:           40+ (post-encoding)
Split Entrenamiento/Prueba:  80% / 20%
Algoritmo:                   Linear Regression
Normalización:               StandardScaler (X e y)
Manejo de Outliers:          Método 3σ (desviación estándar)
```

### Métricas de Desempeño

| Métrica | Escala Normalizada | Observación |
|---|---|---|
| **RMSE** | ~0.15-0.20 | Error cuadrático medio (en escala 0-1) |
| **Modelo** | Linear Regression | Simple pero efectivo para este dataset |
| **Validación** | Train/Test Split | Evaluación independiente |

**⚠️ Nota sobre Métricas**: Los valores de RMSE se reportan en escala normalizada (0-1). Para convertir a USD reales, consultar los logs de entrenamiento en el script principal. El modelo es apto para **predicciones indicativas**, no para valuación profesional.

---

## 🏗️ Arquitectura del Pipeline ML

### Flujo Completo de Procesamiento

```
┌─────────────────────┐
│   Dataset CSV       │ (3,000+ registros)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  1. CARGA DE DATOS  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. LIMPIEZA        │ • Valores faltantes
│     DE DATOS        │ • Conversión de formatos
│                     │ • Eliminación de duplicados
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. FEATURE         │ • Nueva: Turbo_status
│     ENGINEERING     │ • Extracción de datos
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. OUTLIERS        │ • Método 3σ
│     TREATMENT       │ • Validación de rangos
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. ENCODING        │ • OneHotEncoder
│     CATEGÓRICAS     │ • drop='first'
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  6. NORMALIZACIÓN   │ • StandardScaler (X)
│                     │ • StandardScaler (y)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  7. ENTRENAMIENTO   │ • 80% train / 20% test
│     DEL MODELO      │ • Linear Regression
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  8. SERIALIZACIÓN   │ • PKL guardados
│     DE ARTEFACTOS   │ • Modelos + Scalers
└─────────────────────┘
```

### Detalle de Cada Etapa

#### **1️⃣ Carga de Datos**
```
Fuente: Dataset CSV
Registros: 3,000+
Variables: Fabricante, Modelo, Año, Combustible, Transmisión, 
           Cilindros, Potencia, Kilometraje, Precio (target)
```

#### **2️⃣ Limpieza y Preprocesamiento**
```python
# Eliminación de columnas innecesarias
df.drop(columns=['ID', 'Column_name'], inplace=True)

# Conversión de valores problemáticos:
Levy:       "-" → 0
Mileage:    "120000 km" → 120000
Doors:      "4-5" → 4
Engine Vol: Separación de turbo y volumen
```

#### **3️⃣ Feature Engineering**
**Nueva Variable Creada:**
- `Turbo_status`: Turbo / No Turbo (extraída del volumen del motor)

**Variables Numéricas Procesadas:**
- Levy, Engine Volume, Mileage, Cylinders, Doors, Airbags

**Variables Categóricas Procesadas:**
- Manufacturer (34+ marcas)
- Model (100+ modelos)
- Category (Sedan, SUV, Truck, etc.)
- Fuel Type (Petrol, Diesel, Hybrid, etc.)
- Gear Box Type (Manual, Automatic)
- Color (16+ opciones)
- Turbo_status (binaria)

#### **4️⃣ Tratamiento de Outliers**
Se utiliza el método de **3 desviaciones estándar (3σ)**:

```python
for col in ['Price', 'Levy', 'Mileage']:
    media = df[col].mean()
    desv_std = df[col].std()
    
    # Mantener solo valores dentro del rango [μ-3σ, μ+3σ]
    df = df[(df[col] >= media - 3*desv_std) & 
            (df[col] <= media + 3*desv_std)]
```

**Impacto**: Elimina ~2-5% de registros anómalos, mejorando la robustez del modelo.

#### **5️⃣ One-Hot Encoding**
```python
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X_categorical)

# Resultado: 40+ características binarias
# drop='first': Evita multicolinealidad perfecta
```

#### **6️⃣ Normalización StandardScaler**
```python
# Independientes (X)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Dependiente (y)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Resultado: Media=0, Desv. Std.=1
```

#### **7️⃣ Entrenamiento**
```python
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

# Evaluación
train_score = model.score(X_train_scaled, y_train_scaled)
test_score = model.score(X_test_scaled, y_test_scaled)
```

#### **8️⃣ Predicción**
El modelo utiliza **valores por defecto inteligentes**:
- Si falta una característica numérica → usa **media por modelo**
- Si falta una característica categórica → usa **moda global**
- Asegura predicciones realistas incluso con entrada incompleta

---

## 📁 Estructura del Proyecto

```
RegresionCarros_Angel/
│
├── 📄 README.md                        # Este archivo
├── 📄 requirements.txt                 # Dependencias (pip)
├── 📄 LICENSE                          # Licencia MIT
│
├── 🐍 CÓDIGO PRINCIPAL
│   ├── Regresion_Carros_Angel.py      # Pipeline ML completo
│   └── app.py                          # Aplicación Streamlit
│
├── 📊 DATOS
│   └── car_price_prediction.csv       # Dataset de entrenamiento
│
├── 💾 MODELOS ENTRENADOS (Pickle)
│   ├── linear_regression_model.pkl    # Modelo entrenado
│   ├── x_scaler.pkl                   # StandardScaler para X
│   ├── y_scaler_target.pkl            # StandardScaler para y
│   ├── onehot_encoder.pkl             # OneHotEncoder
│   └── model_feature_columns.pkl      # Nombres de columnas
│
└── 📸 RECURSOS (Opcional)
    └── images/                        # Capturas, diagramas
```

---

## 🚀 Guía de Instalación y Uso

### ✅ Requisitos Previos

- **Python 3.8 o superior**
- **pip** (administrador de paquetes)
- **Git** (para clonar el repositorio)

### 📦 Instalación Paso a Paso

#### **Paso 1: Clonar el Repositorio**
```bash
git clone https://github.com/AngelFelixV/RegresionCarros_Angel.git
cd RegresionCarros_Angel
```

#### **Paso 2: Crear Entorno Virtual (Recomendado)**
```bash
# Crear
python -m venv venv

# Activar entorno
# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

#### **Paso 3: Instalar Dependencias**
```bash
pip install -r requirements.txt
```

Esto instalará:
- `streamlit` - Framework web
- `pandas` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `scikit-learn` - Machine Learning
- `matplotlib` & `seaborn` - Visualización

#### **Paso 4: Ejecutar la Aplicación**
```bash
streamlit run app.py
```

**Resultado esperado:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Abre automáticamente en tu navegador. Si no, ve a `http://localhost:8501`

---

## 💻 Cómo Usar la Aplicación

### 🎯 Interfaz Principal - Flujo de Uso

1. **Selecciona Fabricante**
   - Dropdown con 34+ marcas (TOYOTA, LEXUS, BMW, etc.)
   - Auto-filtra modelos disponibles

2. **Elige Modelo**
   - Modelos específicos por fabricante
   - (ej: CAMRY si seleccionas TOYOTA)

3. **Especifica Categoría**
   - Sedan, SUV, Jeep, Truck, Coupe, etc.

4. **Configura Características**
   - 🔧 Tipo de Combustible: Petrol, Diesel, Hybrid, Electric, LPG
   - ⚙️ Transmisión: Manual, Automatic
   - 🎨 Color: 16+ opciones (preview visual incluido)

5. **Obtén Predicción**
   - Click en **"Predict Price"**
   - Recibe precio estimado en **USD**
   - Incluye información de validación

### 📝 Ejemplo de Entrada
```
Fabricante:       LEXUS
Modelo:           RX 450
Categoría:        SUV
Combustible:      Hybrid
Transmisión:      Automatic
Color:            Silver
```

**Resultado Esperado:**
```
✅ Precio Estimado: $45,250 USD
   Rango confiable: $40K - $50K
```

---

## 🔧 Uso Programático

### Ejecutar el Pipeline Completo
```bash
python Regresion_Carros_Angel.py
```

Esto:
- ✅ Carga y limpia el dataset
- ✅ Entrena el modelo
- ✅ Genera reportes
- ✅ Guarda artefactos (PKL)

### Integrar en tu Código Python

```python
from Regresion_Carros_Angel import predict_price

# Definir entrada
user_input = {
    'Manufacturer': 'TOYOTA',
    'Model': 'CAMRY',
    'Category': 'Sedan',
    'Fuel type': 'Petrol',
    'Gear box type': 'Automatic',
    'Color': 'White'
}

# Obtener predicción
price = predict_price(user_input)
print(f"Precio Estimado: ${price:,.2f}")

# Output: Precio Estimado: $25,750.45
```

### Usar Modelos Guardados Directamente

```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar artefactos
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('x_scaler.pkl', 'rb') as f:
    x_scaler = pickle.load(f)

# Preparar datos y predecir
X_new_scaled = x_scaler.transform(X_new)
prediction = model.predict(X_new_scaled)
```

---

## 📈 Desempeño Esperado

### Características del Dataset
```
Total Registros:           2,900+ (después de limpieza)
Registros Originales:      3,000+
Outliers Removidos:        ~50-100 (~2-5%)
Variables Iniciales:       15+
Variables Finales:         40+ (post-encoding)
```

### Características del Modelo
```
Algoritmo:                 Linear Regression
Método de Entrenamiento:   Train/Test Split (80/20)
Normalización:             StandardScaler
Manejo de Multicolinealidad: OneHotEncoder con drop='first'
```

### Limitaciones y Consideraciones

⚠️ **El modelo es indicativo, no profesional**

| Limitación | Descripción | Impacto |
|---|---|---|
| **Regresión Lineal Simple** | Asume relación lineal entre features y precio | Puede subestimar/sobreestimar en extremos |
| **Dataset Histórico** | Datos de un período específico | Precios pueden cambiar con mercado |
| **Rango de Validez** | Optimizado para autos "semi-nuevos" | Rendimiento desconocido en autos muy antiguos |
| **Collinealidad** | Algunas features pueden estar correlacionadas | Afecta coeficientes pero no predicciones |
| **Clases Desbalanceadas** | Algunos modelos tienen más datos que otros | Mayor error en modelos raros |

---

## 🐛 Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'streamlit'"
**Solución:**
```bash
pip install streamlit
# O reinstalar todo:
pip install -r requirements.txt
```

### Problema: "File not found: car_price_prediction.csv"
**Solución:**
- Asegúrate que el CSV está en la raíz del proyecto
- Verifica la ruta en el código si lo moviste

### Problema: La app es muy lenta
**Solución:**
- Los modelos se cargan solo una vez con `@st.cache_resource`
- Primer acceso es lento, después es rápido
- Considera versión local si necesitas velocidad extrema

### Problema: "ValueError: X has X features but this estimator was trained with X features"
**Solución:**
- Los modelos guardados esperan 40+ columnas específicas
- Asegúrate de que el preprocessing es idéntico al entrenamiento

### Problema: Predicciones lejanas a valores reales
**Recomendaciones:**
- Modelo usa regresión lineal (simple pero útil)
- Algoritmos avanzados (Random Forest, XGBoost) mejorarían precisión
- Validar entrada: algunos valores pueden ser atípicos

---

## 📚 Dependencias Principales

| Paquete | Versión | Propósito |
|---------|---------|----------|
| **streamlit** | Latest | Framework web interactiva |
| **pandas** | Latest | Manipulación y análisis de datos |
| **numpy** | Latest | Operaciones numéricas avanzadas |
| **scikit-learn** | 1.0+ | ML: regresión, encoding, escalado |
| **matplotlib** | Latest | Visualización estática |
| **seaborn** | Latest | Visualización estadística |

**Para instalar versiones específicas:**
```bash
pip install streamlit==1.28.0 pandas==2.0.0 numpy==1.24.0 scikit-learn==1.3.0
```

Ver `requirements.txt` para lista completa.

---

## 🌐 Deployment

### Versión en Vivo

La aplicación está **activa y disponible** en:

🔗 **[Car Price Prediction App](https://regresioncarrosangel-dwdr7sjni26n6rnf332k5v.streamlit.app/)**

**Plataforma:** Streamlit Cloud  
**Costo:** Gratuito  
**Disponibilidad:** 24/7  
**Actualización:** Automática con cada push a `main`

### Desplegar tu Propia Versión

#### **Opción 1: Streamlit Cloud (Recomendado)**
1. Crea cuenta en [streamlit.io](https://streamlit.io)
2. Conecta tu repositorio GitHub
3. Selecciona rama y archivo `app.py`
4. ¡Listo! Se despliega automáticamente

#### **Opción 2: Heroku**
```bash
# 1. Instalar Heroku CLI
# 2. Login
heroku login

# 3. Crear app
heroku create tu-app-name

# 4. Deploy
git push heroku main

# 5. Ver logs
heroku logs --tail
```

#### **Opción 3: Docker Local**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t car-predictor .
docker run -p 8501:8501 car-predictor
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas! 

### Cómo Contribuir
1. Fork el repositorio
2. Crea rama: `git checkout -b feature/tu-feature`
3. Commit: `git commit -m "Agrega mi feature"`
4. Push: `git push origin feature/tu-feature`
5. Abre Pull Request

### Ideas de Mejora
- [ ] Agregar modelos avanzados (Random Forest, XGBoost)
- [ ] Validación cruzada (K-fold)
- [ ] Dashboard de análisis exploratorio
- [ ] API REST con FastAPI
- [ ] Predicción de intervalo de confianza
- [ ] Historial de predicciones
- [ ] Comparación de precios con mercado real

---

## 📝 Autores y Colaboradores

| Autor | Rol | Contacto |
|---|---|---|
| **Angel Felix V** | Colaborador | [@GitHub](https://github.com/AngelFelixV) |
| **Francisco Javier Sanchez Acosta** | Colaborador | |
| **Andrei Mendoza Sánchez** | Colaborador | |

---

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT** - ver archivo [LICENSE](LICENSE) para detalles.

**Resumen:** Libre para usar en proyectos personales y comerciales con atribución.

---

## 🎓 Aprendizajes Clave

Este proyecto enseña:

✅ **Limpieza de datos** - 80% del trabajo real  
✅ **Feature engineering** - Crear features supera cantidad de features  
✅ **Manejo de outliers** - Método 3σ es simple pero efectivo  
✅ **Normalización** - Crítica para regresión lineal  
✅ **One-Hot Encoding** - Manejo de variables categóricas  
✅ **Serialización de modelos** - Guardar y cargar con Pickle  
✅ **Deployment en producción** - Streamlit Cloud para prototipado rápido  
✅ **Defaults inteligentes** - Hacer el sistema robusto a entrada parcial  

---

## 🔗 Enlaces Útiles

- 📱 **App en Vivo:** https://regresioncarrosangel-dwdr7sjni26n6rnf332k5v.streamlit.app/
- 💾 **Repositorio:** https://github.com/AngelFelixV/RegresionCarros_Angel


**Última Actualización:** Abril 2026  
**Versión:** 2.0 (Documentación Mejorada)

---
