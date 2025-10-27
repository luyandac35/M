# =============================================
# 🏥 National Health Insurance (NHI) ML Project
# Streamlit Web Application (Direct Load)
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import warnings, os
warnings.filterwarnings('ignore')

# ---------------------------------------------
# 1️⃣ APP CONFIGURATION
# ---------------------------------------------
st.set_page_config(page_title="NHI Healthcare ML Dashboard", layout="wide")
st.title("🏥 National Health Insurance (NHI) - Machine Learning & Forecasting")

st.markdown("""
This dashboard analyzes **South African healthcare data**  
to support **NHI implementation** through:
- Predictive modeling (Decision Tree, Random Forest)
- Data exploration & visualization
- Forecasting healthcare trends using Prophet
""")

# ---------------------------------------------
# 2️⃣ LOAD MERGED DATASET
# ---------------------------------------------
st.header("📂 Step 1: Load Cleaned & Merged Dataset")

dataset_path = "merged_cleaned_dataset.csv"

if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    st.success("✅ Successfully loaded dataset: 'merged_cleaned_dataset.csv'")
    st.write("### Dataset Preview", df.head())
else:
    st.error("❌ 'merged_cleaned_dataset.csv' not found. Please place it in the same folder as app.py.")
    st.stop()

# ---------------------------------------------
# 3️⃣ UNDERSTAND THE DATA
# ---------------------------------------------
st.header("🔍 Step 2: Understand the Data")

col1, col2 = st.columns(2)
with col1:
    st.write("**Data Types**")
    st.write(df.dtypes)
with col2:
    st.write("**Missing Values**")
    st.write(df.isnull().sum())

st.write("**Duplicates:**", df.duplicated().sum())

# ---------------------------------------------
# 4️⃣ DATA CLEANING
# ---------------------------------------------
st.header("🧹 Step 3: Data Cleaning & Preprocessing")

df.fillna(df.median(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

st.success("✅ Missing values handled & categorical variables encoded.")

# ---------------------------------------------
# 5️⃣ EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------
st.header("📊 Step 4: Exploratory Data Analysis (EDA)")

st.subheader("📈 Line Chart")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    selected_col = st.selectbox("Select numeric column for trend visualization:", numeric_cols)
    st.line_chart(df[selected_col])

st.subheader("🥧 Pie Chart")
pie_col = st.selectbox("Select column for Pie Chart:", df.columns)
pie_data = df[pie_col].value_counts().reset_index()
pie_data.columns = ['Category', 'Count']
fig_pie = px.pie(pie_data, names='Category', values='Count', title=f"Distribution of {pie_col}")
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("🔥 Heatmap")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ---------------------------------------------
# 6️⃣ FEATURE ENGINEERING
# ---------------------------------------------
st.header("⚙️ Step 5: Feature Engineering")

if all(col in df.columns for col in ['age', 'tobacco', 'ldl', 'adiposity', 'alcohol']):
    df['risk_score'] = (
        df['age'].astype(float) * 0.2 +
        df['tobacco'].astype(float) * 0.15 +
        df['ldl'].astype(float) * 0.25 +
        df['adiposity'].astype(float) * 0.2 +
        df['alcohol'].astype(float) * 0.2
    )
    st.success("✅ Risk score feature successfully added.")
else:
    st.warning("⚠️ Risk score could not be computed (missing columns).")

# ---------------------------------------------
# 7️⃣ TRAIN / TEST SPLIT
# ---------------------------------------------
st.header("🧪 Step 6: Train / Test Split")

target_col = st.selectbox("Select Target Variable (y):", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("Training Size:", X_train.shape)
st.write("Testing Size:", X_test.shape)

# ---------------------------------------------
# 8️⃣ MODEL TRAINING
# ---------------------------------------------
st.header("🤖 Step 7: Train Models")

dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42, n_estimators=100)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)

st.write("### 🌳 Decision Tree Accuracy:", round(acc_dt, 3))
st.write("### 🌲 Random Forest Accuracy:", round(acc_rf, 3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, cmap="Blues", ax=ax[0])
ax[0].set_title("Decision Tree Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap="Greens", ax=ax[1])
ax[1].set_title("Random Forest Confusion Matrix")
st.pyplot(fig)

# ---------------------------------------------
# 9️⃣ FORECASTING (Prophet)
# ---------------------------------------------
st.header("🔮 Step 8: Forecasting with Prophet")

if 'date' in df.columns and 'total' in df.columns:
    forecast_df = df[['date', 'total']].rename(columns={'date': 'ds', 'total': 'y'})
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.write("### Forecast Preview", forecast.tail())

    fig_forecast = px.line(forecast, x='ds', y='yhat', title='Healthcare Forecast (Prophet)')
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("⚠️ Prophet forecasting requires 'date' and 'total' columns.")

# ---------------------------------------------
# ✅ END
# ---------------------------------------------
st.success("✅ Analysis Completed — NHI Insights Generated Successfully!")
st.markdown("Developed by **Simba** 🧠 | Powered by Streamlit, Prophet & scikit-learn")
