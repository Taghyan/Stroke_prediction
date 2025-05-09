
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Stroke Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("🩺 Stroke Prediction Data Exploration")

# Sidebar - Data upload or example
# st.sidebar.header("1. Upload or Load Data")
# uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
# if uploaded_file is not None:
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
# else:
#     st.sidebar.info("Using example healthcare stroke dataset")
#     @st.cache_data
#     def load_data():
#         return pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/healthcare-dataset-stroke-data.csv')
#     df = load_data()

# Data cleaning
df.drop(columns=['id'], inplace=True)
df.dropna(subset=['bmi'], inplace=True)
df.drop(df[df['gender'] == 'Other'].index, inplace=True)

# Sidebar - Filters
st.sidebar.header("2. Filter Data")
# Numerical filters
age_range = st.sidebar.slider("Age range", float(df.age.min()), float(df.age.max()), (float(df.age.min()), float(df.age.max())))
glucose_range = st.sidebar.slider("Avg Glucose Level range", float(df.avg_glucose_level.min()), float(df.avg_glucose_level.max()), (float(df.avg_glucose_level.min()), float(df.avg_glucose_level.max())))
bmi_min, bmi_max = float(df.bmi.min()), float(df.bmi.max())
bmi_range = st.sidebar.slider("BMI range", bmi_min, bmi_max, (bmi_min, bmi_max))

# Categorical filters
gender = st.sidebar.multiselect("Gender", options=df.gender.unique(), default=list(df.gender.unique()))
hypertension = st.sidebar.selectbox("Hypertension", options=["All", 0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", options=["All", 0, 1])
ever_married = st.sidebar.multiselect("Ever Married", options=df.ever_married.unique(), default=list(df.ever_married.unique()))
work_type = st.sidebar.multiselect("Work Type", options=df.work_type.unique(), default=list(df.work_type.unique()))
residence_type = st.sidebar.multiselect("Residence Type", options=df.Residence_type.unique(), default=list(df.Residence_type.unique()))
smoking_status = st.sidebar.multiselect("Smoking Status", options=df.smoking_status.unique(), default=list(df.smoking_status.unique()))

# Apply filters
df_filtered = df[
    (df.age.between(*age_range)) &
    (df.avg_glucose_level.between(*glucose_range)) &
    (df.bmi.between(*bmi_range)) &
    (df.gender.isin(gender)) &
    (df.ever_married.isin(ever_married)) &
    (df.work_type.isin(work_type)) &
    (df.Residence_type.isin(residence_type)) &
    (df.smoking_status.isin(smoking_status))
]
if hypertension != "All":
    df_filtered = df_filtered[df_filtered.hypertension == hypertension]
if heart_disease != "All":
    df_filtered = df_filtered[df_filtered.heart_disease == heart_disease]

# Tabs for content
tab1, tab2 = st.tabs(["📋 Data & Stats", "📊 Charts"])

with tab1:
    st.subheader("Filtered Data Preview")
    st.dataframe(df_filtered, use_container_width=True)

    st.subheader("Statistical Summary")
    st.write(df_filtered.describe(exclude='O'))

    st.subheader("Categorical Summary")
    st.write(df_filtered.describe(include='O'))

    st.subheader("Correlation Matrix")
    corr = df_filtered.select_dtypes(exclude='O').corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.subheader("Distribution of Age")
    fig1 = px.histogram(df_filtered, x='age', nbins=20, title='Age Distribution')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Stroke Count by Gender")
    counts = df_filtered.groupby('gender')['stroke'].sum().reset_index()
    fig2 = px.bar(counts, x='gender', y='stroke', title='Stroke Count by Gender', labels={'stroke':'Number of Strokes'})
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("BMI vs. Glucose Level")
    fig3 = px.scatter(df_filtered, x='bmi', y='avg_glucose_level', title='BMI vs. Avg Glucose Level', labels={'bmi':'BMI', 'avg_glucose_level':'Avg Glucose Level'})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Smoking Status vs. Stroke Rate")
    smoke_rate = df_filtered.groupby('smoking_status')['stroke'].mean().reset_index()
    fig4 = px.bar(smoke_rate, x='smoking_status', y='stroke', title='Smoking Status vs. Stroke Rate', labels={'stroke':'Stroke Rate'})
    st.plotly_chart(fig4, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Devolped with ❤️ by Amr Taghyan")
