# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------------------
# Load dataset & model
# ---------------------------
df = pd.read_excel("HM-Sales-2018.xlsx")

model = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")
le_cat = joblib.load("labelencoder_category.pkl")
le_sub = joblib.load("labelencoder_subcategory.pkl")

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("📌 Filters & Prediction")

# Date filter (if available)
if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors='coerce')
    min_date = df["Order Date"].min()
    max_date = df["Order Date"].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    if len(date_range) == 2:
        df = df[(df["Order Date"] >= pd.to_datetime(date_range[0])) &
                (df["Order Date"] <= pd.to_datetime(date_range[1]))]

# Category filter
category_filter = st.sidebar.multiselect("Category", options=df["Category"].unique(), default=df["Category"].unique())
df = df[df["Category"].isin(category_filter)]

# Risk filter
df["Risk"] = df.apply(lambda x: 1 if (x["Discount"] > 0.5 or x["Profit"] < 0) else 0, axis=1)
risk_filter = st.sidebar.multiselect("Risk Level", options=[0, 1], default=[0, 1])
df = df[df["Risk"].isin(risk_filter)]

# ---------------------------
# Prediction Form
# ---------------------------
st.sidebar.subheader("🔮 Predict Product Risk")
category = st.sidebar.selectbox("Category", le_cat.classes_)
subcategory = st.sidebar.selectbox("Sub-Category", le_sub.classes_)
sales = st.sidebar.number_input("Sales", min_value=0.0)
quantity = st.sidebar.number_input("Quantity", min_value=1)
discount = st.sidebar.slider("Discount", 0.0, 1.0, 0.0)
profit = st.sidebar.number_input("Profit", min_value=-1000.0, max_value=1000.0)

if st.sidebar.button("Predict Risk"):
    cat_encoded = le_cat.transform([category])[0]
    sub_encoded = le_sub.transform([subcategory])[0]
    features = [[cat_encoded, sub_encoded, sales, quantity, discount, profit]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    st.sidebar.success("🚨 High Risk" if prediction == 1 else "✅ Low Risk")

# ---------------------------
# Dashboard Title
# ---------------------------
st.title("📊 H&M Fast Fashion Sustainability Dashboard")
st.markdown("### ML-powered risk detection & sales insights")

# ---------------------------
# KPIs
# ---------------------------
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
avg_discount = df["Discount"].mean()

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("💰 Total Sales", f"${total_sales:,.0f}")
kpi2.metric("📈 Total Profit", f"${total_profit:,.0f}")
kpi3.metric("🏷 Avg Discount", f"{avg_discount:.2%}")

# ---------------------------
# Charts
# ---------------------------
# Sales by Category
fig1 = px.bar(df.groupby("Category")["Sales"].sum().reset_index(),
              x="Category", y="Sales", title="Sales by Category",
              text_auto=True)
st.plotly_chart(fig1, use_container_width=True)

# Profit vs Discount
fig2 = px.scatter(df, x="Discount", y="Profit", size="Sales",
                  color="Category", hover_data=["Sub-Category"],
                  title="Profit vs Discount")
st.plotly_chart(fig2, use_container_width=True)

# Monthly Sales Trend
if "Order Date" in df.columns:
    monthly_sales = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly_sales["Order Date"] = monthly_sales["Order Date"].astype(str)
    fig3 = px.line(monthly_sales, x="Order Date", y="Sales", title="Monthly Sales Trend", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

# Risk Distribution
risk_counts = df["Risk"].value_counts().reset_index()
risk_counts.columns = ["Risk", "Count"]
fig4 = px.pie(risk_counts, names="Risk", values="Count", title="Risk Distribution", hole=0.4)
st.plotly_chart(fig4, use_container_width=True)

# ---------------------------
# Data Table
# ---------------------------
st.subheader("📄 Filtered Data Preview")
st.dataframe(df.head(20))
