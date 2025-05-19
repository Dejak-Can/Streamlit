# Optimized app.py (Stable Emoji Fix & Performance Tweaks)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    st.sidebar.warning("⚠️ XGBoost not installed; skipping XGBoost option.")

st.set_page_config(page_title="Tuwaiq Admissions Dashboard", layout="wide")

@st.cache_data

def load_cleaned():
    return pd.read_pickle("Cleaned_train.pickle")

@st.cache_resource

def load_clf():
    return joblib.load("model.pkl")

cleaned_df = load_cleaned()

st.sidebar.title("Controls")
page = st.sidebar.radio("View", ["EDA", "Cleaning", "Modeling", "Predict"])

if page == "EDA":
    st.header("Exploratory Data Analysis")
    st.subheader("Sample & Stats")
    st.dataframe(cleaned_df.head(10))
    st.dataframe(cleaned_df.describe(include="all").T)

elif page == "Cleaning":
    st.header("Data Cleaning Steps")
    df = cleaned_df.copy()
    df.drop(columns=[c for c in df.columns if c.startswith("Home Region")], inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    st.dataframe(df.head(10))
    st.table(df.isna().sum())

elif page == "Modeling":
    st.header("Model Training")
    X = cleaned_df.drop("Y", axis=1)
    y = cleaned_df["Y"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if "Program Skill Level" in cat_cols:
        cat_cols.remove("Program Skill Level")
    skill_order = [["متقدم", "متوسط", "مبتدئ", "مفقود"]]

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("ord", OrdinalEncoder(categories=skill_order), ["Program Skill Level"]),
    ], remainder="passthrough")

    choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "Random Forest + SMOTE"] + (["XGBoost"] if has_xgb else []))
    test_pct = st.slider("Test Size (%)", 10, 50, 20, step=5)

    if st.button("Train & Evaluate"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42, stratify=y)

        if choice == "XGBoost":
            proc = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer()), ("scale", StandardScaler())]), X.select_dtypes(include=["number"]).columns.tolist()),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
            ])
            clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1], random_state=42)
            pipe = Pipeline([("prep", proc), ("clf", clf)])
        else:
            pipe = ImbPipeline([
                ("prep", preproc),
                ("clf", LogisticRegression(max_iter=1000) if choice == "Logistic Regression" else
                        DecisionTreeClassifier(max_depth=5, random_state=42) if choice == "Decision Tree" else
                        RandomForestClassifier(n_estimators=200, random_state=42))
            ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else preds

        st.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
        st.text(classification_report(y_test, preds))
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, preds)
        ax.imshow(cm, cmap='Blues')
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                ax.text(j, i, cm[i, j], ha='center', va='center')
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

elif page == "Predict":
    st.header("Predict an Applicant")
    X = cleaned_df.drop("Y", axis=1)
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if "Program Skill Level" in cat_cols:
        cat_cols.remove("Program Skill Level")
    skill_order = [["متقدم", "متوسط", "مبتدئ", "مفقود"]]

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("ord", OrdinalEncoder(categories=skill_order), ["Program Skill Level"]),
    ], remainder="passthrough")
    preproc.fit(X)

    with st.form("predict_form"):
        user_input = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                user_input[col] = st.number_input(col, float(X[col].min()), float(X[col].max()), float(X[col].median()))
            else:
                user_input[col] = st.selectbox(col, sorted(X[col].dropna().unique()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])
        Xp = preproc.transform(input_df)
        clf = load_clf()
        prob = clf.predict_proba(Xp)[0, 1]
        st.metric("Predicted Probability", f"{prob:.2%}")
