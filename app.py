# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Guard XGBoost import
try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    st.sidebar.warning("⚠️ XGBoost not installed; skipping XGBoost option.")

st.set_page_config(page_title="🎯 Tuwaiq Admissions Dashboard", layout="wide")

# -- Data loading ------------------------------------------------------
@st.cache_data
def load_raw():
    return pd.read_csv("train.csv")

@st.cache_data
def load_cleaned():
    return pd.read_pickle("cleaned_train.pickle")

@st.cache_resource
def load_clf():
    # This should be *only* your pre-trained classifier
    return joblib.load("model.pkl")

raw_df     = load_raw()
cleaned_df = load_cleaned()

# -- Sidebar navigation -----------------------------------------------
st.sidebar.title("Controls")
page = st.sidebar.radio("View", ["EDA", "Cleaning", "Modeling", "Predict"])

# -- EDA ----------------------------------------------------------------
if page == "EDA":
    st.header("🔍 Exploratory Data Analysis")
    st.subheader("Missing Values per Column")
    st.table(raw_df.isna().sum())
    st.subheader("Raw Data Sample")
    st.dataframe(raw_df.head(20))
    st.subheader("Summary Statistics")
    st.dataframe(raw_df.describe(include="all").T)

# -- Cleaning -----------------------------------------------------------
elif page == "Cleaning":
    st.header("🧹 Data Cleaning Steps")
    df_disp = cleaned_df.drop(columns=[c for c in cleaned_df.columns if c.startswith("Home Region")])
    df_disp['Age'].fillna(df_disp['Age'].median(), inplace=True)
    st.subheader("Age Distribution (after median fill)")
    st.write(df_disp['Age'].value_counts())
    st.subheader("Cleaned Data Sample")
    st.dataframe(df_disp.head(20))
    st.subheader("Remaining Missing Values")
    st.table(df_disp.isna().sum())

# -- Modeling -----------------------------------------------------------
elif page == "Modeling":
    st.header("🤖 Model Training & Evaluation")
    X = cleaned_df.drop("Y", axis=1)
    y = cleaned_df["Y"]

    # Identify categorical columns and skill ordering
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if "Program Skill Level" in cat_cols:
        cat_cols.remove("Program Skill Level")
    skill_order = [["متقدم", "متوسط", "مبتدئ", "مفقود"]]

    # Build preprocessor for sklearn-based models
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("ord", OrdinalEncoder(categories=skill_order), ["Program Skill Level"]),
    ], remainder="passthrough")

    # Model selection
    options = ["Logistic Regression", "Decision Tree", "Random Forest + SMOTE"]
    if has_xgb:
        options.append("XGBoost")
    choice   = st.selectbox("Choose Model", options)
    test_pct = st.slider("Test Set Size (%)", 10, 50, 20, step=5)

    if st.button("Train & Evaluate"):
        # Split raw X,y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_pct/100, random_state=42, stratify=y
        )

        # SKLearn family
        if choice in options[:3]:
            # Fit preprocessor on entire X, then split
            Xp = preproc.fit_transform(X)
            Xt, Xv, yt, yv = train_test_split(
                Xp, y, test_size=test_pct/100, random_state=42, stratify=y
            )
            if choice == "Logistic Regression":
                clf = LogisticRegression(max_iter=1000)
            elif choice == "Decision Tree":
                clf = DecisionTreeClassifier(max_depth=5, random_state=42)
            else:
                clf = ImbPipeline([
                    ("smote", SMOTE(random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
                ])
            clf.fit(Xt, yt)
            preds = clf.predict(Xv)
            try:
                probs = clf.predict_proba(Xv)[:, 1]
            except AttributeError:
                scores = clf.decision_function(Xv)
                probs = (scores - scores.min()) / (scores.max() - scores.min())

        # XGBoost pipeline
        else:
            from sklearn.pipeline import Pipeline
            from sklearn.impute   import SimpleImputer
            from sklearn.preprocessing import StandardScaler

            num_cols    = X.select_dtypes(include=["number"]).columns.tolist()
            cat_cols_x  = X.select_dtypes(include=["object"]).columns.tolist()
            num_pipe    = Pipeline([("imp", SimpleImputer()), ("scale", StandardScaler())])
            cat_pipe    = Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ])
            proc_xgb    = ColumnTransformer([
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols_x),
            ])
            scale_w     = y_train.value_counts()[0] / y_train.value_counts()[1]
            xgb_clf     = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=scale_w,
                random_state=42
            )
            pipeline_xgb = Pipeline([("prep", proc_xgb), ("clf", xgb_clf)])
            pipeline_xgb.fit(X_train, y_train)
            probs = pipeline_xgb.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

        # Display metrics
        st.subheader("Accuracy")
        st.write(f"{accuracy_score(y_test, preds):.3f}")
        st.subheader("Classification Report")
        st.text(classification_report(y_test, preds))
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        st.pyplot(fig)
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0, 1], [0, 1], "--")
        ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate'); ax2.legend()
        st.pyplot(fig2)

# -- Predict ------------------------------------------------------------
elif page == "Predict":
    st.header("🔮 Predict a Single Applicant")

    # Rebuild the preprocessor exactly as in Modeling
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

    # Model selector in Predict
    model_options = ["Saved Model"]
    if has_xgb:
        model_options.append("XGBoost")
    model_choice = st.selectbox("Choose model", model_options)

    # Dynamic manual input form
    user_input = {}
    with st.form("single_app_form"):
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                mn, mx, md = float(X[col].min()), float(X[col].max()), float(X[col].median())
                user_input[col] = st.number_input(f"{col}", min_value=mn, max_value=mx, value=md, key=col)
            else:
                opts = X[col].dropna().unique().tolist()
                user_input[col] = st.selectbox(f"{col}", opts, key=col)
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([user_input])
        Xp  = preproc.transform(row)

        if model_choice == "Saved Model":
            clf = load_clf()
            prob = clf.predict_proba(Xp)[0, 1]
        else:
            # Rebuild & fit XGBoost on full training data
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler

            num_cols    = X.select_dtypes(include=["number"]).columns.tolist()
            cat_cols_x  = X.select_dtypes(include=["object"]).columns.tolist()
            num_pipe    = Pipeline([("imp", SimpleImputer()), ("scale", StandardScaler())])
            cat_pipe    = Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ])
            proc_xgb    = ColumnTransformer([
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols_x),
            ])
            scale_w     = cleaned_df["Y"].value_counts()[0] / cleaned_df["Y"].value_counts()[1]
            xgb_clf     = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=scale_w,
                random_state=42
            )
            pipeline_xgb = Pipeline([("prep", proc_xgb), ("clf", xgb_clf)])
            pipeline_xgb.fit(cleaned_df.drop("Y", axis=1), cleaned_df["Y"])
            prob = pipeline_xgb.predict_proba(row)[0, 1]

        st.subheader("🎲 Completion Probability")
        st.write(f"{prob:.1%}")
