import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data & model ─────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv("data/Customer-Churn-Records.csv")

@st.cache_resource
def load_assets():
    model        = joblib.load("churn_model.pkl")
    scaler       = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, feature_cols

df_raw           = load_raw()
model, scaler, feature_cols = load_assets()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
st.sidebar.title("Churn Dashboard")
st.sidebar.markdown("**Bank Customer Churn Prediction**")
st.sidebar.markdown("---")
tab = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 EDA", "🤖 Model Performance", "🔮 Live Prediction"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df_raw):,} customers")
st.sidebar.markdown(f"**Churn Rate:** {df_raw['Exited'].mean()*100:.1f}%")
st.sidebar.markdown(f"**Model:** Random Forest")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if tab == "🏠 Overview":
    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown("Analyzing **10,000 bank customers** to identify churn risk using Machine Learning.")
    st.markdown("---")

    # KPI metrics
    total      = len(df_raw)
    churned    = df_raw['Exited'].sum()
    retained   = total - churned
    churn_rate = df_raw['Exited'].mean() * 100
    avg_age    = df_raw['Age'].mean()
    avg_score  = df_raw['CreditScore'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",  f"{total:,}")
    c2.metric("Churned",          f"{churned:,}",  delta=f"{churn_rate:.1f}%", delta_color="inverse")
    c3.metric("Retained",         f"{retained:,}", delta=f"{100-churn_rate:.1f}%")
    c4.metric("Avg Age",          f"{avg_age:.1f} yrs")
    c5.metric("Avg Credit Score", f"{avg_score:.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Churn distribution pie
        churn_counts = df_raw['Exited'].value_counts().reset_index()
        churn_counts.columns = ['Status', 'Count']
        churn_counts['Status'] = churn_counts['Status'].map({0: 'Retained', 1: 'Churned'})
        fig = px.pie(
            churn_counts, names='Status', values='Count',
            title="Overall Churn Distribution",
            color='Status',
            color_discrete_map={'Retained': '#2ecc71', 'Churned': '#e74c3c'},
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Churn by location
        churn_loc = df_raw.groupby('Location')['Exited'].mean().reset_index()
        churn_loc['Churn Rate (%)'] = (churn_loc['Exited'] * 100).round(1)
        fig = px.bar(
            churn_loc, x='Location', y='Churn Rate (%)',
            title="Churn Rate by Location",
            color='Location',
            text='Churn Rate (%)',
            color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71']
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Churn by gender
        churn_gender = df_raw.groupby('Gender')['Exited'].mean().reset_index()
        churn_gender['Churn Rate (%)'] = (churn_gender['Exited'] * 100).round(1)
        fig = px.bar(
            churn_gender, x='Gender', y='Churn Rate (%)',
            title="Churn Rate by Gender",
            color='Gender', text='Churn Rate (%)',
            color_discrete_sequence=['#9b59b6', '#f39c12']
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Churn by number of products
        churn_prod = df_raw.groupby('NumOfProducts')['Exited'].mean().reset_index()
        churn_prod['Churn Rate (%)'] = (churn_prod['Exited'] * 100).round(1)
        fig = px.bar(
            churn_prod, x='NumOfProducts', y='Churn Rate (%)',
            title="Churn Rate by Number of Products",
            color='NumOfProducts', text='Churn Rate (%)',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif tab == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Deep dive into customer features and their relationship with churn.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df_raw, x='Age', color='Exited', barmode='overlay',
            title="Age Distribution by Churn",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'Exited': 'Churned'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df_raw, x='Exited', y='Account Balance',
            color='Exited', title="Account Balance vs Churn",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'Exited': 'Churned (0=No, 1=Yes)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.histogram(
            df_raw, x='CreditScore', color='Exited', barmode='overlay',
            title="Credit Score Distribution by Churn",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(
            df_raw, x='Exited', y='EstimatedSalary',
            color='Exited', title="Estimated Salary vs Churn",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Satisfaction score vs churn
    sat = df_raw.groupby('Satisfaction Score')['Exited'].mean().reset_index()
    sat['Churn Rate (%)'] = (sat['Exited'] * 100).round(1)
    fig = px.bar(
        sat, x='Satisfaction Score', y='Churn Rate (%)',
        title="Churn Rate by Satisfaction Score",
        color='Churn Rate (%)', text='Churn Rate (%)',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Account Balance',
                'NumOfProducts', 'EstimatedSalary', 'Satisfaction Score',
                'Point Earned', 'Complain', 'Exited']
    corr = df_raw[num_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r', aspect='auto',
        zmin=-1, zmax=1
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw data preview
    st.markdown("### Raw Data Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif tab == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("Evaluation metrics for the trained **Random Forest** model.")
    st.markdown("---")

    # Load test data
    X, y, _ = load_and_preprocess("data/Customer-Churn-Records.csv")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Metric cards
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    rauc = roc_auc_score_val = auc(*roc_curve(y_test, y_prob)[:2])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{acc*100:.2f}%")
    c2.metric("AUC-ROC",   f"{roc_auc_score_val:.4f}")
    c3.metric("F1 Score",  f"{f1:.4f}")
    c4.metric("Precision", f"{prec:.4f}")
    c5.metric("Recall",    f"{rec:.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm, text_auto=True,
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=['Retained', 'Churned'],
            y=['Retained', 'Churned'],
            color_continuous_scale='Blues'
        )
        fig.update_layout(width=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f"Random Forest (AUC = {roc_auc:.4f})",
            line=dict(color='royalblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash')
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("### Feature Importance")
    st.image("feature_importance.png", use_column_width=True)

    # Prediction probability distribution
    st.markdown("### Predicted Probability Distribution")
    prob_df = pd.DataFrame({'Probability': y_prob, 'Actual': y_test.values})
    fig = px.histogram(
        prob_df, x='Probability', color='Actual', barmode='overlay',
        title="Churn Probability Distribution",
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        nbins=50
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif tab == "🔮 Live Prediction":
    st.title("🔮 Live Churn Prediction")
    st.markdown("Enter customer details below to get an **instant churn probability**.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Info**")
        age         = st.slider("Age", 18, 92, 35)
        gender      = st.selectbox("Gender", ["Female", "Male"])
        location    = st.selectbox("Location", ["France", "Germany", "Spain"])
        card_type   = st.selectbox("Card Type", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])

    with col2:
        st.markdown("**Financial Info**")
        credit_score = st.slider("Credit Score", 300, 850, 650)
        balance      = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0, step=1000.0)
        salary       = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 75000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

    with col3:
        st.markdown("**Account Activity**")
        tenure       = st.slider("Tenure (years)", 0, 10, 5)
        has_cr_card  = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active    = st.selectbox("Is Active Member?", ["Yes", "No"])
        complain     = st.selectbox("Has Complaint?",   ["No", "Yes"])
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        points       = st.slider("Points Earned", 100, 1000, 400)

    st.markdown("---")

    # Encode inputs
    loc_map    = {"France": 0, "Germany": 1, "Spain": 2}
    gender_map = {"Female": 0, "Male": 1}
    card_map   = {"DIAMOND": 0, "GOLD": 1, "PLATINUM": 2, "SILVER": 3}
    yes_no     = {"Yes": 1, "No": 0}

    input_data = np.array([[
        credit_score,
        loc_map[location],
        gender_map[gender],
        age,
        tenure,
        balance,
        num_products,
        yes_no[has_cr_card],
        yes_no[is_active],
        salary,
        yes_no[complain],
        satisfaction,
        card_map[card_type],
        points
    ]])

    input_scaled = scaler.transform(input_data)

    if st.button("🔍 Predict Churn", type="primary", use_container_width=True):
        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred == 1:
                st.error(f"### ⚠️ High Churn Risk\nThis customer is likely to **leave**.")
            else:
                st.success(f"### ✅ Low Churn Risk\nThis customer is likely to **stay**.")

            # Risk level label
            if prob < 0.3:
                risk = "🟢 Low Risk"
            elif prob < 0.6:
                risk = "🟡 Medium Risk"
            else:
                risk = "🔴 High Risk"

            st.markdown(f"**Churn Probability:** `{prob*100:.1f}%`")
            st.markdown(f"**Risk Level:** {risk}")

        with res_col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                title={"text": "Churn Probability (%)"},
                delta={"reference": 20},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#e74c3c" if prob > 0.5 else "#2ecc71"},
                    "steps": [
                        {"range": [0, 30],  "color": "#d5f5e3"},
                        {"range": [30, 60], "color": "#fdebd0"},
                        {"range": [60, 100],"color": "#fadbd8"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Input summary table
        st.markdown("### Customer Summary")
        summary = {
            "Age": age, "Gender": gender, "Location": location,
            "Credit Score": credit_score, "Account Balance": f"${balance:,.0f}",
            "Salary": f"${salary:,.0f}", "Tenure": f"{tenure} yrs",
            "Products": num_products, "Active Member": is_active,
            "Has Complaint": complain, "Satisfaction": satisfaction,
            "Card Type": card_type, "Points": points
        }
        summary_df = pd.DataFrame(summary.items(), columns=["Feature", "Value"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)