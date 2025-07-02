import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib

st.set_page_config(
    page_title="MTN Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        df.columns = df.columns.str.strip()

        df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], format='%b-%y', errors='coerce')
        df.dropna(subset=['Date of Purchase'], inplace=True)
        
        df['Total Revenue'] = pd.to_numeric(df['Total Revenue'], errors='coerce')
        df['Total Revenue'].fillna(0, inplace=True)

        df['Customer Tenure in months'] = pd.to_numeric(df['Customer Tenure in months'], errors='coerce')
        df['Customer Tenure in months'].fillna(df['Customer Tenure in months'].median(), inplace=True)

        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Age'] = df['Age'].astype(int)

        df['Churn'] = df['Customer Churn Status'].map({'Yes': 1, 'No': 0})

        columns_to_drop_initial = ['Customer ID', 'Full Name', 'Customer Churn Status']
        df.drop(columns=columns_to_drop_initial, inplace=True, errors='ignore')
        
        return df
    except Exception as e:
        st.error(f"Error during data loading or initial cleaning: {e}")
        st.stop()

@st.cache_resource
def train_model(df_processed):
    
    df_ml = df_processed.copy()

    columns_to_drop_for_ml = ['Date of Purchase', 'Customer Review', 'Reasons for Churn']
    df_ml.drop(columns=columns_to_drop_for_ml, inplace=True, errors='ignore')

    categorical_cols = df_ml.select_dtypes(include='object').columns
    df_ml = pd.get_dummies(df_ml, columns=categorical_cols, drop_first=True)

    X = df_ml.drop('Churn', axis=1)
    y = df_ml['Churn']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # Evaluate Random Forest Model
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1-Score": f1_score(y_test, y_pred_rf),
        "ROC AUC": roc_auc_score(y_test, y_prob_rf),
        "Confusion Matrix": confusion_matrix(y_test, y_pred_rf),
        "Classification Report": classification_report(y_test, y_pred_rf, output_dict=True)
    }

    # Train Logistic Regression Model
    log_reg_model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', max_iter=1000)
    log_reg_model.fit(X_train, y_train)

    # Evaluate Logistic Regression Model
    y_pred_lr = log_reg_model.predict(X_test)
    y_prob_lr = log_reg_model.predict_proba(X_test)[:, 1]
    lr_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1-Score": f1_score(y_test, y_pred_lr),
        "ROC AUC": roc_auc_score(y_test, y_prob_lr),
        "Confusion Matrix": confusion_matrix(y_test, y_pred_lr),
        "Classification Report": classification_report(y_test, y_pred_lr, output_dict=True)
    }

    return rf_model, X.columns, rf_metrics, lr_metrics

file_path = 'mtn_customer_churn.csv'
if not os.path.exists(file_path):
    st.error(f"Error: '{file_path}' file not found. Please ensure the CSV file is in the same directory as the Streamlit app.")
    st.stop()

df = load_data(file_path)

model, feature_names, rf_metrics, lr_metrics = train_model(df.copy())

st.title("📊 MTN Customer Churn Prediction Dashboard")
st.markdown("""
Welcome to the **MTN Customer Churn Prediction Dashboard!**
This project aims to provide a comprehensive understanding of customer churn behavior for MTN Nigeria, identifying key factors that lead to customer attrition. Through this interactive dashboard, we address critical business questions such as:

* **Churn Trends & Patterns:** What are the overall churn rates, and how do they vary across different customer segments?
* **Demographic Impact:** How do factors like age and gender influence customer churn?
* **Service & Usage Analysis:** What is the relationship between customer tenure, data usage, total revenue, and the likelihood of churning?
* **Product & Plan Effectiveness:** Which MTN devices and subscription plans are associated with higher or lower churn rates?
* **Customer Satisfaction & Churn:** How does customer satisfaction correlate with churn, and what insights can be gained?

Use the tabs above to explore the data dynamically and uncover valuable insights into customer churn!
""")

tab1, tab2 = st.tabs(["Dashboard (EDA)", "Churn Prediction"])

with tab1:
    st.header("📈 Customer Churn Dashboard (Exploratory Data Analysis)")
    st.markdown("Here you can explore customer churn patterns and drivers.")

    st.subheader("1. Churn Distribution")
    churn_counts = df['Churn'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sns.countplot(x='Churn', data=df, hue='Churn', palette='pastel', legend=False, ax=ax1)
    for p in ax1.patches:
        ax1.text(p.get_x() + p.get_width() / 2, p.get_height(), p.get_height(), ha='center', va='bottom')
    ax1.set_title('Distribution of Customer Churn (0: No Churn, 1: Churn)')
    ax1.set_xlabel('Churn Status')
    ax1.set_ylabel('Number of Customers')
    ax1.set_xticks(ticks=[0, 1], labels=['No Churn', 'Churn'])
    st.pyplot(fig1)

    st.subheader("2. Churn Rate by Gender")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.countplot(x='Gender', hue='Churn', data=df, palette='viridis', ax=ax2)
    ax2.set_title('Churn Rate by Gender')
    ax2.set_xlabel('Gender')
    ax2.set_ylabel('Number of Customers')
    ax2.legend(title='Churn', labels=['No Churn', 'Churn'])
    st.pyplot(fig2)

    st.subheader("3. Churn Rate by Subscription Plan")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Subscription Plan', hue='Churn', data=df, palette='magma', ax=ax3)
    ax3.set_title('Churn Rate by Subscription Plan')
    ax3.set_xlabel('Subscription Plan')
    ax3.set_ylabel('Number of Customers')
    ax3.legend(title='Churn', labels=['No Churn', 'Churn'])
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig3)

    st.subheader("4. Churn Rate by Customer Tenure")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Customer Tenure in months', hue='Churn', multiple='stack', bins=30, palette='rocket', ax=ax4)
    ax4.set_title('Churn Rate by Customer Tenure (Months)')
    ax4.set_xlabel('Customer Tenure (Months)')
    ax4.set_ylabel('Number of Customers')
    ax4.legend(title='Churn', labels=['No Churn', 'Churn'])
    st.pyplot(fig4)

    st.subheader("5. Churn Rate by Total Revenue")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Total Revenue', hue='Churn', multiple='stack', bins=50, palette='viridis', ax=ax5)
    ax5.set_title('Churn Rate by Total Revenue')
    ax5.set_xlabel('Total Revenue')
    ax5.set_ylabel('Number of Customers')
    ax5.legend(title='Churn', labels=['No Churn', 'Churn'])
    st.pyplot(fig5)

    if 'Satisfaction Rate' in df.columns:
        st.subheader("6. Churn Rate by Satisfaction Rate")
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Satisfaction Rate', hue='Churn', data=df, palette='coolwarm', ax=ax6)
        ax6.set_title('Churn Rate by Satisfaction Rate')
        ax6.set_xlabel('Satisfaction Rate')
        ax6.set_ylabel('Number of Customers')
        ax6.legend(title='Churn', labels=['No Churn', 'Churn'])
        st.pyplot(fig6)
    else:
        st.info("'Satisfaction Rate' column not found for analysis.")

    st.subheader("7. Top Feature Importances for Churn Prediction")
    feature_importances = pd.Series(model.feature_importances_, index=feature_names)
    top_n = 15
    top_features = feature_importances.nlargest(top_n)
    
    fig_imp, ax_imp = plt.subplots(figsize=(12, 8))
    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis', hue=top_features.index, legend=False, ax=ax_imp)
    ax_imp.set_title(f'Top {top_n} Feature Importances for Churn Prediction')
    ax_imp.set_xlabel('Importance')
    ax_imp.set_ylabel('Feature')
    st.pyplot(fig_imp)

    # New Section: Model Evaluation Metrics
    st.header("📊 Model Evaluation Metrics")
    st.markdown("Here you can see the performance metrics for the trained models.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest Classifier Performance")
        st.write(f"**Accuracy:** {rf_metrics['Accuracy']:.4f}")
        st.write(f"**Precision:** {rf_metrics['Precision']:.4f}")
        st.write(f"**Recall:** {rf_metrics['Recall']:.4f}")
        st.write(f"**F1-Score:** {rf_metrics['F1-Score']:.4f}")
        st.write(f"**ROC AUC Score:** {rf_metrics['ROC AUC']:.4f}")
        st.markdown("**Confusion Matrix:**")
        st.code(rf_metrics['Confusion Matrix'])
        st.markdown("**Classification Report (Class 1 - Churn):**")
        st.dataframe(pd.DataFrame(rf_metrics['Classification Report']).transpose().iloc[1])

    with col2:
        st.subheader("Logistic Regression Performance")
        st.write(f"**Accuracy:** {lr_metrics['Accuracy']:.4f}")
        st.write(f"**Precision:** {lr_metrics['Precision']:.4f}")
        st.write(f"**Recall:** {lr_metrics['Recall']:.4f}")
        st.write(f"**F1-Score:** {lr_metrics['F1-Score']:.4f}")
        st.write(f"**ROC AUC Score:** {lr_metrics['ROC AUC']:.4f}")
        st.markdown("**Confusion Matrix:**")
        st.code(lr_metrics['Confusion Matrix'])
        st.markdown("**Classification Report (Class 1 - Churn):**")
        st.dataframe(pd.DataFrame(lr_metrics['Classification Report']).transpose().iloc[1])


with tab2:
    st.header("🔮 Predict Customer Churn")
    st.markdown("Enter customer details below to predict churn.")

    with st.form("churn_prediction_form"):
        st.subheader("Customer Details")
        
        age = st.number_input("Age", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].median()))
        
        customer_tenure = st.number_input("Customer Tenure in months", min_value=int(df['Customer Tenure in months'].min()), max_value=int(df['Customer Tenure in months'].max()), value=int(df['Customer Tenure in months'].median()))
        
        unit_price = st.number_input("Unit Price", min_value=float(df['Unit Price'].min()), max_value=float(df['Unit Price'].max()), value=float(df['Unit Price'].median()))
        
        num_times_purchased = st.number_input("Number of Times Purchased", min_value=int(df['Number of Times Purchased'].min()), max_value=int(df['Number of Times Purchased'].max()), value=int(df['Number of Times Purchased'].median()))
        
        total_revenue = st.number_input("Total Revenue", min_value=float(df['Total Revenue'].min()), max_value=float(df['Total Revenue'].max()), value=float(df['Total Revenue'].median()))
        
        data_usage = st.number_input("Data Usage", min_value=float(df['Data Usage'].min()), max_value=float(df['Data Usage'].max()), value=float(df['Data Usage'].median()))
        
        state = st.selectbox("State", options=df['State'].unique())
        mtn_device = st.selectbox("MTN Device", options=df['MTN Device'].unique())
        gender = st.radio("Gender", options=df['Gender'].unique())
        subscription_plan = st.selectbox("Subscription Plan", options=df['Subscription Plan'].unique())
        
        satisfaction_rate = None
        if 'Satisfaction Rate' in df.columns:
            satisfaction_rate = st.number_input("Satisfaction Rate", min_value=int(df['Satisfaction Rate'].min()), max_value=int(df['Satisfaction Rate'].max()), value=int(df['Satisfaction Rate'].median()))

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        processed_input = pd.DataFrame(0, index=[0], columns=feature_names)
        
        for col in feature_names:
            if col.startswith(('State_', 'MTN Device_', 'Gender_', 'Subscription Plan_')):
                processed_input[col] = False

        processed_input['Age'] = age
        processed_input['Customer Tenure in months'] = customer_tenure
        processed_input['Unit Price'] = unit_price
        processed_input['Number of Times Purchased'] = num_times_purchased
        processed_input['Total Revenue'] = total_revenue
        processed_input['Data Usage'] = data_usage
        if satisfaction_rate is not None:
            processed_input['Satisfaction Rate'] = satisfaction_rate
        else:
            processed_input['Satisfaction Rate'] = int(df['Satisfaction Rate'].median()) 

        state_col_name = f'State_{state}'
        if state_col_name in processed_input.columns:
            processed_input[state_col_name] = True

        mtn_device_col_name = f'MTN Device_{mtn_device}'
        if mtn_device_col_name in processed_input.columns:
            processed_input[mtn_device_col_name] = True

        gender_col_name = f'Gender_{gender}'
        if gender_col_name in processed_input.columns:
            processed_input[gender_col_name] = True

        subscription_plan_col_name = f'Subscription Plan_{subscription_plan}'
        if subscription_plan_col_name in processed_input.columns:
            processed_input[subscription_plan_col_name] = True
            
        input_for_prediction = processed_input[feature_names]

        prediction = model.predict(input_for_prediction)[0]
        prediction_proba = model.predict_proba(input_for_prediction)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"**Customer will likely CHURN!** (Probability: {prediction_proba[1]*100:.2f}%)")
            st.write("Take proactive steps to retain this customer.")
        else:
            st.success(f"**Customer will likely NOT CHURN.** (Probability: {prediction_proba[0]*100:.2f}%)")
            st.write("This customer appears stable.")