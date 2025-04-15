import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the saved models
from sklearn.metrics import confusion_matrix

# Load the test data (assuming you have it or can recreate it)
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('titanic_train.csv')
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = df['Survived']
    X_cat = df[['Pclass', 'Sex', 'Embarked']]
    X_num = df[['Age', 'SibSp', 'Parch', 'Fare']]
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].mean())
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis=1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    from sklearn.preprocessing import StandardScaler
    try:
        scaler = joblib.load("scaler.save")
        X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
        X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])
    except FileNotFoundError:
        st.warning("Warning: 'scaler.save' not found. Using unscaled numerical features for evaluation.")
    return X_test, y_test

X_test, y_test = load_and_preprocess_data()

st.title('Evaluate Trained Titanic Models')

# Function to load a model
def load_model(model_name):
    try:
        loaded_model = joblib.load(model_name)
        st.success(f"Model '{model_name}' loaded successfully!")
        return loaded_model
    except FileNotFoundError:
        st.error(f"Error: '{model_name}' file not found. Make sure it's in the same directory.")
        return None

# Load the pre-trained models
rf_model = load_model("model") # Assuming your saved model is named "model"
svc_model = load_model("model") # If you saved different models, load them with their names
lr_model = load_model("model") # Adjust the filenames if necessary

models = {
    "Random Forest": rf_model,
    "SVC": svc_model,
    "Logistic Regression": lr_model,
}

selected_model_name = st.selectbox("Choose a model to evaluate", list(models.keys()))
selected_model = models[selected_model_name]

if selected_model:
    st.subheader(f"Evaluation of {selected_model_name}")

    # Display Accuracy
    accuracy = selected_model.score(X_test, y_test)
    st.write(f"**Accuracy:** {accuracy:.4f}")

    # Display Confusion Matrix
    y_pred = selected_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=['Real Neg.', 'Real Positive'],
                         columns=['Predicted Negative', 'Predicted Positive'])
    st.write("**Confusion Matrix:**")
    st.dataframe(cm_df)