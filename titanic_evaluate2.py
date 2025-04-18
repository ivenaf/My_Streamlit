import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For loading the saved model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#read dataframe
df=pd.read_csv('titanic_train.csv')

# initial page
st.title('Titanic: binary classification project')
st.sidebar.title('Table of contents')
pages=['Exploration', 'Data Visualization', 'Modelling']
page=st.sidebar.radio('Go to', pages)

###################################### Exploration ########################################
# write in the first page
if page==pages[0]:
    st.write('### Presentation of Data')
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())

#missing values checbox
    if st.checkbox('Show NA'):
        st.dataframe(df.isna().sum())

##################################### Data Viz ###########################################
# writing in the second page / plotting
if page==pages[1]:
    st.write('### Data Visualization')

    # plot distribution of 'Survived' variable
    fig=plt.figure(); sns.countplot(x='Survived', data=df, hue='Sex', palette='Set3'); st.pyplot(fig)
    fig=plt.figure(); sns.countplot(x='Sex', data=df, palette='Set3', hue='Sex'); plt.title('Distribution of the passengers gender'); st.pyplot(fig)
    fig = plt.figure(); sns.countplot(x = 'Pclass', data = df); plt.title("Distribution of the passengers class"); st.pyplot(fig)
    fig= sns.catplot(x='Pclass', y='Survived', data=df, kind='point', palette='Set3'); st.title('Distribution of the passengers class'); st.pyplot(fig)
    fig=sns.displot(x='Age', data=df, color='yellow', kde=True); plt.title('Distribution of the passengers age'); st.pyplot(fig)
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df); st.pyplot(fig)
    st.write('### Data types'); st.write(df.dtypes); df_num = df.select_dtypes(include=[np.number]); fig = plt.figure(); sns.heatmap(df_num.corr(), annot=True, cmap='viridis', center=True); st.pyplot(fig)

################################# MODELLING #############################################
if page== pages[2]:
    st.write('### Modelling')
    #drop irrelevant vars
    df_processed = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    #X/y split
    y=df_processed['Survived']
    X_cat=df_processed[['Pclass', 'Sex', 'Embarked']]
    X_num=df_processed[['Age', 'SibSp', 'Parch', 'Fare']]
    # replace missing values X_cat with mode
    for col in X_cat.columns:
        X_cat[col]=X_cat[col].fillna(X_cat[col].mode()[0])
    # replace missing values X_num with mean
    for col in X_num.columns:
        X_num[col]=X_num[col].fillna(X_num[col].mean())
    # encode categorical vars
    X_cat_scaled=pd.get_dummies(X_cat, columns=X_cat.columns)
    # concatenate
    X=pd.concat([X_cat_scaled, X_num], axis=1)
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # standardize numerical values
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    # Function to load a pre-trained model
    def load_pretrained_model(classifier_name):
        filename = "model"
        try:
            loaded_model = joblib.load(filename)
            st.success(f"Pre-trained model loaded successfully (as '{classifier_name}').")
            return loaded_model
        except FileNotFoundError:
            st.error(f"Error: Pre-trained model file 'model' not found in the same directory.")
            return None

    # accuracy or confusion matrix
    def scores (clf, choice):
        if clf is not None:
            if choice == 'Accuracy':
                return clf.score(X_test, y_test)
            elif choice == 'Confusion matrix':
                y_pred = clf.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, index=['Real Neg.', 'Real Positive'], columns=['Predicted Negative', 'Predicted Positive'])
                return cm_df
        return None

    # create a selectbox to choose the classifier
    choice= ['Random Forest', 'SVC', 'Logistic Regression']
    option=st.selectbox('Choice of the model', choice)
    st.write('You selected model:', option)

    # Load the pre-trained model
    loaded_clf = load_pretrained_model(option)

    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        if loaded_clf is not None:
            st.write(scores(loaded_clf, display))
    elif display == 'Confusion matrix':
        if loaded_clf is not None:
            st.dataframe(scores(loaded_clf, display))