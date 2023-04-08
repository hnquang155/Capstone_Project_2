import streamlit as st

st.markdown("# Dashboard ❄️")
st.sidebar.markdown("# Dashboard ❄️")

option = st.selectbox(
    'Select model you want to view result',
    ('Landsat 8', 'Sentinel 2'))

st.write('You selected:', option)

##########################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pickle

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import gzip




def plot_confusion_matrix(true_value,predicted_value,title,labels):
    '''
    Plots a confusion matrix.
    Attributes:
    true_value - The ground truth value for comparision.
    predicted_value - The values predicted by the model.
    title - Title of the plot.
    labels - The x and y labels of the plot.
    '''
    cm = confusion_matrix(true_value,predicted_value)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues');
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title(title); 
    ax.xaxis.set_ticklabels(labels); 
    ax.yaxis.set_ticklabels(labels)

###########################################
if option == 'Landsat 8':
    crop_presence_data = pd.read_csv("pages\\train-landsat.csv")
    crop_presence_data.head()

    clean_df = crop_presence_data.dropna()
    clean_df['ndvi_diff'] = clean_df['filter_ndvi']-clean_df['original_ndvi']

    clean_df['time_frames']= pd.to_datetime(clean_df['time_frames'])


    v1 = clean_df[(clean_df['time_frames'] > '2021-11-01') & (clean_df['time_frames'] < '2021-12-10')]
    v2 = clean_df[(clean_df['time_frames'] > '2022-03-15') & (clean_df['time_frames'] < '2022-04-15')]
    v3 = clean_df[(clean_df['time_frames'] > '2022-06-01') & (clean_df['time_frames'] < '2022-06-30')]
    clean_df1 = pd.concat([v1,v2,v3])

    train_df = clean_df1[clean_df1['ndvi_diff'] < 0.25 ][['Latitude_Longtitudes', \
            'filter_red', 'filter_blue', 'filter_green', 'filter_ndvi','filter_lir','filter_swir','filter_nir', 'filter_evi', 'filter_savi', 'filter_arvi', 'Class of Land']]

    st.title("Dataset Exploration")
    st.dataframe(train_df) 

    X = train_df.drop(columns=['Class of Land', 'Latitude_Longtitudes','filter_red', 'filter_blue','filter_green', 'filter_savi', 'filter_nir'])
    y = train_df['Class of Land']
    y = np.where(y=="Rice",1,0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=40)

    ##########################################

    fig = plt.figure(figsize=(20, 10))
    sns.countplot(train_df["Class of Land"])

    st.title("Count plot of Rice and Non Rice")
    st.pyplot(fig)

    ##########################################

    fig, axes = plt.subplots(5, 2, figsize=(25,25))

    features = ['filter_red', 'filter_blue', 'filter_green', 'filter_ndvi','filter_lir','filter_swir','filter_nir', 'filter_evi', 'filter_savi', 'filter_arvi']
    for i in range(len(features)):
        ax=axes[i//2, i%2]
        ax.hist(train_df[features[i]])
        ax.title.set_text(features[i])

    fig.suptitle("Frequency distribution of TerraClimate variables",  fontsize=20)

    st.title("Distribution of features")
    st.pyplot(fig)

    ##########################################

    fig, axes = plt.subplots(5, 2, figsize=(25,25))

    features = ['filter_red', 'filter_blue', 'filter_green', 'filter_ndvi','filter_lir','filter_swir','filter_nir', 'filter_evi', 'filter_savi', 'filter_arvi']
    for i in range(len(features)):
        sns.kdeplot(data = train_df,x = features[i], hue = "Class of Land", ax=axes[i//2, i%2])

    fig.suptitle("Frequency distribution of TerraClimate variables",  fontsize=20)

    st.title("Distribution of features by Class of Land")
    st.pyplot(fig)

    ##########################################

    cor_matrix=train_df[[ 'filter_ndvi','filter_lir','filter_swir','filter_nir', 'filter_evi', 'filter_savi', 'filter_arvi']].corr().round(decimals=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(cor_matrix, annot = True, cmap='Blues', ax = ax)

    st.title("Correlation of features")
    st.pyplot(fig)

    ##########################################

    fig, axes = plt.subplots(5, 2, figsize=(10, 10))
    
    for i in range(len(features)):
        sns.boxplot(ax = axes[i//2, i%2], data=train_df, y=features[i], x="Class of Land")

    st.title("Boxplot of each feature")
    st.pyplot(fig)

    ###########################################
    st.title("Performance on test set")
    # load
    model = pickle.load(open('Landsat\landsatmodel.pkl', "rb"))

    y_score_xgb = cross_val_predict(model, X_test, y_test, method='predict_proba', cv=10)
    fpr_xgb, tpr_xgb, threshold_xgb = roc_curve(y_test,y_score_xgb[:,1])

    fig = plt.figure(figsize=(10, 6))

    plt.plot(fpr_xgb, tpr_xgb, linewidth = 3, label='XGB(area = %0.2f)' % roc_auc_score(y_test,y_score_xgb[:,1]))
    plt.legend()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve of Models')
    plt.figure(figsize=(10, 6))

    st.pyplot(fig)


    ##############################################################
    insample_predictions = model.predict(X_test)
    fig = plt.figure(figsize=(10, 6))
    plot_confusion_matrix(y_test,insample_predictions,"Model Level 1: Logistic\nRegression Model In-Sample Results",['Rice', 'Non Rice'])
    st.pyplot(fig)

else: 
    crop_presence_data = pd.read_csv("pages\\train-Sentinel-2.csv")
    crop_presence_data.head()

    clean_df = crop_presence_data.dropna()
    clean_df['ndvi_diff'] = clean_df['filter_ndvi']-clean_df['original_ndvi']

    clean_df['time_frames']= pd.to_datetime(clean_df['time_frames'])


    clean_df1 = clean_df[(clean_df['time_frames'] == '2022-03-21')]
    


    train_df = clean_df1[clean_df1['ndvi_diff'] < 0.25 ][['Latitude_Longtitudes', \
            'filter_red', 'filter_blue', 'filter_green', 'filter_ndvi', 'filter_evi', 'filter_savi', 'Class of Land']]

    st.title("Dataset Exploration")
    st.dataframe(train_df)
    

    X = train_df[['filter_ndvi', 'filter_evi' ,'filter_savi']]
    y = train_df['Class of Land']
    y = np.where(y=="Rice",1,0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=40)

    ##########################################

    fig = plt.figure(figsize=(20, 10))
    sns.countplot(train_df["Class of Land"])

    st.title("Count plot of Rice and Non Rice")
    st.pyplot(fig)

    ##########################################

    fig, axes = plt.subplots(3, 2, figsize=(25,25))

    features = ['filter_red', 'filter_blue', 'filter_green', 'filter_ndvi', 'filter_evi', 'filter_savi']
    for i in range(len(features)):
        ax=axes[i//2, i%2]
        ax.hist(train_df[features[i]])
        ax.title.set_text(features[i])

    fig.suptitle("Frequency distribution of TerraClimate variables",  fontsize=20)

    st.title("Distribution of features")
    st.pyplot(fig)

    ##########################################
    

    fig, axes = plt.subplots(3, 2, figsize=(25,25))

    features = ['filter_red', 'filter_blue', 'filter_green', 'filter_ndvi','filter_evi', 'filter_savi',]
    for i in range(len(features)):
        sns.kdeplot(data = train_df,x = features[i], hue = "Class of Land", ax=axes[i//2, i%2])

    fig.suptitle("Frequency distribution of TerraClimate variables",  fontsize=20)

    st.title("Distribution of features by Class of Land")
    st.pyplot(fig)

    ##########################################

    cor_matrix=train_df[['filter_red', 'filter_blue', 'filter_green', 'filter_ndvi', 'filter_evi', 'filter_savi']].corr().round(decimals=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(cor_matrix, annot = True, cmap='Blues', ax = ax)

    st.title("Correlation of features")
    st.pyplot(fig)

    ##########################################

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    
    for i in range(len(features)):
        sns.boxplot(ax = axes[i//2, i%2], data=train_df, y=features[i], x="Class of Land")

    st.title("Boxplot of each feature")
    st.pyplot(fig)

    #########################
     
    st.title("Performance on test set")
    # Model
    model = pickle.load(open('Sentinel2\sentinel2.pkl','rb'))
    scaler = pickle.load(open('Sentinel2\\transform.pkl','rb'))
    scaled_input = scaler.transform(X_test)

    y_score_xgb = cross_val_predict(model, scaled_input, y_test, method='predict_proba', cv=10)
    fpr_xgb, tpr_xgb, threshold_xgb = roc_curve(y_test,y_score_xgb[:,1])

    fig = plt.figure(figsize=(10, 6))

    plt.plot(fpr_xgb, tpr_xgb, linewidth = 3, label='RandomForestClassifier(area = %0.2f)' % roc_auc_score(y_test,y_score_xgb[:,1]))
    plt.legend()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve of Models')
    plt.figure(figsize=(10, 6))

    st.pyplot(fig)


    ##############################################################
    insample_predictions = model.predict(scaled_input)
    insample_predictions = np.where(insample_predictions=="Rice",1,0)
    fig = plt.figure(figsize=(10, 6))
    plot_confusion_matrix(y_test,insample_predictions,"Model Level 1: Logistic\nRegression Model In-Sample Results",['Rice', 'Non Rice'])
    st.pyplot(fig)
