import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import t, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels
from sklearn import datasets
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

from sklearn import metrics

import streamlit as st

import PIL as P

# implémentation des données ----------------------------------------------------------------------------------------------------------
    # reg log

lr_select = pickle.load(open('lr_select.pickle', 'rb'))

#X_test=pd.read_csv(r"C:\Users\kylli\OC_notebook\P9\X_test.csv",sep=",")
X_train=pd.read_csv("X_train.csv",sep=",")
Y_test=pd.read_csv("Y_test.csv",sep=",")
#Y_pred_2_training=pd.read_csv(r"C:\Users\kylli\OC_notebook\P9\Y_pred_2_training.csv",sep=",")
#Y_pred_kmeans_training=pd.read_csv(r"C:\Users\kylli\OC_notebook\P9\Y_pred_kmeans_training.csv",sep=",")

    # Kmeans

res = pickle.load(open('res.pickle', 'rb'))

# fonction reg log et Kmeans ----------------------------------------------------------------------------------------------------------

def algo(df_test_ini, method):
    
    if method != '...':
    
        X_train=pd.read_csv("X_train.csv",sep=",")
        df_test=df_test_ini[['diagonal','height_left','height_right','margin_low','margin_up','length']]
        std.fit_transform(X_train)
        X_test_std = pd.DataFrame(std.transform(df_test))
        X_test_std.columns = df_test.columns

        if method == 'Regression logistique':
            Y_pred_2=lr_select.estimator_.predict(X_test_std[X_test_std.columns[lr_select.support_]])
            Y_pred_2=pd.DataFrame(Y_pred_2)
            
            
            result = pd.DataFrame(Y_pred_2.value_counts(normalize=True)*100)
            result.rename(columns = {0 : '%'}, inplace = True)
            result['count']=[Y_pred_2[0][Y_pred_2[0]==0].count() , Y_pred_2[0][Y_pred_2[0]==1].count()]
            
            df_result=pd.DataFrame(df_test_ini['id'].copy())
            df_result['result']=Y_pred_2

        elif method == 'Kmeans':
            Y_pred_kmeans = pd.DataFrame(res.fit_predict(std.transform(df_test)))

            for i in range(len(Y_pred_kmeans)):
                if int(Y_pred_kmeans.iloc[i]) == 0 :
                    Y_pred_kmeans.iloc[i]=1
                else:
                    Y_pred_kmeans.iloc[i]=0
            # cette boucle inverse les 0 et 1 car le cluster 1 correspond aux faux billets
                       
            result = pd.DataFrame(Y_pred_kmeans.value_counts(normalize=True)*100)
            result.rename(columns = {0 : '%'}, inplace = True)
            result['count']=[Y_pred_kmeans[0][Y_pred_kmeans[0]==0].count() , Y_pred_kmeans[0][Y_pred_kmeans[0]==1].count()]

            df_result=pd.DataFrame(df_test_ini['id'].copy())
            df_result['result']=Y_pred_kmeans
            
    else:
        result = 'Aucun algorithme choisis'
        df_result=''
        
    return result,df_result

# Affichage ------------------------------------------------------------------------------------------------------------------------------

st.write("""
# Application Vrais/Faux billets
""")
st.image(P.Image.open('photo_2.png'))
st.image(P.Image.open('photo_P10.png'))
st.image(P.Image.open('photo_2.png'))
st.write("""
##### Gressier Kyllian \n ##### Mentor : Abdou Karim Kandji
""")
st.image(P.Image.open('photo_2.png'))
    # Importer le fichier

X_test_csv =st.file_uploader('Veuiller déposer les données')

    # Afficher le resultat du modèle

button = st.checkbox('Cliquez pour les importer')
st.image(P.Image.open('photo_2.png'))
if button:
    X_test = pd.read_csv(X_test_csv,sep=',')

    select = st.selectbox('Choisissez un modèle',['...','Regression logistique','Kmeans'], index=0)

    result,df_result=algo(X_test,method=select)
    st.write(result,df_result)


# streamlit run C:\Users\kylli\OC_notebook\P9\app_P10.py
