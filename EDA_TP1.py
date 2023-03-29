import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
#%%
application_data = pd.read_csv("application_data.csv")
#%%
#En el siguiente archivo encontramos una descripcion de las columnas del archivo
coldata = pd.read_csv("columns_description.csv",encoding=("latin"))
print(coldata)
#%%
print(application_data.shape)
print(application_data.head())
print(application_data.describe())
print(application_data.info())
#%%
MODE_cols = [col for col in application_data.columns if '_MODE' in col]
MEDI_cols = [col for col in application_data.columns if '_MEDI' in col]
AVG_cols = [col for col in application_data.columns if '_AVG' in col]
#%%
fig = plt.figure(figsize=(20, 50))
rows, cols = 10, 5
for idx, num in enumerate(MODE_cols):
    ax = fig.add_subplot(rows, cols, idx+1)
    ax.grid(alpha = 0.7, axis ="both")
    sns.kdeplot(x = num, fill = True,color ="#3386FF",linewidth=0.6, data = application_data)     
    ax.set_xlabel(num)
    ax.legend()
fig.tight_layout()
fig.show()
#%%
fig = plt.figure(figsize=(20, 50))
rows, cols = 10, 3
for idx, num in enumerate(MEDI_cols):
    ax = fig.add_subplot(rows, cols, idx+1)
    ax.grid(alpha = 0.7, axis ="both")
    sns.kdeplot(x = num, fill = True,color ="#4432FF",linewidth=0.6, data = application_data)     
    ax.set_xlabel(num)
    ax.legend()
fig.tight_layout()
fig.show()
#%%
fig = plt.figure(figsize=(20, 50))
rows, cols = 10, 3
for idx, num in enumerate(AVG_cols):
    ax = fig.add_subplot(rows, cols, idx+1)
    ax.grid(alpha = 0.7, axis ="both")
    sns.kdeplot(x = num, fill = True,color ="#ed14f5",linewidth=0.6, data = application_data)     
    ax.set_xlabel(num)
    ax.legend()
fig.tight_layout()
fig.show()
#%% Quitamos las columnas normalizadas ya que no son necesarias de momento para realizar un análisis, posiblemente sean útiles para una segunda etapa para armar clusters
application_data.drop(columns = MODE_cols,inplace=True)
application_data.drop(columns = MEDI_cols,inplace=True)
application_data.drop(columns = AVG_cols,inplace=True)
#%% Transformamos variables en una tabla temporal para poder hacer un pivot table
temp = application_data.copy()
temp['FLAG_OWN_CAR'] = temp['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
temp['FLAG_OWN_REALTY'] = temp['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})
temp['TARGET'] = temp['TARGET'].map({1: 'Y', 0: 'N'})
#%%
FLAGS_cols = [col for col in application_data.columns if 'FLAG' in col]
#%%
pivot = pd.pivot_table(temp, values=(FLAGS_cols), index=('TARGET'))
#pivot.to_csv("FLAG_DIFFS.csv")
#%% Ya que no parece que ninguna FLAG sea relevante para TARGET que sera mas adelante nuestra variable a predecir, descartamos esas columnas
application_data.drop(columns = FLAGS_cols,inplace=True)
#%% Ordenamos NULLs
nulls = pd.DataFrame()
nulls['Percentage'] = application_data.isnull().sum()/len(application_data)*100
nulls['Count'] = application_data.isnull().sum()
print(nulls.sort_values('Percentage',ascending=(False)))
#%% Note que algunos valores en días tienen valores negativos, para hacer un analisis mas adelante seguro tengo que convertirlos a positivos para poder tener una linea de tiempo normal asi que me ahorro el trabajo a futuro
application_data['DAYS_BIRTH'] = abs(application_data['DAYS_BIRTH'])
application_data['DAYS_ID_PUBLISH'] = abs(application_data['DAYS_ID_PUBLISH'])
application_data['DAYS_EMPLOYED'] = abs(application_data['DAYS_EMPLOYED'])
application_data['DAYS_REGISTRATION'] = abs(application_data['DAYS_REGISTRATION'])
application_data['DAYS_LAST_PHONE_CHANGE'] = abs(application_data['DAYS_LAST_PHONE_CHANGE'])
#%% Ahora que limpiamos data, podemos empezar a jugar un poco
defaulters=application_data[application_data.TARGET==1]
nondefaulters=application_data[application_data.TARGET==0]
#%%
percentage_defaulters=round((len(defaulters)*100)/len(application_data),2)
print('%',percentage_defaulters)
#%%
sns.countplot(application_data.TARGET)
plt.xlabel("TARGET")
plt.ylabel("Number of TARGET")
plt.title("Distribution of TARGET Variable")
plt.show()
#%% Esta funcion la robé de internet
def biplot(data, var,label_rotation):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,10))
    s1=sns.barplot(ax=ax1,x=defaulters[var].value_counts().index, data=defaulters, y= 100.* defaulters[var].value_counts(normalize=True))
    if(label_rotation):
        s1.set_xticklabels(s1.get_xticklabels(),rotation=90,fontsize=12)
    ax1.set_title('Distribution of '+ '%s' %var +' - Defaulters', fontsize=15)
    ax1.set_xlabel('%s' %var,fontsize=15)
    ax1.set_ylabel("% of Loans",fontsize=15)
   
    s2=sns.barplot(ax=ax2,x=nondefaulters[var].value_counts().index, data=nondefaulters, y= 100.* nondefaulters[var].value_counts(normalize=True))
    if(label_rotation):
        s2.set_xticklabels(s2.get_xticklabels(),rotation=90,fontsize=12)
    ax2.set_xlabel('%s' %var, fontsize=15)
    ax2.set_ylabel("% of Loans", fontsize=15)
    ax2.set_title('Distribution of '+ '%s' %var +' - Non-Defaulters', fontsize=15)
    plt.show()
#%%
OBJ_cols = application_data.select_dtypes('object').columns
fig = plt.figure(figsize=(20, 50))
rows, cols = 10, 3
for j in OBJ_cols:
    biplot(application_data,j,True)
#%%
AMT_cols = [col for col in application_data.columns if 'AMT_' in col]
#%%
fig = plt.figure(figsize=(20, 50))
rows, cols = 10, 5
for idx, num in enumerate(AMT_cols):
    ax = fig.add_subplot(rows, cols, idx+1)
    ax.grid(alpha = 0.7, axis ="both")
    sns.kdeplot(x = num, fill = True,color ="#eb345e",linewidth=0.6, data = application_data)     
    ax.set_xlabel(num)
    ax.legend()
fig.tight_layout()
fig.show()
#%%
AMT_cols = [col for col in application_data.columns if 'AMT' in col]
AMT_cols = [col for col in AMT_cols if 'BUREAU' not in col]
#%%
fig = plt.figure(figsize=(20, 50))
rows, cols = 10, 5
for idx, num in enumerate(AMT_cols):
    ax = fig.add_subplot(rows, cols, idx+1)
    ax.grid(alpha = 0.7, axis ="both")
    sns.boxplot(x = num, data = application_data)     
    ax.set_xlabel(num)
    ax.legend()
fig.tight_layout()
fig.show()
#%%
DAYS_cols = [col for col in application_data.columns if 'DAYS' in col]
#%%
fig = plt.figure(figsize=(20, 50))
rows, cols = 10, 5
for idx, num in enumerate(DAYS_cols):
    ax = fig.add_subplot(rows, cols, idx+1)
    ax.grid(alpha = 0.7, axis ="both")
    sns.kdeplot(x = num, fill = True,color ="#a83234",linewidth=0.6, data = application_data)     
    ax.set_xlabel(num)
    ax.legend()
fig.tight_layout()
fig.show()
#%%
INT_cols = application_data.select_dtypes('int64').columns
FLOAT_cols = OBJ_cols = application_data.select_dtypes('float64').columns
#%%
CORR_cols = [j for i in [INT_cols, FLOAT_cols] for j in i]
#%%
defaulters_1=defaulters[CORR_cols]
defaulters_correlation = defaulters_1.corr(method = 'pearson', numeric_only = True)
round(defaulters_correlation, 3)
#%%
plt.figure(figsize=(20,15))
sns.heatmap(defaulters_correlation, cmap='YlGnBu', annot=True)
plt.show()
#%%
nondefaulters_1=nondefaulters[CORR_cols]
nondefaulters_correlation = nondefaulters_1.corr(method = 'pearson')
round(nondefaulters_correlation, 3)
#%%
plt.figure(figsize=(20,15))
sns.heatmap(nondefaulters_correlation, cmap='YlGnBu', annot=True)
plt.show()
#%%
def IQR_method (df,n,features):
    """
    Takes a dataframe and returns an index list corresponding to the observations 
    containing more than n outliers according to the Tukey IQR method.
    """
    outlier_list = []
    
    for column in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[column], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[column],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determining a list of indices of outliers
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step )].index
        # appending the list of outliers 
        outlier_list.extend(outlier_list_column)
        
    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)        
    multiple_outliers = list( k for k, v in outlier_list.items() if v > n )
    
    # Calculate the number of records below and above lower and above bound value respectively
    out1 = df[df[column] < Q1 - outlier_step]
    out2 = df[df[column] > Q3 + outlier_step]
    
    print('Total number of deleted outliers is:', out1.shape[0]+out2.shape[0])
    
    return multiple_outliers
#%%
Outliers_IQR = IQR_method(application_data,1,CORR_cols)
df_out = application_data.drop(Outliers_IQR, axis = 0).reset_index(drop=True)
#%%
print ('Cantidad de personas con problemas de pago con outliers: ', len(application_data[application_data['TARGET'] == 1]))
print ('Cantidad de personas con problemas de pago sin outliers: ', len(df_out[df_out['TARGET'] == 1]))
#%%
defaulters2=df_out[df_out.TARGET==1]
nondefaulters2=df_out[df_out.TARGET==0]
#%%
defaulters_2=defaulters2[CORR_cols]
defaulters_correlation2 = defaulters_2.corr(method = 'pearson')
round(defaulters_correlation2, 3)
#%%
plt.figure(figsize=(20,15))
sns.heatmap(defaulters_correlation, cmap='YlGnBu', annot=True)
plt.show()
#%%
nondefaulters_2=nondefaulters2[CORR_cols]
nondefaulters_correlation2 = nondefaulters_2.corr(method = 'pearson')
round(nondefaulters_correlation2, 3)
#%%
plt.figure(figsize=(20,15))
sns.heatmap(nondefaulters_correlation2, cmap='YlGnBu', annot=True)
plt.show()





































































































































































































