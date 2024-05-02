import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px


df = pd.read_csv('train-data.csv')
df.head()
df.info()

round((df.isna().sum()/len(df))*100,2)

df.drop(columns = ['Unnamed: 0','New_Price'],inplace = True)

df.duplicated().sum()

df['Mileage'] = df['Mileage'].str.split(expand=True)[0].astype(float)
df['Engine'] = df['Engine'].str.split(expand=True)[0].astype(float)

(df['Power'] == 'null bhp').sum()

df['Power'] = df['Power'].replace('null bhp',None)
df['Power'] = df['Power'].str.split(expand=True)[0].astype(float)

df.sample()

df.describe()

df.select_dtypes('object').describe()


for col in ['Engine','Power','Kilometers_Driven','Mileage','Price']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.5*IQR
    upper_bound = Q3 + 2.5*IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

df['Name'].nunique()

df['Company'] = df['Name'].str.split(expand=True)[0].str.lower()
df['Model'] = df['Name'].str.split().str[0:2].str.join(' ').str.lower()
df.drop(columns = ['Name'],inplace = True)

df['Company'].nunique() , df['Model'].nunique()

def remove_outliers(df,col,lower,upper):
    df = df[ (df[col]>lower) & (df[col]<upper) ]
    return df

def plot_num(df,col):
    fig ,ax = plt.subplots(1,2,figsize=(16,4))
    sns.histplot(df,x=col,kde=True,ax=ax[0])
    sns.boxplot(df,x=col,ax=ax[1])
    ax[0].set_title(f'Distribution of {col}')
    ax[1].set_title(f'{col} Boxplot')
    fig.show();


plot_num(df,'Kilometers_Driven')


df['Fuel_Type'].value_counts()


df = df[~df['Fuel_Type'].isin(['LPG','Electric'])]


df['Transmission'].value_counts()



df['Owner_Type'] = df['Owner_Type'].map({"First":3,"Second":2,"Third":1,"Fourth & Above":0})

plot_num(df,'Mileage')

plot_num(df,'Engine')

plot_num(df,'Power')


df['Seats'].value_counts()

df = df[df.Seats.between(5,8)]

df.Model.value_counts()

threshold = 10
counts = df.Model.value_counts()
to_drop = counts[counts<threshold].index.tolist()
df = df[~df['Model'].isin(to_drop)]
