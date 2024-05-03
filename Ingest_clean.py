import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# df = dataframe using pandas
df = pd.read_csv('train-data.csv')

print(df.head())
print(df.info())

# Calculate and round the percentage of missing values
print(round((df.isna().sum() / len(df)) * 100, 2))

# drops columns from the dataframe that are not used
df.drop(columns=['Unnamed: 0', 'New_Price'], inplace=True)

# Check/remove duplicates
print(df.duplicated().sum())

# Converting numbers in strings to floats
df['Mileage'] = df['Mileage'].str.split(expand=True)[0].astype(float)
df['Engine'] = df['Engine'].str.split(expand=True)[0].astype(float)

# null bhp is set to none
print((df['Power'] == 'null bhp').sum())
df['Power'] = df['Power'].replace('null bhp', None)
df['Power'] = df['Power'].str.split(expand=True)[0].astype(float)

# Display a random sample and describes the data
print(df.sample())
print(df.describe())

# describe dtype object columns
print(df.select_dtypes('object').describe())

# remove outliers for columns listed
for col in ['Engine', 'Power', 'Kilometers_Driven', 'Mileage', 'Price']:
    quantity1 = df[col].quantile(0.25)
    quantity3 = df[col].quantile(0.75)
    IQR = quantity3 - quantity1
    lower_bound = quantity1 - 2.5 * IQR
    upper_bound = quantity3 + 2.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# drop company and model from name column
df['Company'] = df['Name'].str.split(expand=True)[0].str.lower()
df['Model'] = df['Name'].str.split().str[0:2].str.join(' ').str.lower()
df.drop(columns=['Name'], inplace=True)

# string to numerical
df['Owner_Type'] = df['Owner_Type'].map({"First": 3, "Second": 2, "Third": 1, "Fourth & Above": 0})

# distributions and boxplots for each listed columns
def plot_num(df, col):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    sns.histplot(data=df, x=col, kde=True, ax=ax[0])
    sns.boxplot(data=df, x=col, ax=ax[1])
    ax[0].set_title(f'Distribution of {col}')
    ax[1].set_title(f'{col} Boxplot')
    plt.show()

plot_num(df, 'Kilometers_Driven')
print(df['Fuel_Type'].value_counts())

# Filter out 'LPG' and 'Electric' from Fuel_Type column due to very few of both and cannot trend accurately
df = df[~df['Fuel_Type'].isin(['LPG', 'Electric'])]

print(df['Transmission'].value_counts())

plot_num(df, 'Mileage')
plot_num(df, 'Engine')
plot_num(df, 'Power')

print(df['Seats'].value_counts())

# Filter out rows with Seats less than 5 or greater than 8 due to very few of both and cannot trend accurately
df = df[df.Seats.between(5, 8)]

print(df.Model.value_counts())

# Drop models with counts less than threshold to avoid bad data
threshold = 10
counts = df.Model.value_counts()
to_drop = counts[counts < threshold].index.tolist()
df = df[~df['Model'].isin(to_drop)]

print(df.head())

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()
