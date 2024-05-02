import pandas as pd

# Assuming the sample data is in a string format
sample_data = """0,Maruti Wagon R LXI CNG,Mumbai,2010,72000,CNG,Manual,First,26.6 km/kg,998 CC,58.16 bhp,5.0,,1.75
1,Hyundai Creta 1.6 CRDi SX Option,Pune,2015,41000,Diesel,Manual,First,19.67 kmpl,1582 CC,126.2 bhp,5.0,,12.5
2,Honda Jazz V,Chennai,2011,46000,Petrol,Manual,First,18.2 kmpl,1199 CC,88.7 bhp,5.0,8.61 Lakh,4.5
3,Maruti Ertiga VDI,Chennai,2012,87000,Diesel,Manual,First,20.77 kmpl,1248 CC,88.76 bhp,7.0,,6.0
4,Audi A4 New 2.0 TDI Multitronic,Coimbatore,2013,40670,Diesel,Automatic,Second,15.2 kmpl,1968 CC,140.8 bhp,5.0,,17.74
5,Hyundai EON LPG Era Plus Option,Hyderabad,2012,75000,LPG,Manual,First,21.1 km/kg,814 CC,55.2 bhp,5.0,,2.35
6,Nissan Micra Diesel XV,Jaipur,2013,86999,Diesel,Manual,First,23.08 kmpl,1461 CC,63.1 bhp,5.0,,3.5"""

# Split the sample data into lines and then split each line by commas to create a list of lists
data_lines = [line.split(',') for line in sample_data.split('\n')]

# Load the data into a DataFrame
df = pd.DataFrame(data_lines, columns=['Index', 'Car_Name', 'Location', 'Year', 'Kilometers_Driven',
                                       'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage',
                                       'Engine', 'Power', 'Seats', 'New_Price', 'Price'])
# Convert columns to numeric types
df['Index'] = pd.to_numeric(df['Index'])
df['Year'] = pd.to_numeric(df['Year'])
df['Kilometers_Driven'] = pd.to_numeric(df['Kilometers_Driven'])
df['Mileage'] = pd.to_numeric(df['Mileage'].str.split().str[0])
df['Engine'] = pd.to_numeric(df['Engine'].str.split().str[0])
df['Power'] = pd.to_numeric(df['Power'].str.split().str[0], errors='coerce')  # Use errors='coerce' to handle non-numeric values
df['Seats'] = pd.to_numeric(df['Seats'])
df['New_Price'] = pd.to_numeric(df['New_Price'].str.replace(' Lakh', ''), errors='coerce')  # Remove 'Lakh' and convert to numeric
df['Price'] = pd.to_numeric(df['Price'])

# Display the DataFrame and its information
print(df)
print(df.info())
