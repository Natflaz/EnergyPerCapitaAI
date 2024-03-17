import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def convert_column_to_float(dataframe, column_name):
    try:
        dataframe[column_name] = dataframe[column_name].str.replace(',', '.').astype(float)
    except ValueError:
        print(f"mauvais format avec {column_name}")
    return dataframe


df = pd.read_csv("/home/natflaz/Documents/IUTinfo/s4/data/globalAnalyse/data/global-data-on-sustainable-energy.csv")

df.drop_duplicates()

df.drop(columns=['Entity','Financial flows to developing countries (US $)', 'Renewables (% equivalent primary energy)',
                 'Renewable-electricity-generating-capacity-per-capita', 'Value_co2_emissions_kt_by_country', 'Access to clean fuels for cooking'], inplace=True)

df = convert_column_to_float(df, "Density\\n(P/Km2)")

columns_to_fill_mean = ['Renewable energy share in the total final energy consumption (%)',
                        'Electricity from nuclear (TWh)', 'Energy intensity level of primary energy (MJ/$2017 PPP GDP)', 'gdp_growth', 'gdp_per_capita', 'Access to electricity (% of population)', 'Electricity from fossil fuels (TWh)', 'Electricity from renewables (TWh)', 'Low-carbon electricity (% electricity)',  'Land Area(Km2)', 'Latitude', 'Longitude', "Density\\n(P/Km2)" ]

df[columns_to_fill_mean] = df[columns_to_fill_mean].fillna(method='bfill')


columns_to_normalize = [
    'Renewable energy share in the total final energy consumption (%)',
    'Electricity from fossil fuels (TWh)', 'Electricity from nuclear (TWh)',
    'Electricity from renewables (TWh)',
    'Primary energy consumption per capita (kWh/person)',
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
    'gdp_growth', 'gdp_per_capita',
    "Density\\n(P/Km2)"
]

scaler = MinMaxScaler()

df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
# %%


"""
voting : 
avec mediane
  MSE: 12946405.548422912
  MAE: 2086.4761135962704
  R2: 0.9892213359173022
  
avec Moyenne 
  MSE: 13076242.629232751
  MAE: 2074.8787226045893
  R2: 0.9891132387103754
  
  
"""