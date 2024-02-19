import os
import xml.etree.ElementTree as ET

import pandas as pd

def create_dataframe_from_xml(xml_folder, root_findall):
    """
    Create a pandas DataFrame from XML files in a given folder.

    Args:
        xml_folder (str): Path to the folder containing XML files.
        root_findall (str): XPath expression to find the root elements in the XML files.

    Returns:
        pandas.DataFrame: DataFrame containing the data extracted from the XML files.
    """

    df = pd.DataFrame() 

    for filename in os.listdir(xml_folder):
        if filename.endswith(".xml"):
            xml_file = os.path.join(xml_folder, filename)

            # Analizza il file XML
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Inizializza una lista per memorizzare i dati
            data = []

            # Itera sugli elementi desiderati nel file XML
            for element in root.findall(root_findall):
                data_dict = {}
                for child in element:
                    data_dict[child.tag] = child.text
                data.append(data_dict)

            # Crea un DataFrame dai dati del file XML
            temp_df = pd.DataFrame(data)

            # Aggiungi il DataFrame del file corrente al DataFrame principale
            df = pd.concat([df, temp_df], ignore_index=True)

    return df

def nan_counts_func(df):
    """
    Calculates the number of NaN values in each column of a DataFrame.
    """
    nan_counts = df.isna().sum()

    # Trova le colonne con valori NaN (se ce ne sono)
    columns_with_nan = nan_counts[nan_counts > 0]

    if columns_with_nan.empty:
        print("Nessun valore NaN nel DataFrame.")
    else:
        print("Colonne con valori NaN:")
        print(columns_with_nan)

if __name__ == "__main__":
    df_price = create_dataframe_from_xml("prices", './/Prezzi')
    print('Price data created')
    df_stime = create_dataframe_from_xml("stimafabbisogno", './/marketintervaldetail')
    print('Stime data created')
    df_gas = create_dataframe_from_xml('gas', './/negoziazione_continua')
    print('Gas data created')
    df_demand = create_dataframe_from_xml("fabbisogno", './/Fabbisogno')
    print('Demand data created')

    # Preprocessing GAS data
    df_gas = df_gas.groupby('NomeProdotto')['PrezzoMedio'].last().reset_index()
    df_gas.columns = ['NomeProdotto', 'GAS']
    df_gas['NomeProdotto'] = df_gas['NomeProdotto'].str.replace('MGP-', '').str.replace('WE-', '')

    # To FIX: bisogna rimuovere i valori WE- dal dataset, per ora li rimuovo manualmente
    df_gas = df_gas.iloc[:2176]
    # df_gas = df_gas.iloc[:31,]
    df_gas['DataG'] = pd.to_datetime(df_gas['NomeProdotto'])
    df_gas = df_gas[['DataG', 'GAS']]
    df_gas['GAS'] = pd.to_numeric(df_gas['GAS'], errors='coerce')

    df_gas = df_gas.loc[df_gas.index.repeat(24)].reset_index(drop=True)

    # Preprocessing demand data
            # Estrai le colonne numeriche da "PUN" a "XGRE"
    numeric_columns = df_demand.loc[:, 'Italia':'SUD']

    for col in numeric_columns.columns:
        df_demand[col] = df_demand[col].str.replace(',', '.').astype(float)
    df_demand['Data'] = pd.to_datetime(df_demand['Data'])

    df_demand = df_demand.drop(columns = 'Mercato')

    df_demand.rename(columns={"Italia": "Total_Load"}, errors="raise")

    # Preprocessing price data
    numeric_columns = df_price.loc[:, 'PUN':'XGRE']

    for col in numeric_columns.columns:
        df_price[col] = df_price[col].str.replace(',', '.').astype(float)
    
    df_price['Data'] = pd.to_datetime(df_price['Data'])

    df_price = df_price.drop(columns = 'Mercato')

    df_price = df_price.drop(df_price.columns[11:25], axis=1)


    df = pd.concat([df_price, df_gas, df_demand], axis=1)
    df.to_csv("data_to_process.csv", index=True)
    print('Data saved to data_to_process.csv')