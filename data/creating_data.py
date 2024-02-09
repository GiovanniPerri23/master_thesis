import os
import xml.etree.ElementTree as ET

import pandas as pd

def create_dataframe_from_xml(xml_folder, root_findall):
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
# Conta i valori NaN nel dataframe
def nan_counts_func(df):
    nan_counts = df.isna().sum()

    # Trova le colonne con valori NaN (se ce ne sono)
    columns_with_nan = nan_counts[nan_counts > 0]

    if columns_with_nan.empty:
        print("Nessun valore NaN nel DataFrame.")
    else:
        print("Colonne con valori NaN:")
        print(columns_with_nan)
    
df_price = create_dataframe_from_xml("data_0", './/Prezzi')
df_stime = create_dataframe_from_xml("data_0", './/marketintervaldetail')
df_gas = create_dataframe_from_xml('data_1', './/negoziazione_continua')
df_demand = create_dataframe_from_xml("data_1", './/Fabbisogno')
df_price2 = create_dataframe_from_xml("data_old", './/Prezzi')