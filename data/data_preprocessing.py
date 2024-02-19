import pandas as pd
import numpy as np

from datetime import date
from workalendar.europe import Italy


def load_data():
    # Load data
    data = pd.read_csv('data_to_process.csv')
    return data


def split_temporal_indices(df):
    # Estrai l'anno, il mese, il giorno della settimana e l'ora dalla colonna 'DataCompleta'
    df['Mese'] = df['Data'].dt.month
    df['GiornoSettimana'] = df['Data'].dt.dayofweek
    df['Weekend'] = df['GiornoSettimana'].apply(lambda x: 1 if x > 4 else 0)  # 5 e 6 rappresentano sabato e domenica
    
    return df

def set_index(df):
    df['Time_data'] = df['Data'].apply(lambda t: pd.Timestamp(t).normalize().tz_localize('CET').tz_convert('UTC')) + \
                            df['Ora'].apply(lambda h: pd.Timedelta(hours=h))

    df.set_index('Time_data', inplace=True)
    return df

def create_holidays(df):
    years = list(range(2018, 2024))  # 2018 incluso, 2024 escluso
    holiday_dates = []

    for year in years:
        holidays_year = Italy().holidays(year)
        holiday_dates.extend([date for date, _ in holidays_year])

    holiday_dates_str = [date.strftime('%Y-%m-%d') for date in holiday_dates]

    # crea un indice date time index
    date_range = pd.date_range(start="2018-01-01", end="2023-12-14", freq='H')


    df_holiday = pd.DataFrame(index=date_range)

    # Add a column 'is_holiday' and set values based on holiday_dates_str
    df_holiday['is_holiday'] = df_holiday.index.strftime('%Y-%m-%d').isin(holiday_dates_str).astype(int)

    # Applico la stessa timezone a df_holiday
    df_holiday.index = df_holiday.index.tz_localize(df.index.tz)

    df = pd.merge(df, df_holiday, left_index=True, right_index=True, how='left')

    df['is_holiday'] = df['is_holiday'].astype(int)
    df['holiday'] = df['is_holiday'] | df['Weekend'] # operatore come l' OR 

    return df


if __name__ == '__main__':
    df = load_data()
    df = split_temporal_indices(df)
    # df.reset_index(inplace=True)
    df = set_index(df)

    