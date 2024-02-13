import numpy as np


def make_yearly_dataset(df, start_year, end_year):
    if start_year == end_year:
        return df[df.index.year == start_year]
    else:
        return df[(df.index.year >= start_year) & (df.index.year <= end_year)]

def transform_time_features(time_data, max_value):
    sin_values = np.sin(2 * np.pi * time_data / max_value)
    cos_values = np.cos(2 * np.pi * time_data / max_value)
    return sin_values, cos_values

def create_cyclic_features(dataset_input):
    dataset_input.loc[:, 'hour_sin'], dataset_input.loc[:, 'hour_cos'] = transform_time_features(dataset_input['Ora'], 24)
    dataset_input.loc[:, 'day_sin'], dataset_input.loc[:, 'day_cos'] = transform_time_features(dataset_input['GiornoSettimana'], 7)
    dataset_input.drop(['Ora', 'GiornoSettimana'], axis=1, inplace=True)
    return dataset_input