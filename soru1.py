import numpy as np
import pandas as pd

data_set = [
    {'inputs': [2, 3, 1, 4, 5], 'output': 15},
    {'inputs': [1, 2, 1, 0, 3], 'output': 7},
    {'inputs': [5, 2, 7, 1, 8], 'output': 23},
    {'inputs': [3, 1, 4, 6, 2], 'output': 16},
    {'inputs': [3, 5, 9, 6, 8], 'output': 12},
    {'inputs': [3, 5, 1, 2, 9], 'output': 11},
    {'inputs': [8, 2, 1, 5, 0], 'output': 20}
]


inputs_array = np.array([data_point['inputs'] for data_point in data_set])
correlation_matrix = np.corrcoef(inputs_array, rowvar=False)

# Korelasyon matrisini gösterme
print("Korelasyon Matrisi:")
print(correlation_matrix)

# Veri Seti Tamlığı
df = pd.DataFrame(data_set)
missing_values = df.isnull().sum()
print("Eksik Değerler:")
print(missing_values)


