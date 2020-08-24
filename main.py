# from scipy.io import arff
# import pandas as pd

# data = arff.loadarff('Haralick.arff')
# df = pd.DataFrame(data[0])

# print(df.head())

import arff, numpy as np
dataset = arff.load(open('mydataset.arff', 'rb'))
data = np.array(dataset['data'])
print(data)
