from scipy.io.arff import loadarff
import pandas as pd


def open_arff(path):
    data = loadarff(path)
    df = pd.DataFrame(data[0])
    return df
