import os
import dataset_constants

from pandas import read_csv
from util.open_arff import open_arff


def get_data(file_path, dataset, is_arff=True):
    file_path = os.path.realpath(file_path)
    if is_arff:
        data = open_arff(file_path)
        data = rename(data, dataset)
    else:
        data = read_csv(file_path)
        data = rename_col_name(data)
    return data


def rename_col_name(data):
    cols_rename = {
        'class': 'classes'
    }
    data = data.rename(columns=cols_rename)
    return data


def rename(data, dataset):
    data = rename_col_name(data)
    data_rename = {}

    if dataset == dataset_constants.BRODATZ:
        data_rename = rename_brodatz_dataset()
    if dataset == dataset_constants.KYLBERG:
        data_rename= rename_kylberg_dataset()

    data.classes = data.classes.map(data_rename)
    return data


def rename_kylberg_dataset():
    return {
        b'BL1': 'BL1',
        b'BL2': 'BL2',
        b'CAN': 'CAN',
        b'CE1': 'CE1',
        b'CE2': 'CE2',
        b'CUS': 'CUS',
        b'FL1': 'FL1',
        b'FL2': 'FL2',
        b'GRA': 'GRA',
        b'LEN': 'LEN',
        b'LIN': 'LIN',
        b'OAT': 'OAT',
        b'PEA': 'PEA',
        b'RI1': 'RI1',
        b'RI2': 'RI2',
        b'RUG': 'RUG',
        b'SAN': 'SAN',
        b'SC1': 'SC1',
        b'SC2': 'SC2',
        b'SCR': 'SCR',
        b'SE1': 'SE1',
        b'SE2': 'SE2',
        b'SES': 'SES',
        b'ST1': 'ST1',
        b'ST2': 'ST2',
        b'ST3': 'ST3',
        b'STL': 'STL',
        b'WAL': 'WAL',
    }


def rename_brodatz_dataset():
    return {
       b'BAR': 'BAR',
       b'BRI': 'BRI',
       b'BUB': 'BUB',
       b'GRA': 'GRA',
       b'LEA': 'LEA',
       b'PIG': 'PIG',
       b'RAF': 'RAF',
       b'SAN': 'SAN',
       b'STR': 'STR',
       b'WAT': 'WAT',
       b'WEA': 'WEA',
       b'WOL': 'WOL',
       b'WOO': 'WOO'
    }
