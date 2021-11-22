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
    elif dataset == dataset_constants.KYLBERG:
        data_rename = rename_kylberg_dataset()
    elif dataset == dataset_constants.USPTEX:
        data_rename = rename_usptex_dataset()
    elif dataset == dataset_constants.VISTEX:
        data_rename = rename_vistex_dataset()

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


def rename_vistex_dataset():
    return {
        b'BAR': 'BAR',
        b'BRI': 'BRI',
        b'BUI': 'BUI',
        b'CLO': 'CLO',
        b'FAB': 'FAB',
        b'FLO': 'FLO',
        b'FOO': 'FOO',
        b'GRA': 'GRA',
        b'LEA': 'LEA',
        b'MET': 'MET',
        b'MIS': 'MIS',
        b'PAI': 'PAI',
        b'SAN': 'SAN',
        b'STO': 'STO',
        b'TER': 'TER',
        b'TIL': 'TIL',
        b'WAT': 'WAT',
        b'WHE': 'WHE',
        b'WOO': 'WOO'
    }


def rename_usptex_dataset():
    return {
        b'001': '001',
        b'002': '002',
        b'003': '003',
        b'004': '004',
        b'005': '005',
        b'006': '006',
        b'007': '007',
        b'008': '008',
        b'009': '009',
        b'010': '010',
        b'011': '011',
        b'012': '012',
        b'013': '013',
        b'014': '014',
        b'015': '015',
        b'016': '016',
        b'017': '017',
        b'018': '018',
        b'019': '019',
        b'020': '020',
        b'021': '021',
        b'022': '022',
        b'023': '023',
        b'024': '024',
        b'025': '025',
        b'026': '026',
        b'027': '027',
        b'028': '028',
        b'029': '029',
        b'030': '030',
        b'031': '031',
        b'032': '032',
        b'033': '033',
        b'034': '034',
        b'035': '035',
        b'036': '036',
        b'037': '037',
        b'038': '038',
        b'039': '039',
        b'040': '040',
        b'041': '041',
        b'042': '042',
        b'043': '043',
        b'044': '044',
        b'045': '045',
        b'046': '046',
        b'047': '047',
        b'048': '048',
        b'049': '049',
        b'050': '050',
        b'051': '051',
        b'052': '052',
        b'053': '053',
        b'054': '054',
        b'055': '055',
        b'056': '056',
        b'057': '057',
        b'058': '058',
        b'059': '059',
        b'060': '060',
        b'061': '061',
        b'062': '062',
        b'063': '063',
        b'064': '064',
        b'065': '065',
        b'066': '066',
        b'067': '067',
        b'068': '068',
        b'069': '069',
        b'070': '070',
        b'071': '071',
        b'072': '072',
        b'073': '073',
        b'074': '074',
        b'075': '075',
        b'076': '076',
        b'077': '077',
        b'078': '078',
        b'079': '079',
        b'080': '080',
        b'081': '081',
        b'082': '082',
        b'083': '083',
        b'084': '084',
        b'085': '085',
        b'086': '086',
        b'087': '087',
        b'088': '088',
        b'089': '089',
        b'090': '090',
        b'091': '091',
        b'092': '092',
        b'093': '093',
        b'094': '094',
        b'095': '095',
        b'096': '096',
        b'097': '097',
        b'098': '098',
        b'099': '099',
        b'100': '100',
        b'101': '101',
        b'102': '102',
        b'103': '103',
        b'104': '104',
        b'105': '105',
        b'106': '106',
        b'107': '107',
        b'108': '108',
        b'109': '109',
        b'110': '110',
        b'111': '111',
        b'112': '112',
        b'113': '113',
        b'114': '114',
        b'115': '115',
        b'116': '116',
        b'117': '117',
        b'118': '118',
        b'119': '119',
        b'120': '120',
        b'121': '121',
        b'122': '122',
        b'123': '123',
        b'124': '124',
        b'125': '125',
        b'126': '126',
        b'127': '127',
        b'128': '128',
        b'129': '129',
        b'130': '130',
        b'131': '131',
        b'132': '132',
        b'133': '133',
        b'134': '134',
        b'135': '135',
        b'136': '136',
        b'137': '137',
        b'138': '138',
        b'139': '139',
        b'140': '140',
        b'141': '141',
        b'142': '142',
        b'143': '143',
        b'144': '144',
        b'145': '145',
        b'146': '146',
        b'147': '147',
        b'148': '148',
        b'149': '149',
        b'150': '150',
        b'151': '151',
        b'152': '152',
        b'153': '153',
        b'154': '154',
        b'155': '155',
        b'156': '156',
        b'157': '157',
        b'158': '158',
        b'159': '159',
        b'160': '160',
        b'161': '161',
        b'162': '162',
        b'163': '163',
        b'164': '164',
        b'165': '165',
        b'166': '166',
        b'167': '167',
        b'168': '168',
        b'169': '169',
        b'170': '170',
        b'171': '171',
        b'172': '172',
        b'173': '173',
        b'174': '174',
        b'175': '175',
        b'176': '176',
        b'177': '177',
        b'178': '178',
        b'179': '179',
        b'180': '180',
        b'181': '181',
        b'182': '182',
        b'183': '183',
        b'184': '184',
        b'185': '185',
        b'186': '186',
        b'187': '187',
        b'188': '188',
        b'189': '189',
        b'190': '190',
        b'191': '191'
    }
