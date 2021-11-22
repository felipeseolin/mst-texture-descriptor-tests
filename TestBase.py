from util.suit_arff_data import get_data
from util.save_results import save_results, save_plot, save_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    precision_recall_fscore_support

import sys
import numpy as np
import classifier_constants
import dataset_constants
import descriptor_constants


class TestBase:
    SEED = 5
    file = None
    is_arff = True
    classifier = None
    dataset = None
    descriptor = None

    base_path_results = '/Users/seolin/Documents/TCC/image-texture-classification-tests/results/'
    path_results = None
    result_header = None

    def test_model(self):
        print('=================== STARTED ===================\n\n')
        # Set the SEED for the entire model
        np.random.seed(self.SEED)
        # Get data and format
        data = get_data(self.get_data_file_path(), dataset=self.dataset, is_arff=self.is_arff)
        x = data.loc[:, data.columns != 'classes']
        y = data['classes']
        # Set train and test
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, stratify=y)
        # Get model, fit and predict
        model = self.set_model()
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        # Get accuracy and report
        accuracy = accuracy_score(test_y, predicted)
        prfs_micro = precision_recall_fscore_support(test_y, predicted, average='micro')
        prfs_macro = precision_recall_fscore_support(test_y, predicted, average='macro')
        report = classification_report(test_y, predicted)
        report_csv = classification_report(test_y, predicted, output_dict=True)
        # Get confusion_matrix
        cmatrix = confusion_matrix(test_y, predicted)
        confusion_matrix_display = ConfusionMatrixDisplay(cmatrix)
        # Save results
        content = self.get_result_header() + "\n\n" \
                                             f"{len(x)} images\n\n" \
                                             f"SEED: {self.SEED}\n" \
                                             f"Train size: {len(train_x)}\n" \
                                             f"Test size: {len(test_x)}\n\n" \
                                             f"Accuracy: {accuracy:.2%}\n\n" \
                                             f"Precision Micro: {prfs_micro[0]:.2%}\n" \
                                             f"Precision Macro: {prfs_macro[0]:.2%}\n\n" \
                                             f"Recall Micro: {prfs_micro[1]:.2%}\n" \
                                             f"Recall Macro: {prfs_macro[1]:.2%}\n\n" \
                                             f"F1-Score Micro: {prfs_micro[2]:.2%}\n" \
                                             f"F1-Score Macro: {prfs_macro[2]:.2%}\n\n" \
                                             f"Report: \n\n {report}"
        complete_path = self.get_result_path()
        save_results(content, complete_path)
        save_plot(confusion_matrix_display, complete_path)
        save_csv(report_csv, complete_path)

        print('OPEN FILES RESULTS IN: ' + complete_path)
        print('\n\n=================== END ===================')
        sys.exit(0)

    def set_model(self):
        if self.classifier == classifier_constants.DECISION_TREE:
            return DecisionTreeClassifier()
        elif self.classifier == classifier_constants.KNN:
            return KNeighborsClassifier()
        elif self.classifier == classifier_constants.LINEAR_SVC:
            return LinearSVC()
        elif self.classifier == classifier_constants.SVC:
            return SVC()
        else:
            return None

    def get_data_file_path(self):
        if self.file:
            return self.file
        else:
            file_path = '../../../datasets/'
            # Dataset
            if self.dataset == dataset_constants.KYLBERG:
                file_path = file_path + 'kylberg-texture-dataset/'
            if self.dataset == dataset_constants.BRODATZ:
                file_path = file_path + 'brodatz-texture-dataset/'
            if self.dataset == dataset_constants.USPTEX:
                file_path = file_path + 'usptex-texture-dataset/'
            if self.dataset == dataset_constants.VISTEX:
                file_path = file_path + 'vistex-texture-dataset/'

            file_path = file_path + 'data/'
            # Descriptor
            if self.descriptor == descriptor_constants.HARALICK:
                file_path = file_path + 'Haralick.arff'
            elif self.descriptor == descriptor_constants.GABOR:
                file_path = file_path + 'Gabor.arff'
            elif self.descriptor == descriptor_constants.LBP:
                file_path = file_path + 'LBP.arff'
            elif self.descriptor == descriptor_constants.TAMURA:
                file_path = file_path + 'Tamura.arff'
            elif self.descriptor == descriptor_constants.MST:
                file_path = file_path + 'mst.csv'

            return file_path

    def get_result_path(self):
        if self.path_results:
            return self.path_results
        else:
            path = self.base_path_results
            # Descriptor
            if self.descriptor == descriptor_constants.HARALICK:
                path = path + 'haralick/'
            elif self.descriptor == descriptor_constants.GABOR:
                path = path + 'gabor/'
            elif self.descriptor == descriptor_constants.LBP:
                path = path + 'lbp/'
            elif self.descriptor == descriptor_constants.TAMURA:
                path = path + 'tamura/'
            elif self.descriptor == descriptor_constants.MST:
                path = path + 'mst/'
            # Dataset
            if self.dataset == dataset_constants.KYLBERG:
                path = path + 'kylberg_texture_dataset/'
            if self.dataset == dataset_constants.BRODATZ:
                path = path + 'brodatz_texture_dataset/'
            if self.dataset == dataset_constants.USPTEX:
                path = path + 'usptex_texture_dataset/'
            if self.dataset == dataset_constants.VISTEX:
                path = path + 'vistex_texture_dataset/'
            # Classifier
            if self.classifier == classifier_constants.DECISION_TREE:
                path = path + 'decision_tree'
            elif self.classifier == classifier_constants.KNN:
                path = path + 'k_nearest_neighbors'
            elif self.classifier == classifier_constants.LINEAR_SVC:
                path = path + 'linear_SVC'
            elif self.classifier == classifier_constants.SVC:
                path = path + 'SVC'

            return path

    def get_result_header(self):
        if self.result_header:
            return self.result_header
        else:
            header = '====== '
            # Descriptor
            if self.descriptor == descriptor_constants.HARALICK:
                header = header + 'HARALICK - '
            elif self.descriptor == descriptor_constants.GABOR:
                header = header + 'GABOR - '
            elif self.descriptor == descriptor_constants.LBP:
                header = header + 'LBP - '
            elif self.descriptor == descriptor_constants.TAMURA:
                header = header + 'TAMURA - '
            elif self.descriptor == descriptor_constants.MST:
                header = header + 'MST - '
            # Classifier
            if self.classifier == classifier_constants.DECISION_TREE:
                header = header + 'DECISION TREE - '
            elif self.classifier == classifier_constants.SVC:
                header = header + 'SVC - '
            elif self.classifier == classifier_constants.KNN:
                header = header + 'K_NEAREST_NEIGHBORS - '
            elif self.classifier == classifier_constants.LINEAR_SVC:
                header = header + 'LINEAR_SVC - '
            # Dataset
            if self.dataset == dataset_constants.KYLBERG:
                header = header + 'KYLBERG TEXTURE DATASET'
            if self.dataset == dataset_constants.BRODATZ:
                header = header + 'BRODATZ TEXTURE DATASET'
            if self.dataset == dataset_constants.USPTEX:
                header = header + 'USPTEX TEXTURE DATASET'
            if self.dataset == dataset_constants.VISTEX:
                header = header + 'VISTEX TEXTURE DATASET'

            return header + " ======"

    def get_classes_y(self):
        if self.dataset == dataset_constants.BRODATZ:
            return [
                'BAR', 'BRI', 'BUB', 'GRA', 'LEA', 'PIG', 'RAF', 'SAN', 'STR', 'WAT', 'WEA', 'WOL', 'WOO'
            ]
        if self.dataset == dataset_constants.VISTEX:
            return [
                'BAR', 'BRI', 'BUI', 'CLO', 'FAB', 'FLO', 'FOO', 'GRA', 'LEA', 'MET', 'MIS', 'PAI', 'SAN', 'STO', 'TER',
                'TIL', 'WAT', 'WHE', 'WOO'
            ]
        if self.dataset == dataset_constants.USPTEX:
            return [
                '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015',
                '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030',
                '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045',
                '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060',
                '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075',
                '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090',
                '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105',
                '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
                '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135',
                '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150',
                '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165',
                '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180',
                '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191'
            ]
        if self.dataset == dataset_constants.KYLBERG:
            return [
                'BL1',
                'BL2',
                'CAN',
                'CE1',
                'CE2',
                'CUS',
                'FL1',
                'FL2',
                'GRA',
                'LEN',
                'LIN',
                'OAT',
                'PEA',
                'RI1',
                'RI2',
                'RUG',
                'SAN',
                'SC1',
                'SC2',
                'SCR',
                'SE1',
                'SE2',
                'SES',
                'ST1',
                'ST2',
                'ST3',
                'STL',
                'WAL',
            ]

        return []

    def set_result_header(self, value):
        self.result_header = value

    def set_path_results(self, value):
        self.path_results = value

    def set_base_path_results(self, value):
        self.base_path_results = value

    def set_seed(self, value):
        self.SEED = value

    def set_file(self, value):
        self.file = value

    def set_is_arff(self, value):
        self.is_arff = value

    def set_decision_tree_classifier(self):
        self.classifier = classifier_constants.DECISION_TREE

    def set_knn_classifier(self):
        self.classifier = classifier_constants.KNN

    def set_linear_svc_classifier(self):
        self.classifier = classifier_constants.LINEAR_SVC

    def set_svc_classifier(self):
        self.classifier = classifier_constants.SVC

    def set_kylberg_dataset(self):
        self.dataset = dataset_constants.KYLBERG

    def set_usptex_dataset(self):
        self.dataset = dataset_constants.USPTEX

    def set_vistex_dataset(self):
        self.dataset = dataset_constants.VISTEX

    def set_brodatz_dataset(self):
        self.dataset = dataset_constants.BRODATZ

    def set_haralick_descriptor(self):
        self.descriptor = descriptor_constants.HARALICK

    def set_gabor_descriptor(self):
        self.descriptor = descriptor_constants.GABOR

    def set_lbp_descriptor(self):
        self.descriptor = descriptor_constants.LBP

    def set_mst_descriptor(self):
        self.descriptor = descriptor_constants.MST

    def set_tamura_descriptor(self):
        self.descriptor = descriptor_constants.TAMURA
