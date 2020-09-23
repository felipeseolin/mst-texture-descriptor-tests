from util.suit_arff_data import get_data
from util.save_results import save_results, save_plot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

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

    base_path_results = '/mnt/5022A63622A620C8/TCC/tests/results/'
    path_results = None
    result_header = None

    def test_model(self):
        print('=================== STARTED ===================\n\n')
        # Set the SEED for the entire model
        np.random.seed(self.SEED)
        # Get data and format
        data = get_data(self.get_data_file_path(), is_arff=self.is_arff)
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
        report = classification_report(test_y, predicted)
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
                                             f"Report: \n\n {report}"
        complete_path = self.get_result_path()
        save_results(content, complete_path)
        save_plot(confusion_matrix_display, complete_path)

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

            return header + " ======"

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
