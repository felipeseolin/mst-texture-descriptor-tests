from util.suit_arff_data import get_data
from util.save_results import save_results, save_plot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

SEED = 5


def main():
    arff_file = '../../../datasets/kylberg-texture-dataset/data/Tamura.arff'
    data = get_data(arff_file)
    np.random.seed(SEED)
    # Set x and y
    x = data.loc[:, data.columns != 'classes']
    y = data['classes']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, stratify=y)

    model = DecisionTreeClassifier()
    model.fit(train_x, train_y)
    predict = model.predict(test_x)

    accuracy = accuracy_score(test_y, predict)

    report = classification_report(test_y, predict)

    cmatrix = confusion_matrix(test_y, predict)
    confusion_matrix_display = ConfusionMatrixDisplay(cmatrix)

    base_path_results = '/mnt/5022A63622A620C8/TCC/tests/results/'
    path_results = base_path_results + 'tamura/kylberg_texture_dataset/decision_tree'

    content = f"====== TAMURA - DECISION TREE- KYLBERG TEXTURE DATASET====== \n\n" \
              f"{len(x)} images\n\n" \
              f"Train size: {len(train_x)}\n" \
              f"Test size: {len(test_x)}\n\n" \
              f"Accuracy: {accuracy:.2%}\n\n" \
              f"Report: \n\n {report}"

    save_results(content, path_results)
    save_plot(confusion_matrix_display, path_results)

    print('OPEN FILES RESULTS IN: ' + path_results)


if __name__:
    main()
