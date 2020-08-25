from util.suit_arff_data import get_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

SEED = 5


def main():
    arff_file = '../../../datasets/kylberg-texture-dataset/data/Gabor.arff'
    data = get_data(arff_file)
    np.random.seed(SEED)
    # Set x and y
    x = data.loc[:, data.columns != 'classes']
    y = data['classes']
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, stratify=y)
    print(f"Train - {len(train_x)}")
    print(f"Test - {len(test_x)}")

    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    predict = model.predict(test_x)

    accuracy = accuracy_score(test_y, predict)
    print(f"Accuracy = {accuracy:.2%}")

    report = classification_report(test_y, predict)
    print(f"\n\nReport \n\n {report}")

    cmatrix = confusion_matrix(test_y, predict)
    confusion_matrix_display = ConfusionMatrixDisplay(cmatrix)
    confusion_matrix_display.plot()
    plt.show()


if __name__:
    main()
