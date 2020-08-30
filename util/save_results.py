import matplotlib.pyplot as plt
from pathlib import Path


def save_results(content, diretory_path):
    Path(diretory_path).mkdir(parents=True, exist_ok=True)

    file = open(diretory_path + '/results.txt', mode='w')
    file.write(content)
    file.close()


def save_plot(confusion_matrix_display, path_results):
    confusion_matrix_display.plot()
    plt.savefig(path_results + '/results.png')
