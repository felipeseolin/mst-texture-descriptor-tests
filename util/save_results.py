from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_results(content, diretory_path):
    Path(diretory_path).mkdir(parents=True, exist_ok=True)

    file = open(diretory_path + '/results.txt', mode='w')
    file.write(content)
    file.close()


def save_plot(confusion_matrix_display, path_results):
    confusion_matrix_display.plot()
    plt.savefig(path_results + '/results.png')


def save_csv(report, path_results):
    df = pd.DataFrame.from_dict(report)
    df.to_csv(path_results + '/report.csv', index=False, header=True, encoding='utf-8')
