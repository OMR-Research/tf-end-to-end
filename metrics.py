import numpy as np
from tabulate import tabulate
from sklearn.metrics import f1_score, jaccard_score

class Metrics:
    def __init__(self):
        self.jaccard_index = 0
        self.f1_score = 0

    def compute_from_semantics(self, predicted_semantics, expected_semantics):
        pred = np.array(predicted_semantics)
        true = np.array(expected_semantics)
        self.jaccard_index = jaccard_score(pred, true, average='micro')
        self.f1_score = f1_score(pred, true, average='micro')

    def print_table(self):
        print(tabulate([['Jaccard Index', self.jaccard_index],
                        ['F1-Score', self.f1_score]], headers=['Metric name', 'Metric value'], tablefmt='orgtbl'))
