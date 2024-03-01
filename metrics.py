from tabulate import tabulate

class Metrics:
    def __init__(self):
        self.jaccard_index = 0
        self.f1_score = 0

    def compute_from_semantics(self, predicted_semantics, expected_semantics):
        # todo
        self.jaccard_index = 0
        self.f1_score = 0

    def print_table(self):
        print(tabulate([['Jaccard Index', self.jaccard_index],
                        ['F1-Score', self.f1_score]], headers=['Metric name', 'Metric value'], tablefmt='orgtbl'))
