from metrics import Metrics


class EncodedSheet:
    def __init__(self, vocabulary_file_path="./data/vocabulary_semantic.txt"):
        int2word = {}
        with open(vocabulary_file_path, 'r') as dict_file:
            for idx, word in enumerate(dict_file):
                int2word[idx] = word.strip()

        self.int2word = int2word
        self.output_symbols = []
        self.output_indexes = []

    def add_from_semantic_file(self, semantic_file_path):
        with open(semantic_file_path, 'r') as semantic_file:
            predictions = []
            for value in semantic_file.readlines()[0].split('\t'):
                try:
                    predictions.append(int(value))
                except ValueError:
                    continue
            self.add_from_predictions(predictions)

    def add_from_predictions(self, predictions):
        self.output_symbols = []
        for symbol_index in predictions:
            self.output_indexes.append(symbol_index)
            self.output_symbols.append(self.int2word[symbol_index])

    def compare(self, true_sheet):
        metrics = Metrics()
        metrics.compute_from_semantics(self.output_indexes, true_sheet.output_indexes)
        return metrics

    def print_symbols(self):
        print(self.output_symbols)

    def write_to_file(self, filename, output_format="symbol"):
        to_write = []
        if output_format == "symbol":
            to_write = self.output_symbols
        elif output_format == "index":
            to_write = self.output_indexes
        else:
            raise ValueError("Expected non empty output format")

        with open(filename, 'w') as file:
            for value in to_write:
                file.write(str(value) + '\t')  # Separating values by "\t"
            file.write('\n')  # Adding a newline at the end
