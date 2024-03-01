class EncodedSheet:
    def __init__(self, vocabulary_file_path="./data/vocabulary_semantic.txt"):
        int2word = {}
        with open(vocabulary_file_path, 'r') as dict_file:
            for idx, word in enumerate(dict_file):
                int2word[idx] = word.strip()

        self.int2word = int2word
        self.output_symbols = []

    def add_from_predictions(self, predictions):
        self.output_symbols = []
        for symbol_index in predictions:
            self.output_symbols.append(self.int2word[symbol_index])

    def print_symbols(self):
        print(self.output_symbols)

    def write_to_file(self, filename):
        with open(filename, 'w') as file:
            for string in self.output_symbols:
                file.write(string + '\t')  # Separating strings by "\t"
            file.write('\n')  # Adding a newline at the end
