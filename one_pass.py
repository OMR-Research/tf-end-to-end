from ctc_predict import CTC
from sheet import EncodedSheet


def one_pass_sheet(image_path):
    sheet = EncodedSheet()
    model = CTC()
    sheet.add_from_predictions(model.predict(image_path))
    return sheet


def one_pass_metric(image_path, ground_truth_path):
    sheet = one_pass_sheet(image_path)
    true_sheet = EncodedSheet()
    true_sheet.add_from_semantic_file(ground_truth_path)
    return sheet.compare(true_sheet)


if __name__ == '__main__':
    image_path = "./data/input/test1.png"
    one_pass_sheet(image_path).write_to_file("./output/test1.semantic", "index")

    metric = one_pass_metric(image_path, "./output/test1.semantic")
    metric.print_table()