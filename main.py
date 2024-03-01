
import argparse

from ctc_predict import CTC
from translator import SheetTranslator
from sheet import EncodedSheet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
    parser.add_argument('-image', dest='image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
    args = parser.parse_args()

    sheet = EncodedSheet(args.voc_file)
    model = CTC(args.model)
    print("Processing image...")
    sheet.add_from_predictions(model.predict(args.image))
    print("Done!")
    print("Symbols found:")
    sheet.print_symbols()

    print("Converting to MIDI...")
    output_midi_path = "output.mid"
    translator = SheetTranslator(sheet)
    translator.translate(output_midi_path)
    print("Done!")
