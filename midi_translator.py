import os

from midi2audio import FluidSynth

from utils import Delegate


class MidiTranslator:
    def __init__(self, sheet, conversor_path="./converter.sh"):
        self.sheet = sheet
        self.delegate = Delegate(conversor_path)

    def translate(self, output_file_name, no_temp_files=True):
        semantic_file_path = "temp.semantic"
        self.sheet.write_to_file(semantic_file_path)
        self.delegate.run(semantic_file_path, output_file_name)

        if no_temp_files and os.path.exists(semantic_file_path):
            # Delete the semantic temporary file
            os.remove(semantic_file_path)


class MidiPlayer:
    def __init__(self, midi_path):
        self.midi_path = midi_path

    def to_audio_file(self, output_file_name="output.flac", delete_midi=True):
        # using the default sound font in 44100 Hz sample rate
        fs = FluidSynth()
        fs.midi_to_audio(self.midi_path, output_file_name)

        if delete_midi and os.path.exists(self.midi_path):
            # Delete the semantic temporary file
            os.remove(self.midi_path)
