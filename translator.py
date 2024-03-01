import os
import subprocess


class SheetTranslator:
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


class Delegate:
    def __init__(self, path):
        self.path = path

    def run(self, input_file_path, output_file_path):
        try:
            subprocess.run(["sh", self.path, input_file_path, output_file_path])
            print("Shell script executed successfully")
        except subprocess.CalledProcessError as e:
            print("Error executing shell script:", e)
