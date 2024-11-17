import json
import os
import tkinter as tk
from tkinter import filedialog

HOME = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HOME,'config.json')

# Function to update JSON fields
def update_json_field(key):
    file_path = filedialog.askopenfilename(title="Select a file")
    if file_path:
        config_data[key] = file_path
        with open(CONFIG_PATH, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)

if __name__ == '__main__':
    # Load the JSON config file
    with open(CONFIG_PATH, 'r') as config_file:
        config_data = json.load(config_file)

    # Create a basic GUI window
    window = tk.Tk()
    window.title("JSON Config Updater")

    # Create buttons for each field in the JSON
    for key in config_data:
        button = tk.Button(window, text=f"Update {key}", command=lambda k=key: update_json_field(k))
        button.pack()

    window.mainloop()