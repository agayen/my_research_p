import os
import glob
import pandas as pd
import numpy as np

Myfiles = {}
filname = {}
path1 = 'data_set'
dir_list = os.listdir(path1)

for fol in dir_list:
    Myfiles[fol] = [I for I in glob.glob(f"{path1}/{fol}/*.txt")]
    
import re


def get_file_id(path):
  # Extract numeric part using regular expression
  match = re.search(r'/(\d+)\.txt$', path)

  if match:
      numeric_part = match.group(1)
      return numeric_part
  else:
      print(f"Numeric part not found. -> {path}")
      return'1234'
    
data_obj = {}

for folder in Myfiles:
  for file_path in Myfiles[folder]:
    with open(file_path,'rb') as fin:
      text_data = fin.read().decode('utf-8', 'ignore')
      file_id = get_file_id(file_path)
      data_obj[file_id] ={
          'doc_type': folder,
          'doc_text': text_data,
          'doc_id': file_id
      }
      
import json

# Specify the file path where you want to save the JSON file
file_path = "data_set.json"

# Save the dictionary as a JSON file
with open(file_path, 'w') as json_file:
    json.dump(data_obj, json_file)

print(f'The dictionary has been saved as {file_path}')