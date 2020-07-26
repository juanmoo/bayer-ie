import os
import json
from spacy.lang.en.stop_words import STOP_WORDS

path = os.getcwd()
directory = "/Desktop/Spring2020UROP"
data_type = "VendorEMAforMIT"


words = {}

review_files_xls = os.listdir(path + directory + "/reviews-xls/"+data_type)

for file in review_files_xls:
    with open(path + directory + "/reviews-xls/"+data_type+"/"+file) as f:
        data = json.load(f)
        for paragraph in data:
            word_list = paragraph.split()
            for w in word_list:
                words[w] = 1 if w not in words else words[w]+1
                
N = 5 
words_copy = list(words.keys())              
for k in words_copy:
    if words[k] <= N or k in STOP_WORDS:
        del words[k]

out_file = path + directory + "/scripts/bag-of-words.json"
out_file.write(json.dumps(words.keys()), indent=4)       
      
              