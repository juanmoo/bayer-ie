'''
Utility functions to be used in Bayer Project
'''

import pandas as pd
import os, sys, json, pickle
from fuzzywuzzy import fuzz

def parse_spreadsheet(path):
    '''
    Parses spreadsheet located at given path and outputs a dictionary. This parser assumes the
    following spreadsheet format:

    Row 1: Title Row. The ith entry in this row will correspond to the title of the data located
    in the ith column.

    Row 2-L: Data Row

    Returns: Dictionary keyed by filenames containing two lists of equal length with names
    "paragraphs" and "labels". The list "paragraphs" contains text entries and the "labels"
    entries contain a list of lists of labels.
    '''

    file_path = os.path.normpath(path)

    if not os.path.isfile(file_path):
        raise Exception('Unable to find file at %s.'%file_path)

    data = pd.read_excel(file_path, sheet_name=1)

    name_list = data['Link to label (or filename if using files sent by file transfer)']
    text_list = data['Broad Concept Paragraph']
    labels_list = data['Broad concept']

    output = dict()

    for name, text, label in zip(name_list, text_list, labels_list):
        if type(name) == str and '.pdf' in name:
            n = os.path.basename(name).split('.pdf')[0]
            if n not in output:
                output[n] = {
                             'texts': [],
                             'labels': []
                            }

            output[n]['texts'].append(text)
            output[n]['labels'].append(label.strip())


    return output

# Match raw JSON-styled data to annotations

def contains_test(piece, whole, exact_match=False):
    # Fuzzy matching tests to see if the max
    # similarity substring of size len(piece)
    # has a partial ratio > threshold
    if exact_match:
        return piece in whole

    threshold = 95
    return fuzz.partial_ratio(piece, whole) >= threshold

def match_labels(raw_data, annotations, exact_match=False):
    labeled_raw_documents = {}

    for parsed_doc_name in raw_data:
        parsed_doc = raw_data[parsed_doc_name]
        label_doc = annotations[parsed_doc_name] #label doc w/ same name
        
        matchings = [] # el: [parsed paragraph, label_paragraph_id, label]
        other = []
        
        paragraphs = []
        labels = []
        tags = []
        
        for parsed_p, tag in zip(parsed_doc['element_text'], parsed_doc['element_tag']):
            found = False
            for i, label_p in enumerate(label_doc['texts']):
                if contains_test(parsed_p, label_p, exact_match=exact_match):
                    found = True
                    matchings.append([parsed_p, i, label_doc['labels'][i]])
                    paragraphs.append(parsed_p)
                    labels.append(label_doc['labels'][i])
                    tags.append(tag)
                    break
            if not found:
                other.append(parsed_p)
                
                paragraphs.append(parsed_p)
                labels.append('other')
                tags.append(tag)
                
        
        labeled_raw_documents[parsed_doc_name] = {
            'matches': matchings,
            'other': other,
            'paragraphs': paragraphs,
            'labels': labels,
            'tags': tags
        }       
    
    return labeled_raw_documents

# Save / Load values from checkpoint file
def save_value(key, val, path=None):
    with open(path, 'rb') as f:
        try:
            saved_env = pickle.load(f)
            assert(type(saved_env) == dict)
        except:
            saved_env = dict()
            
    saved_env[key] = val
    
    with open(path, 'wb') as f:
        f.write(pickle.dumps(saved_env))
    
def load_value(key, path=None):
    
    with open(path, 'rb') as f:
        try:
            saved_env = pickle.load(f)
            ans = saved_env[key]
        except:
            ans = None
        
        return ans

def load_env_keys(path):
    with open(path, 'rb') as f:
        try:
            saved_env = pickle.load(f)
        except:
            saved_env = dict()
        
        return saved_env.keys()



if __name__ == '__main__':

    p = "/data/rsg/nlp/fake_proj/__temp__juanmoo__/bayer/VendorEMAforMIT/annotations.xmls"
    parse_spreadsheet(p)



