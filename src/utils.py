'''
Utility functions to be used in Bayer Project
'''

import pandas as pd
import os, sys, json, pickle

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



