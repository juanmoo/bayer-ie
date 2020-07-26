'''
Utility functions to be used in Bayer Project
'''

import pandas as pd
import os, sys, json, pickle, re, spacy
from fuzzywuzzy import fuzz
from tqdm import tqdm

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

    data = pd.read_excel(file_path, sheet_name=0)

    name_list = data['Link to label']
    text_list = data['Broad Concept Paragraph - corrected']
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
            found = False
            for i, lb in enumerate(output[n]['labels']):
                if lb == label:
                    output[n]['texts'][i] += '\n' + text
                    found = True
                    break
            if not found:
                output[n]['texts'].append(text)
                output[n]['labels'].append(label)


    return output



# Match raw JSON-styled data to annotations
def match_labels(raw_data, annotations, exact_match=False):
    labeled_raw_documents = {}

    for parsed_doc_name in tqdm(raw_data):
        try:
            parsed_doc = raw_data[parsed_doc_name]
            label_doc = annotations[parsed_doc_name.split('.pdf')[0]] 
        except:
            print("Did not work for ", parsed_doc_name)
            continue
        
        matchings = [] # el: [parsed paragraph, label, section]
        other = []
        
        paragraphs = []
        labels = []
        tags = []

        head1 = []
        # head2 only consider x.x
        head2 = []

        cur_sec = ''

        for e in parsed_doc['elements']:
            if is_section(e['head']):
                cur_sec = e['head']

            # ignore div that contains only a <head> and no other <p>
            if len(e['text']) == 1:
                continue

            div_paragraphs = e['text'][1:] # All paragraphs in <div> excluding header
            parsed_p = '\n'.join(div_paragraphs)

            lb = 'other'
            for i, label_p in enumerate(label_doc['texts']):
                if not isinstance(label_p,str):
                    #print("not string = ", label_p)
                    continue
                
                if contains_test(div_paragraphs, label_p, exact_match=exact_match): #Match
                    lb = clean_label(label_doc['labels'][i])
                    matchings.append([parsed_p, lb, cur_sec])

                    paragraphs.append(parsed_p)
                    labels.append(lb)
                    head1.append(e['head'])
                    head2.append(cur_sec)
                    # more than one label is possible, so no break

            if lb == 'other':
                other.append(parsed_p)

                paragraphs.append(parsed_p)
                labels.append(lb)
                head1.append(e['head'])
                head2.append(cur_sec)



            # paragraphs.append(parsed_p)
            # labels.append(lb)
            # head1.append(e['head'])
            # head2.append(cur_sec)
        
        
        labeled_raw_documents[parsed_doc_name] = {
            'matches': matchings,
            'other': other,
            'texts': paragraphs,
            'labels': labels,
            'head1': head1,
            'head2': head2
        }   
    
    return labeled_raw_documents

########## String Utilities ##########

def contains_test(pieces, whole, exact_match=False):
    # Fuzzy matching tests to see if the max
    # similarity substring of size len(piece)
    # has a partial ratio > threshold
    if exact_match:
        return any([piece in whole for piece in pieces])

    threshold = 95
    return any([fuzz.partial_ratio(piece, whole) >= threshold for piece in pieces])

def is_section(head):
    if not head:
        return False
    tok = head.split()[0]
    if tok.isdigit():
        return True
    if tok.replace('.','',1).isdigit():
        return True
    return False

def clean_label(lb):
    lb = lb.strip()
    if lb == 'Significant findings - pregnancy':
        return 'Significant Findings - Pregnancy'
    return lb


NLP = spacy.load('en_core_web_sm')
MAX_CHARS = 20000
def tokenize_string(comment):
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return ' '.join([x.text.lower() for x in NLP.tokenizer(comment) if x.text != " "])

############## Logistical Utilities ##############

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



