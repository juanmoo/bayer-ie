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
            label = clean_label(label)
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
def match_labels_div(raw_data, annotations, exact_match=False, minimum_paragraph_length=10):
    labeled_raw_documents = {}

    for parsed_doc_name in tqdm(raw_data):
        parsed_doc = raw_data[parsed_doc_name]
        label_doc = annotations[parsed_doc_name] 
        
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

            lb_list = []
            for i, label_p in enumerate(label_doc['texts']):
                if contains_test(div_paragraphs, label_p, exact_match=exact_match, minimum_paragraph_length=minimum_paragraph_length): #Match
                    lb = clean_label(label_doc['labels'][i])
                    lb_list.append(lb)

            if len(lb_list) > 0:
                paragraphs.append(parsed_p)
                labels.append(tuple(lb_list))
                head1.append(e['head'])
                head2.append(cur_sec)

            else:
                paragraphs.append(parsed_p)
                lb_list.append('other')
                labels.append(tuple(lb_list))
                head1.append(e['head'])
                head2.append(cur_sec)

        
        labeled_raw_documents[parsed_doc_name] = {
            'texts': paragraphs,
            'labels': labels,
            'head1': head1,
            'head2': head2
        }   
    
    return labeled_raw_documents

def match_labels_p(raw_data, annotations, exact_match=False, minimum_paragraph_length=10):
    labeled_raw_documents = {}

    for parsed_doc_name in tqdm(raw_data):
        parsed_doc = raw_data[parsed_doc_name]
        label_doc = annotations[parsed_doc_name] 
        
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

            for p in div_paragraphs:
                if len(p.split()) >= minimum_paragraph_length:
                    lb_list = []
                    for i, label_p in enumerate(label_doc['texts']):
                        if contains_test([p], label_p, exact_match=exact_match): #Match
                            lb = clean_label(label_doc['labels'][i])
                            lb_list.append(lb)

                    if len(lb_list) > 0:
                        paragraphs.append(p)
                        labels.append(tuple(lb_list))
                        head1.append(e['head'])
                        head2.append(cur_sec)

                    else:
                        paragraphs.append(p)
                        lb_list.append('other')
                        labels.append(tuple(lb_list))
                        head1.append(e['head'])
                        head2.append(cur_sec)

        
        labeled_raw_documents[parsed_doc_name] = {
            'texts': paragraphs,
            'labels': labels,
            'head1': head1,
            'head2': head2
        }   
    
    return labeled_raw_documents


def clean_matches(matches):
    processed_document_list = []
    
    for doc_name in matches:
        texts = [tokenize_string(raw) for raw in matches[doc_name]['texts']]
        labels = matches[doc_name]['labels']
        head1 = [tokenize_string(raw) for raw in matches[doc_name]['head1']]
        head2 = [tokenize_string(raw) for raw in matches[doc_name]['head2']]
    
        for i in range(len(texts)):
            processed_document_list.append([doc_name, head1[i], head2[i], labels[i], texts[i]])


    return pd.DataFrame(processed_document_list, columns=['document', 'head1', 'head2', 'label', 'text'])

########## String Utilities ##########

def contains_test(pieces, whole, exact_match=False, minimum_paragraph_length=10):
    # Fuzzy matching tests to see if the max
    # similarity substring of size len(piece)
    # has a partial ratio > threshold
    if exact_match:
        return any([(len(piece.split())>=minimum_paragraph_length) and (piece in whole) for piece in pieces])

    threshold = 95
    return any([fuzz.partial_ratio(piece, whole) >= threshold for piece in pieces if len(piece.split())>=minimum_paragraph_length])

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



