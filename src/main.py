#! /usr/bin/env python

'''
Driver Module
'''

import os, pickle, csv
from utils import *
import pandas as pd
from training import train_model
from linear_model import svm_predict
from pdf_parser import process_documents

'''
Use pre-built model to extract sections from set of PDFs.

model: Dictionary mapping labels to trained linear classifiers or path
to file containing serialized version.

label: String representing label to be extracted.
'''
def extractSections(model, data, label, output_path, checkpoints_dir=None, **kwargs):

    if checkpoints_dir is not None and not os.path.isdir(checkpoints_dir):
        raise Exception('Check-point directory at %s not found.'%checkpoints_dir)

    # Using pre-saved model
    if type(model) == str:
        path = os.path.realpath(model)
        if not os.path.isfile(path):
            raise Exception('Unable to find model at %s'%path)
        
        with open(path, 'rb') as f:
            try:
                model = pickle.load(f)
                assert(type(model) == dict)
            except:
                raise Exception('Unable to load model from file at %s'%path)
        
    # Using PDF(s) as input
    if type(data) == str and not data.lower().endswith('.json'):
        path = os.path.realpath(data)
        if not os.path.exists(path):
            raise Exception('Unable to find data in %s'%s)

        if checkpoints_dir is not None:
            parsed_docs_path = os.path.join(checkpoints_dir, 'parsed_documents.json')
        data = process_documents(source_path, output_path=parsed_docs_path, **kwargs)
        data = parsed_to_df(data)

    # Use pre-parsed JSON input
    elif type(data) == str and data.lower().endswith('.json'):
        path = os.path.realpath(data)
        data = load_parsed_file(path)

    else:
        raise Exception('Unable to load data from %s'%data)

    tok_data = pd.DataFrame(data)
    tok_data = tokenize_matches(tok_data)

    model = model.get(label, None)
    if not model:
        return np.array([False] * len(data)).reshape(-1)

    bool_locs = svm_predict(tok_data, model)
    res = data.loc[bool_locs == 1][['doc_name', 'section', 'subsection', 'header', 'subheader', 'text']]

    output = pd.DataFrame(columns=['document', 'label', 'text'])

    old_name = None
    section = None
    subsection = None
    header = None
    subheader = None
    cell_text = None

    for i in range(len(list(res.index))):
        row = res.iloc[i]

        if cell_text is None:
            section = row['section']
            subsection = row['subsection']
            header = row['header']
            subheader = row['subheader']
            cell_text = row['text']
        
        elif (row.name - 1 == old_name) and (subheader == row['subheader']):
            cell_text += '\n' + row['text']

        elif i == list(res.index)[-1]:
            # Add finished row #
            cell = section + '\n'
            cell += subsection + '\n'
            cell += header + '\n'
            cell += subheader + '\n'
            cell += cell_text

        else:
            # Add finished row #
            cell = section + '\n'
            cell += subsection + '\n'
            cell += header + '\n'
            cell += subheader + '\n'
            cell += cell_text

            row_dict = {
                'document': row['doc_name'],
                'label': label,
                'text': cell
            }
            output = output.append(row_dict, ignore_index=True)

            # Start new row #
            section = row['section']
            subsection = row['subsection']
            header = row['header']
            subheader = row['subheader']
            cell_text = row['text']
        
        old_name = row.name


    output.to_excel(output_path)
    return output



