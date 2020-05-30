'''
Training and storing a model.
''' 

import pickle, json, os
import pandas as pd
import numpy as np
from utils import *
from linear_model import svm_train, svm_test
from pdf_parser import process_documents 

'''
Trains linear model according to the PDF files stored in 'pdfs_path' and the annotations
given in the spreadsheet in 'annotations_path'. A serialized version of the model is stored
in output_file.
'''
def train_model(pdfs_path, annotations_path, output_dir, pool_workers=1, \
                parsed_docs_path=None, **args):

    # Process PDF Documents or load precomputed
    if not parsed_docs_path:
        parsed_docs_path = os.path.join(output_dir, 'parsed_docs.json')
        parsed = process_documents(pdfs_path, output_path=parsed_docs_path, pool_workers=pool_workers)
        data = parsed_to_df(parsed)
    else:
        data = load_parsed_file(parsed_docs_path)

    # Load Annotations from spreadsheet
    annotations = parse_spreadsheet(annotations_path)
    rationales = parse_rationale(annotations_path)

    # Match Labels to parsed data
    labels = match_labels(data, annotations, **args)

    # Pre-process matches
    data = tokenize_matches(data)

    # Train and store a model for each label
    models = dict()

    for l in labels:
        try:
            model_l = svm_train(data, l, rationales=rationales.get(l, None))
            models[l] = model_l
        except:
            models[l] = None
    
    # Save serialized models
    models_file_path = os.path.join(output_dir, 'models.sav')
    with open(models_file_path, 'wb') as f:
        f.write(pickle.dumps(models))

    return models


if __name__ == '__main__':
    pdfs_path = None
    parsed_docs_path = '/scratch/juanmoo1/bayer/VendorEMAforMIT/Labels/parsed.json'
    annotations_path = '/scratch/juanmoo1/bayer/VendorEMAforMIT/annotations.xlsx'
    output_dir = '/scratch/juanmoo1/shared/'

    models = train_model(pdfs_path, annotations_path, output_dir, pool_workers=16, \
                parsed_docs_path=parsed_docs_path, exact_match=True)