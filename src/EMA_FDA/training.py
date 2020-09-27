'''
Training and storing a model.
''' 

import pickle, json, os
import pandas as pd
import numpy as np
from utils import *
from linear_model import svm_train, svm_test, svm_cross_validate 
from pdf_parser import process_documents
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore division by zero when calculating F1 score
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
# Ignore converge warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

'''
Trains linear model according to the PDF files stored in 'pdfs_path' and the annotations
given in the spreadsheet in 'annotations_path'. A serialized version of the model is stored
in output_file.
'''
def train_model(data_path, annotations_path, output_dir, pool_workers=1, **args):

    # Process PDF Documents or load precomputed
    if os.path.isdir(data_path): # From PDFs
        parsed_docs_path = os.path.join(output_dir, 'parsed_docs.json')
        parsed = process_documents(data_path, output_path=parsed_docs_path, pool_workers=pool_workers)
        data = parsed_to_df(parsed)
    elif os.path.isfile(data_path) and data_path.lower().endswith('.json'): # From JSON
        data = load_parsed_file(data_path)
    else:
        raise Exception('Unable to load data from %s'%data_path)

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
