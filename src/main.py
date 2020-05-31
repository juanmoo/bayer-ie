#! /usr/bin/env python

'''
Driver Module
'''

import argparse
import os, pickle, csv, re
from utils import *
import pandas as pd
from training import train_model
from linear_model import svm_predict
from pdf_parser import process_documents

'''
Use pre-built model to extract sections from set of PDFs.

model: Dictionary mapping labels to trained linear classifiers or path
to file containing serialized version.
'''
def extractSections(models, data, output_path, checkpoints_dir=None, **kwargs):

    if checkpoints_dir is not None and not os.path.isdir(checkpoints_dir):
        raise Exception('Check-point directory at %s not found.'%checkpoints_dir)

    # Using pre-saved model
    if type(models) == str:
        path = os.path.realpath(models)
        if not os.path.isfile(path):
            raise Exception('Unable to find models at %s'%path)
        
        with open(path, 'rb') as f:
            try:
                models = pickle.load(f)
                assert(type(models) == dict)
            except:
                raise Exception('Unable to load models from file at %s'%path)
        
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

    output = pd.DataFrame(columns=['document', 'label', 'text'])

    for label in models.keys():
        model = models.get(label, None)
        if not model:
            continue
            # return np.array([False] * len(data)).reshape(-1)

        bool_locs = svm_predict(tok_data, model)
        res = data.loc[bool_locs == 1][['doc_name', 'section', 'subsection', 'header', 'subheader', 'text']]


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

                cell = re.sub('\n{2,}', '\r\n', cell)

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


if __name__ == '__main__':
    # Parse Command Line Args #
    parser = argparse.ArgumentParser(prog='<script>')
    subparsers = parser.add_subparsers(help='sub-command help')

    # Extract Sections Parser
    parser_extract_sec = subparsers.add_parser('extractSections', help='extract sections help')
    parser_extract_sec.add_argument('models', type=str, help='Path to serialized trained models.')
    parser_extract_sec.add_argument('data', type=str, help='Path to pdfs or json data.')
    parser_extract_sec.add_argument('output_path', type=str, help='Output directory.')
    parser_extract_sec.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory.')
    parser_extract_sec.add_argument('--pool-workers', type=int, default=1, help='Number of pool workers to be used.')
    def extract_cli(args):
        extractSections(args.models, args.data, args.output_path, args.checkpoint_dir, pool_workers=args.pool_workers)
    parser_extract_sec.set_defaults(func=extract_cli)

    # Extract Significant Findings Parser
    # TODO

    # Train Model Parser #
    parser_train_model = subparsers.add_parser('trainModel', help='Train model')
    parser_train_model.add_argument('data', type=str, help='Path to pdfs or json data.')
    parser_train_model.add_argument('annotations', type=str, help='Path to spreadsheet with annotations.')
    parser_train_model.add_argument('--output-dir', type=str, help='Path to output directory.')
    parser_train_model.add_argument('--pool-workers', type=int, default=1, help='Number of pool workers to be used.')
    def train_cli(args):
        model = train_model(args.data, args.annotations, args.output_dir, pool_workers=args.pool_workers)
    parser_train_model.set_defaults(func=train_cli)

    # Parse and execute
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args.func(args)