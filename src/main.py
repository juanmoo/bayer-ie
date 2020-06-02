#! /usr/bin/env python

'''
Driver Module
'''

import argparse
import os, pickle, csv, re
from utils import *
import pandas as pd
from training import train_model
from extractor import extractSignificantFindings
from linear_model import svm_predict, svm_cross_validate
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

def cross_validate(data_path, annotations_path, output_dir, num_folds, pool_workers=1, **args):

    # Process PDF Documents or load precomputed
    print('Loading input data ... \n')
    if os.path.isdir(data_path): # From PDFs
        parsed_docs_path = os.path.join(output_dir, 'parsed_docs.json')
        parsed = process_documents(data_path, output_path=parsed_docs_path, pool_workers=pool_workers)
        data = parsed_to_df(parsed)
    elif os.path.isfile(data_path) and data_path.lower().endswith('.json'): # From JSON
        data = load_parsed_file(data_path)
    else:
        raise Exception('Unable to load data from %s'%data_path)

    # Load Annotations from spreadsheet
    print('Loading annotations ... \n')
    annotations = parse_spreadsheet(annotations_path)
    rationales = parse_rationale(annotations_path)

    # Match Labels to parsed data
    print('Matching annotations to data ... \n')
    labels = match_labels(data, annotations, **args)

    # Pre-process matches
    data = tokenize_matches(data)

    # Cross Validate and Report results
    print('Running cross-validation ... \n')
    results = svm_cross_validate(data, labels, num_folds, rationales=rationales)

    summary_path = os.path.join(output_dir, 'results.txt')
    with open(summary_path, 'w') as f:
        for l in labels:
            precisions = results[l]['precisions']
            recalls = results[l]['recalls']
            f1s = results[l]['f1s']

            if l == 'warning' or l == 'warning':
                print('precisions:', precisions)
                print('recalls:', recalls)
                print('f1s:', f1s)
            
            # avg_p = sum(precisions)/len(precisions)
            # avg_r = sum(recalls)/len(recalls)
            # avg_f1 = sum(f1s)/len(f1s)

            summary = ''
            summary += '-' * 2 + ' Label: ' + l  + ' ' + '-' * 20 + '\n'
            # summary += '\t precision: ' + str(avg_p) + '\n'
            # summary += '\t recall: ' + str(avg_r) + '\n'
            # summary += '\t f1: ' + str(avg_f1) + '\n'

            summary += '\t precision: ' + str(precisions) + '\n'
            summary += '\t recall: ' + str(recalls) + '\n'
            summary += '\t f1: ' + str(f1s) + '\n'

            print(summary)
            f.write(summary)

if __name__ == '__main__':
    # Parse Command Line Args #
    parser = argparse.ArgumentParser(prog='<script>')
    subparsers = parser.add_subparsers(help='sub-command help')

    # Extract Sections Parser
    parser_extract_sec = subparsers.add_parser('extractSections', help='extract sections help')
    parser_extract_sec.add_argument('models', type=str, help='Path to serialized trained models.')
    parser_extract_sec.add_argument('data', type=str, help='Path to pdfs or json data.')
    parser_extract_sec.add_argument('output_path', type=str, help='Path to desired output file.')
    parser_extract_sec.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory.')
    parser_extract_sec.add_argument('--pool-workers', type=int, default=1, help='Number of pool workers to be used.')
    parser_extract_sec.add_argument('--exact-match', type=bool, default=False,\
                            help='Choose whether or not to use fuzzy-mathing to match labels.')

    def extract_cli(args):
        extractSections(args.models, args.data, args.output_path, args.checkpoint_dir,\
        pool_workers=args.pool_workers, exact_match=args.exact_match)
    parser_extract_sec.set_defaults(func=extract_cli)

    # Extract Significant Findings Parser
    parser_significant = subparsers.add_parser('extractSignificant', help='Extract sections of significant findings.')
    parser_significant.add_argument('data', type=str, help='Path to pdfs or json data.')
    parser_significant.add_argument('output_path', type=str, help='Path to desired output file.')
    parser_significant.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory.')
    parser_significant.add_argument('--pool-workers', type=int, default=1, help='Number of pool workers to be used.')

    def extract_significant_cli(args):
        extractSignificantFindings(args.data, args.output_path, output_dir=args.checkpoint_dir,\
        pool_workers=args.pool_workers)
    
    parser_significant.set_defaults(func=extract_significant_cli)

    # Train Model Parser #
    parser_train_model = subparsers.add_parser('trainModel', help='Train model')
    parser_train_model.add_argument('data', type=str, help='Path to pdfs or json data.')
    parser_train_model.add_argument('annotations', type=str, help='Path to spreadsheet with annotations.')
    parser_train_model.add_argument('output_dir', type=str, help='Path to output directory.')
    parser_train_model.add_argument('--pool-workers', type=int, default=1, help='Number of pool workers to be used.')
    parser_train_model.add_argument('--exact-match', type=bool, default=False,\
                            help='Choose whether or not to use fuzzy-mathing to match labels.')
    def train_cli(args):
        model = train_model(args.data, args.annotations, args.output_dir, pool_workers=args.pool_workers,\
        exact_match=args.exact_match)
    parser_train_model.set_defaults(func=train_cli)

    # Cross-validator Parser #
    parser_cross_validate = subparsers.add_parser('crossValidate', help='Cross-validation on given dataset.')
    parser_cross_validate.add_argument('data', type=str, help='Path to pdfs or json data.')
    parser_cross_validate.add_argument('annotations', type=str, help='Path to spreadsheet with annotations.')
    parser_cross_validate.add_argument('output_dir', type=str, help='Path to output directory.')
    parser_cross_validate.add_argument('num_folds', type=int, help='Number of folds to use in cross validations.')
    parser_cross_validate.add_argument('--pool-workers', type=int, default=1, help='Number of pool workers to be used.')
    parser_cross_validate.add_argument('--exact-match', type=bool, default=False,\
                            help='Choose whether or not to use fuzzy-mathing to match labels.')
    
    def cross_validate_cli(args):
        cross_validate(args.data, args.annotations, args.output_dir, args.num_folds,\
                pool_workers=args.pool_workers, exact_match=args.exact_match)
    parser_cross_validate.set_defaults(func=cross_validate_cli)

    # Parse and execute
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    if len(argv) > 0:
        args.func(args)
    else:
        parser.print_help()