'''
This file implements a command line interface to do the following tasks:
    1. Segment EMA PDFs and FDA XMLs documents.
    2. Train linear models from annotated segmented EMA/FDA documents.
    3. Utilize trained linear models to make predictions on segmented EMA/FDA documents.
'''
from pdf_parser import process_documents
from utils import parsed_to_df
from linear_model import svm_train_ema, svm_train_fda, svm_predict_ema, svm_predict_fda
from xml_parser import process_xmls
import os
import sys
import json
import pickle
import argparse
import pandas as pd
from functools import reduce
from tqdm import tqdm


## Segmentation ##


def save_separate_documents(df, output_dir):
    for fname in pd.unique(df['file']):
        doc_df = df.loc[df['file'] == fname].to_excel(
            os.path.join(output_dir, fname + '.xlsx'))


def segment_ema(pdfs_dir, output_dir, pool_workers=1, separate_documents=False):
    dict_data = process_documents(pdfs_dir, pool_workers=pool_workers)
    df = parsed_to_df(dict_data)
    df['label'] = ''
    df['significant'] = ''
    if separate_documents:
        save_separate_documents(df, output_dir)
    else:
        df.to_excel(os.path.join(output_dir, 'data.xlsx'))


def segment_fda(xmls_dir, output_dir, separate_documents=False):
    df = process_xmls(xmls_dir)
    if separate_documents:
        save_separate_documents(df, output_dir)
    else:
        df.to_excel(os.path.join(output_dir, 'data.xlsx'))


def segment(data_dir, output_dir, source, pool_workers=1, separate_documents=False):
    if source.lower() == 'ema':
        segment_ema(data_dir, output_dir, pool_workers=pool_workers,
                    separate_documents=separate_documents)
    elif source.lower() == 'fda':
        segment_fda(data_dir, output_dir,
                    separate_documents=separate_documents)
    else:
        raise Exception('Unknown data source {}'.format(source))

## Training ##


def train_ema(data, rationales, output_dir):
    labs = reduce(lambda a, b: a + b, [s.lower().split('||')
                                       for s in pd.unique(data['labels'])])
    labs = sorted(list(set([s.strip() for s in labs])))
    lmap = {l: l for l in labs if l}
    corrections = {
        'hepatic impairment': 'hepatic',
        'renal impairment': 'renal',
        'warning': 'warnings',
        'population - adult': 'populations - adult',
        'population - adolescent': 'populations - adolescent',
        'populations - neonates': 'populations - neonate',
        'populations - paediatric': 'populations - pediatric'
    }
    lmap.update(corrections)
    labs = sorted(list(set(labs[1:]) - set(corrections.keys())))

    # One-hot corrected categories
    for l in labs:
        data[l] = 0

    for j, row in enumerate(data.iloc):
        for l in lmap:
            if l in row['labels'].lower():
                data.iloc[j, data.columns.get_loc(lmap[l])] = 1

    # Add 1-step significance labels
    significant_labs = ['hepatic', 'renal', 'pregnancy']
    for sl in significant_labs:
        data['significant-{}'.format(sl)
             ] = (data['significant'] == 'X') * data[sl]
    labs += ['significant-{}'.format(sl) for sl in significant_labs]

    # Train and store a model for each label
    models = dict()

    for l in tqdm(labs):
        try:
            model_l = svm_train_ema(
                data, l, rationales=rationales.get(l, None))
            models[l] = model_l
        except:
            models[l] = None

    with open(os.path.join(output_dir, 'ema.models'), 'wb') as f:
        pickle.dump(models, f)


def train_fda(data, rationales, output_dir):
    labs = reduce(lambda a, b: a + b, [s.lower().split('||')
                                       for s in pd.unique(data['labels'])])
    labs = sorted(list(set([s.strip() for s in labs])))
    labs.append('warnings')
    lmap = {l: l for l in labs if l}
    corrections = {
        'hepatic impairment': 'hepatic',
        'renal impairment': 'renal',
        'warning': 'warnings'
    }
    lmap.update(corrections)
    labs = sorted(list(set(labs[1:]) - set(corrections.keys())))

    # One-hot corrected categories
    for l in labs:
        data[l] = 0
    for j, row in enumerate(data.iloc):
        for l in lmap:
            corrected = str(row['labels']).lower()
            if l in corrected:
                data.iloc[j, data.columns.get_loc(lmap[l])] = 1

    # Add 1-step significance labels
    significant_labs = ['hepatic', 'renal', 'pregnancy']
    for sl in significant_labs:
        data['significant-{}'.format(sl)
             ] = (data['significant'] == 'X') * data[sl]
    labs += ['significant-{}'.format(sl) for sl in significant_labs]

    models = dict()

    for l in tqdm(labs):
        try:
            model_l = svm_train_fda(
                data, l, rationales=rationales.get(l, None))
            models[l] = model_l
        except:
            models[l] = None

    with open(os.path.join(output_dir, 'fda.models'), 'wb') as f:
        pickle.dump(models, f)


def train(data_dir, output_dir, rationales_path, source):
    # Load Data
    files = [s for s in os.listdir(data_dir) if s.lower().endswith('.xlsx')]
    frames = [pd.read_excel(os.path.join(data_dir, f)).fillna('')
              for f in files]
    data = pd.concat(frames)
    data = data.set_index('Unnamed: 0', drop=True)
    data.index.rename('', inplace=True)
    data = data.sort_index()

    # Load Rationales
    rationales = json.load(open(rationales_path, 'r'))

    if source.lower() == 'ema':
        train_ema(data, rationales, output_dir)
    elif source.lower() == 'fda':
        train_fda(data, rationales, output_dir)
    else:
        raise Exception('Unknown data source {}'.format(source))


## Predictions ##


def predict(data_dir, models_path, output_dir, source, separate_documents=False):
    # Load Data
    files = [s for s in os.listdir(data_dir) if s.lower().endswith('.xlsx')]
    frames = [pd.read_excel(os.path.join(data_dir, f)).fillna('')
              for f in files]
    data = pd.concat(frames)
    data = data.set_index('Unnamed: 0', drop=True)
    data.index.rename('', inplace=True)
    data = data.sort_index()

    # Load Models
    with open(models_path, 'rb') as f:
        models = pickle.load(f)

    if source.lower() == 'ema':
        for l in tqdm(models):
            pred = svm_predict_ema(data, models[l])
            data['pred-' + l] = pred
    elif source.lower() == 'fda':
        for l in tqdm(models):
            pred = svm_predict_fda(data, models[l])
            data['pred-' + l] = pred
    else:
        raise Exception('Unknown data source {}'.format(source))

    if separate_documents:
        save_separate_documents(data, output_dir)
    else:
        data.to_excel(os.path.join(output_dir, 'predictions.xlsx'))


if __name__ == '__main__':
    # Parse Command Line Args #
    parser = argparse.ArgumentParser(prog='<script>')
    subparsers = parser.add_subparsers(help='sub-command help')

    # Segment document parser #
    parser_segment = subparsers.add_parser(
        'segment', help='Create segmented versions of documents.')
    parser_segment.add_argument(
        'source', type=str, help='Data source (EMA or FDA)')
    parser_segment.add_argument('dir', type=str, help='Path to pdfs')
    parser_segment.add_argument(
        'output-dir', type=str, help='Path to desired output file.')
    parser_segment.add_argument(
        '--pool-workers', type=int, default=1, help='Number of pool workers to be used.')
    parser_segment.add_argument('--separate-documents', action='store_true', default=False,
                                help='Separate segmentation in a per-document basis.')

    def segment_cli(args):
        segment(args.dir, args.output_dir, args.source,
                pool_workers=args.pool_workers, separate_documents=args.separate_documents)
    parser_segment.set_defaults(func=segment_cli)

    # Train document parser #
    parser_train = subparsers.add_parser(
        'train', help='Train model from labeled segmented files')
    parser_train.add_argument(
        'source', type=str, help='Data source (EMA or FDA)')
    parser_train.add_argument('data_dir', type=str,
                              help='Path to segmented files')
    parser_train.add_argument('rationales_path', type=str,
                              help='Path to rationales file')
    parser_train.add_argument(
        'output_dir', type=str, help='Path to desired output directory')

    def train_cli(args):
        train(args.data_dir, args.output_dir,
              args.rationales_path, args.source)
    parser_train.set_defaults(func=train_cli)

    # Predict #
    parser_predict = subparsers.add_parser(
        'predict', help='Make predictions on segmented documents.')
    parser_predict.add_argument(
        'source', type=str, help='Data source (EMA or FDA)')
    parser_predict.add_argument('data_dir', type=str,
                                help='Path to segmented files')
    parser_predict.add_argument('models_path', type=str,
                                help='Path to models')
    parser_predict.add_argument(
        'output_dir', type=str, help='Path to desired output directory')
    parser_predict.add_argument('--separate_documents', action='store_true', default=False,
                                help='Separate results in a per-document basis.')

    def predict_cli(args):
        predict(args.data_dir, args.models_path, args.output_dir,
                args.source, separate_documents=args.separate_documents)
    parser_predict.set_defaults(func=predict_cli)

    # Parse and execute
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    if len(argv) > 0:
        args.func(args)
    else:
        parser.print_help()
