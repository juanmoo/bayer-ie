'''
This file implements a command line interface to do the following tasks:
    1. Segment EMA PDFs and FDA XMLs documents.
    2. Train linear models from annotated segmented EMA/FDA documents.
    3. Utilize trained linear models to make predictions on segmented EMA/FDA documents.
'''
from pdf_parser import process_documents
from utils import parsed_to_df
import os
import sys
import json
import argparse
import pandas as pd


def segment_ema(pdfs_dir, output_dir, pool_workers=1, separate_documents=False):
    dict_data = process_documents(pdfs_dir, pool_workers=pool_workers)
    df = parsed_to_df(dict_data)
    df['label'] = ''
    df['significant'] = ''
    if separate_documents:
        for fname in pd.unique(df['file']):
            doc_df = df.loc[df['file'] == fname].to_excel(
                os.path.join(output_dir, fname + '.xlsx'))
    else:
        df.to_excel(os.path.join(output_dir, 'data.xlsx'))


def segment_fda(something):
    pass


def segment(data_dir, output_dir, source, pool_workers=1, separate_documents=False):
    if source.lower() == 'ema':
        segment_ema(data_dir, output_dir, pool_workers=pool_workers,
                    separate_documents=separate_documents)
    else:
        raise Exception('Unknown data source {}'.format(source))


if __name__ == '__main__':
    # Parse Command Line Args #
    parser = argparse.ArgumentParser(prog='<script>')
    subparsers = parser.add_subparsers(help='sub-command help')

    # Segment document parser #
    parser_segment = subparsers.add_parser(
        'segment', help='segment help')
    parser_segment.add_argument(
        'source', type=str, help='Data source (EMA or FDA)')
    parser_segment.add_argument('dir', type=str, help='Path to pdfs')
    parser_segment.add_argument(
        'output_dir', type=str, help='Path to desired output file.')
    parser_segment.add_argument(
        '--pool-workers', type=int, default=1, help='Number of pool workers to be used.')
    parser_segment.add_argument('--separate-documents', action='store_true', default=False,
                                help='Separate segmentation in a per-document basis.')

    def segment_cli(args):
        print(args)
        segment(args.dir, args.output_dir, args.source,
                pool_workers=args.pool_workers, separate_documents=args.separate_documents)
    parser_segment.set_defaults(func=segment_cli)

    # Parse and execute
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    if len(argv) > 0:
        args.func(args)
    else:
        parser.print_help()
