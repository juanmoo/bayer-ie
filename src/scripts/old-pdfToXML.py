#! /usr/bin/python3

'''
The purpose of this script is to transform native PDF files into their corresponding TEI XML representations using GROBID.
Given the path of the folder containing the native PDF files, it will create a folder containing the corresponding xmls.

Usage:  <path to this script> --dest <path to destination folder> --server-url <URL of GROBID server> <path to source folder> 

1. The source flag is required and corresponds to the path of the PDF documents.
2. If no dest flag is given, <source path>/xmls will be the default value of the destination folder.
'''

import os, sys
import requests, argparse
from urllib.parse import urljoin
from multiprocessing import Pool


# Parse Command Line Args #
argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('source', help='Path to folder with PDFs')
parser.add_argument('--dest', help='Path of folder where XMLs are to be placed')
parser.add_argument('--server-url', help='URL of API endpoint to process pdfs')
parser.add_argument('--pool-workers', help='Number of workers to be used to process documents simultaneously')
args = parser.parse_args(argv)

# Verify Args #
source_path = os.path.realpath(args.source)
if not os.path.isdir(source_path):
    raise Exception('Source path is invalid.')

if args.dest is None:
    # Create new folder at <source_path>/xmls and set dest path its path.
    dest_path = os.path.join(source_path, 'xmls')
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
else:
    dest_path = os.path.realpath(args.dest)
    if not os.path.isdir(dest_path):
        raise Exception('Destination path is invalid.')

if args.pool_workers is not None:
    if args.pool_workers.strip().isdigit():
        pool_workers = int(args.pool_workers)
    else:
        raise Exception("Invalid number of pool workers.")
else:
    pool_workers = 1

if args.server_url is not None:
    base = args.server_url + ('' if args.server_url[-1] == '/' else '/')
else:
    # Default to local server if none is given.
    base = 'http://localhost:8070/api/'

print('Using GROBID server at:', base, '\n')

# Test Server Conncetion
alive_url = urljoin(base, './isalive')
res = requests.get(alive_url)

if (res.text != 'true'):
    raise Exception('Server at given address is either not running or unresponsive.')


# Convert Files #
pdf_files = [f for f in os.listdir(source_path) if '.pdf' in f]

print('Attempting to parse', len(pdf_files), 'PDF files!')
print('Output directory:', dest_path)
endpoint = urljoin(base, './processFulltextDocument')

def parse_file(name):
    print('Starting:', name)
    file_path = os.path.join(source_path, name)

    with open(file_path, 'rb') as file:
        r = requests.post(endpoint, files = {'input': file})
        xml_name = os.path.join(dest_path, name.split('.')[0] + '.xml')

        with open(xml_name, 'w') as xml_file:
            xml_file.write(r.text)
    print(name, 'Done!')

with Pool(pool_workers) as p:
    p.map(parse_file, pdf_files)