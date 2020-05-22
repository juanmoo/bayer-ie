#! /usr/bin/python3

'''
This script is meant to be used to process XML files obtained by processing native PDFs
using GROBID's "Process Full Text Document" feature.

Given the location of the TEI XML, this script will extract a flattened list of leaf
elements with text while noting the div to which they belong.
'''

import xml.etree.ElementTree as ET
from multiprocessing import Pool
import os, json, sys, argparse

# Parse Command Line Args #
argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('source', help='Path to folder with TEI XML files')
parser.add_argument('--dest', help='Desired path to output file')
args = parser.parse_args(argv)

# Verify Args #
source_path = os.path.realpath(args.source)
if not os.path.isdir(source_path):
    raise Exception('Source path is invalid.')

if args.dest is None:
    dest_path = os.path.join(source_path, 'parsedInfo.json')
else:
    dest_path = os.path.realpath(os.path.abspath(args.dest))
    base_path = os.path.dirname(dest_path)
    if not os.path.isdir(base_path):
        raise Exception('The path: \'' + base_path + '\' does not exist.')

# Parse XMLs #
def parse_xml_file(file_name):
    root = ET.parse(file_name).getroot()

    # remove tei-rul
    for child in root.iter():
        child.tag = child.tag.replace('{http://www.tei-c.org/ns/1.0}', '')
    
    body = root.find('text').find('body')
    assert(body is not None)

    elements = []
    for child in body.iter('div'):
        h = child.find('head')
        if h is None:
            head = ''
        else:
            head = h.text
            if 'n' in h.attrib:
                head = h.get('n') + ' ' + head
                h.text = head
        e = {
            'head': head,
            'text': [p.text.strip() for p in child],
            'tag': [p.tag for p in child]
        }
        elements.append(e)

    parsed_info = {}
    parsed_info['document_name'] = os.path.basename(file_name).split('.xml')[0]
    parsed_info['elements'] = elements

    return parsed_info

xml_files = [os.path.join(source_path, f) for f in os.listdir(source_path) if '.xml' in f]
with Pool(10) as p:
    parsed_docs = p.map(parse_xml_file, xml_files)

    output = dict()
    for el in parsed_docs:
        name = el['document_name']
        el.pop('document_name')
        output[name] = el

    with open(dest_path, 'w') as out_file:
        out_file.write(json.dumps(output, indent=4))
