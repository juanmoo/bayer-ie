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
    try:
        root = ET.parse(file_name).getroot()
        text = []
        tag = []
        processed_text = []
        for child in root.iter():
            if child.text:
                ptext = [w for w in child.text.strip().lower().split(' ') if w.isalnum()]
    
                # Only append items with a non-zero number of alphanumeric words.
                if len(ptext) > 0:
                    text.append(child.text.strip())
                    tag.append(child.tag[29:]) # prune tei-url from tag
                    processed_text.append(' '.join(ptext))
    
        parsed_info = {}
        parsed_info['document_name'] = os.path.basename(file_name).split('.xml')[0]
        #parsed_info['document_name'] = file_name[:file_name.rfind('.xml')]
        parsed_info['element_text'] = text
        parsed_info['processed_text'] = processed_text
        parsed_info['element_tag'] = tag
        print("completing ", file_name)
        return parsed_info
    except:
        print("skipping file name ", file_name)

    

xml_files = [os.path.join(source_path, f) for f in os.listdir(source_path) if '.xml' in f]
with Pool(10) as p:
    parsed_docs = p.map(parse_xml_file, xml_files)

    output = dict()
    
    for el in parsed_docs:
        name = el['document_name']
        el.pop('document_name')
        output[name] = el
    
    output_dict = {}
    
    for doc in output:
        output_paragraphs = []
        part_list  = output[doc]['element_tag']
        for i,part in enumerate(part_list):
            if part=='p':
                paragraph = output[doc]['processed_text'][i]
                output_paragraphs.append(paragraph)
        output_dict[doc] = output_paragraphs        
    
    for doc in output_dict:
        print("check=", dest_path+"/"+doc+".json")
        with open(dest_path+"/"+doc+".json", 'w') as out_file:
            out_file.write(json.dumps(output_dict[doc], indent=4))
