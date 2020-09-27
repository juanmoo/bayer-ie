'''
Functions to parse FDA-styled XML documents.
'''


import os
import sys
import json
import re
import pandas as pd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from tqdm import tqdm
from functools import reduce
from multiprocessing import Pool


def clean_soup(soup):
    # Remove all tables from XML
    for table in soup.find_all('table'):
        table.extract()

    # Remove Warnings and Precautions Sections
    for sec in soup.find_all('section'):
        skip = True
        for code in sec.find_all('code', attrs={'displayName': 'WARNINGS AND PRECAUTIONS SECTION'}):
            skip = False
        if not skip:
            for h in sec.find_all('highlight'):
                h.extract()

    return soup


def process_xmls(documents_path, output_path=None):
    xmls_path = os.path.normpath(os.path.realpath(documents_path))

    # Input Validation
    if not os.path.isdir(documents_path):
        raise Exception(
            "Could not find valid folder at {}.".format(documents_path))

    # Read XML Files
    filenames = set([os.path.basename(path) for path in os.listdir(
        documents_path) if path.endswith('.xml')])

    raw_data = {}
    print('\nReading Documents ... \n')
    for fname in tqdm(filenames):
        full_path = os.path.join(documents_path, fname)
        soup = BeautifulSoup(open(full_path, errors='ignore').read(), 'xml')
        soup = clean_soup(soup)

        name = fname[:-4]
        raw_data[name] = {'full': soup.text, 'section': ''}

        for tag in soup.find_all('title'):
            sec = tag.text.strip()
            if len(sec) != 0:
                raw_data[name]['section'] += sec + '<SEP>'

    # Parse document data
    data = {}
    print('\nParsing document data ... \n')
    for fname in tqdm(raw_data):
        # example = {'file': fname}
        example = {}

        full = raw_data[name]['full']
        section = (raw_data[name]['section']+'<END>').split('<SEP>')

        # some section names may have change of line
        for i, sec in enumerate(section):
            if '\n' in sec:
                section[i] = sec.replace('\n', ' ')
                full = full.replace(sec, section[i])
        full = full.split('\n')

        paragraph_list = []

        idx = 0
        parent = ''

        for line in full:
            line = line.strip()
            if len(line) == 0:
                continue
            if line == section[idx]:
                idx += 1
                parent = line
            paragraph_list.append([line, parent])

        if section[idx] != '<END>':
            print(name, ' bad')
            print(section[idx])
            continue

        example['text'] = paragraph_list
        data[fname] = example

    # Collapse entries based on 'parent' column
    data = [[{'file': f, 'parent': p, 'text': t}
             for (t, p) in data[f]['text']] for f in data]
    data2 = []
    for rows in data:
        collapsed = []
        i = 0
        for j, row in enumerate(rows):
            if i > 0 and row['parent'] == rows[i]['parent']:
                rows[i]['text'] += '\n' + row['text']
            else:
                collapsed.append(rows[i])
                i = j
        collapsed.append(rows[i])
        data2.extend(collapsed)
    data = pd.DataFrame(data2)
    data['labels'] = ''
    data['significant'] = ''

    # Save parsed data if path provided
    if output_path:
        print('\nSaving parsed files ... \n')
        output_path = os.path.normpath(os.path.realpath(output_path))
        data.to_excel(output_path)
        print('Done!\n')

    return data


# if __name__ == '__main__':
#     xmls_path = '/scratch/juanmoo1/data/bayer/V1/VendorFDAforMIT/Labels'
#     output_path = '/afs/csail.mit.edu/u/j/juanmoo1/project/bayer_fda/test.xlsx'
#     annotations_dir = '/scratch/juanmoo1/data/bayer/V2/FDA_annotations_20200909'

#     data = process_xmls(xmls_path, output_path)
#     data = match_labels(data, annotations_dir)

#     data.to_excel(output_path)

#     # Print Counts
#     for lab in pd.unique(data['matched']):
#         count = len(data['matched'] == lab)
#         print('{} count: {}'.format(lab, count))
