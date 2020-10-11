'''
Utility functions to be used in Bayer Project
'''

import pandas as pd
import numpy as np
import os
import sys
import re
import spacy
import codecs
import json
import pickle
from fuzzywuzzy import fuzz
from tqdm import tqdm
import collections
from multiprocessing import Pool


def translate_labels(data, label_map):
    def mapping_function(ls): return ' || '.join(set(
        [label_map.get(e.strip(), e.strip()) for e in ls.split(' || ') if e]))
    data['labels'] = data['labels'].apply(mapping_function)
    return data


def save_separate_documents(df, output_dir):
    for fname in pd.unique(df['file']):
        doc_df = df.loc[df['file'] == fname].to_excel(
            os.path.join(output_dir, fname + '.xlsx'))


def format_results(data, results):
    df = pd.DataFrame.from_dict(results)
    for c in df.columns:
        df[c] = df[c].apply(lambda l: (sum(l)/len(l)) if len(l) > 0 else 0)

    df = df.transpose()
    df['count'] = 0
    for l in df.index:
        df.at[l, 'count'] = data[l].sum()
    df.columns = ['precision', 'recall', 'F1', 'count']

    return df


def parsed_to_df(parsed_data):
    data = []
    for name in parsed_data:
        for p in parsed_data[name]:
            p['file'] = name
            data.append(p)

    df = pd.DataFrame.from_dict(data)
    cols = ['file'] + [c for c in df.columns if c != 'file']
    df = df[cols].copy().reindex()

    return df


def load_parsed_file(path):
    parsed_str = codecs.open(
        path, 'r', encoding='utf-8', errors='replace').read()
    parsed_data = json.loads(parsed_str)
    return parsed_to_df(parsed_data)


def match_labels(data, annotations, exact_match=False, minimum_paragraph_length=10, **kwargs):
    all_labels = set()
    print('Documents progress:')
    for doc_name in tqdm(pd.unique(data['doc_name'])):
        doc_annotations = annotations.get(doc_name, None)

        if doc_annotations == None:
            print('No annotations found for {docname}'.format(
                docname=doc_name))
            continue

        a_texts, a_labels = list(doc_annotations.items())
        a_texts = a_texts[1]
        a_labels = a_labels[1]

        doc_indices = data.index[data['doc_name'] == doc_name].tolist()
        for i in doc_indices:
            # Join paragraphs lines and ensure single space

            paragraph = re.sub(
                '\s+', ' ', data.iloc[i]['text'].strip().replace('\n', ' '))

            for j, a_text in enumerate(a_texts[:10]):
                if contains_test([paragraph], a_text, exact_match=exact_match, minimum_paragraph_length=minimum_paragraph_length):
                    # Add Label to data point
                    l = a_labels[j].strip().lower()
                    if l not in data.columns:
                        all_labels.add(l)
                        data[l] = 0
                    data.iloc[i, data.columns.get_loc(l)] = 1

    return sorted(list(all_labels))


def tokenize_matches(data):

    data['section'] = data['section'].apply(tokenize_string)
    data['subsection'] = data['subsection'].apply(tokenize_string)
    data['header'] = data['header'].apply(tokenize_string)
    data['subheader'] = data['subheader'].apply(tokenize_string)
    data['text'] = data['text'].apply(tokenize_string)

    return data

########## String Utilities ##########


def contains_test(pieces, whole, exact_match=False, ignore_spacing=True,
                  minimum_paragraph_length=10):
    # Fuzzy matching tests to see if the max
    # similarity substring of size len(piece)
    # has a partial ratio > threshold

    pieces = [p for p in pieces if len(p.split()) >= minimum_paragraph_length]

    if ignore_spacing:
        # Remove spaces for consistent matchings
        pieces = [re.sub('\s|\n', '', p) for p in pieces]
        whole = re.sub('\s|\n', '', whole)

    if exact_match:
        return any([(piece in whole) for piece in pieces])

    threshold = 95
    return any([fuzz.partial_ratio(piece, whole) >= threshold for piece in pieces])


def is_section(head):
    if not head:
        return False
    tok = head.split()[0]
    if tok.isdigit():
        return True
    if tok.replace('.', '', 1).isdigit():
        return True
    return False


def clean_label(lb):
    lb = lb.strip()
    if lb == 'Significant findings - pregnancy':
        return 'Significant Findings - Pregnancy'
    return lb


NLP = spacy.load('en_core_web_sm')
MAX_CHARS = 20000


def tokenize_string(comment):
    if comment == None:
        return ''

    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return ' '.join([x.text.lower() for x in NLP.tokenizer(comment) if x.text != " "])

############## Logistical Utilities ##############

# Save / Load values from checkpoint file


def save_value(key, val, path=None):
    d, base = os.path.split(path)
    if not os.path.isdir(d):
        os.makedirs(d)
    try:
        f = open(path, 'rb+')
        saved_env = pickle.load(f)
        assert(type(saved_env) == dict)
    except:
        saved_env = dict()

    saved_env[key] = val

    with open(path, 'wb') as f:
        f.write(pickle.dumps(saved_env))


def load_value(key, path=None):

    with open(path, 'rb') as f:
        try:
            saved_env = pickle.load(f)
            ans = saved_env[key]
        except:
            ans = None

        return ans


def load_env_keys(path):
    with open(path, 'rb') as f:
        try:
            saved_env = pickle.load(f)
        except:
            saved_env = dict()

        return saved_env.keys()


if __name__ == '__main__':

    p = "/data/rsg/nlp/fake_proj/__temp__juanmoo__/bayer/VendorEMAforMIT/annotations.xmls"
    parse_spreadsheet(p)
