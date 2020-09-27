import os
import pandas as pd
import json
from fuzzywuzzy import fuzz
import re
import multiprocessing
from tqdm import tqdm
from utils import parse_spreadsheet


annotations, all_rationales = parse_spreadsheet(['/data/rsg/nlp/yujieq/data/bayer/VendorEPAforMIT/CS Annotations_2020-01-20.xlsx', 
                                                 '/data/rsg/nlp/yujieq/data/bayer/VendorEPAforMIT/CS Annotations_Additional rows.xlsx'])

all_labels = list(all_rationales.keys())
print(all_labels)

with open('parsed_EPA.json') as f:
    json_obj = json.load(f)


def parsed_to_df(parsed_data):
    data = []
    for name in parsed_data:
        for p in parsed_data[name]:
            p['doc_name'] = name
            data.append(p)

    return pd.DataFrame.from_dict(data)

data = parsed_to_df(json_obj)

def contains_test(query, whole, ignore_spacing=True, minimum_paragraph_length=10):

    if len(query.split()) < minimum_paragraph_length:
        return False
    
    if ignore_spacing:
        # Remove spaces for consistent matchings
        query = re.sub('\s|\n', '', query)
        whole = re.sub('\s|\n', '', whole)

    threshold = 95
    match_score = fuzz.partial_ratio(query, whole)
    return match_score >= threshold

def match_paragraph_with_annotation(paragraph, anno_texts, anno_labels):
    labels = []
    for j, anno_text in enumerate(anno_texts):
        if contains_test(paragraph, anno_text):
            labels.append(anno_labels[j])
    return labels

    
def match_labels(data, annotations, minimum_paragraph_length=10):
    
    data["labels"] = ""
    for l in all_labels:
        data[l] = 0
    
    print('Documents progress:')
    for doc_name in pd.unique(data['doc_name']):
        doc_annotations = annotations.get(doc_name, None)
        
        print('Processing', doc_name)
        if doc_annotations == None:
            print(f'No annotations found for {doc_name}')
            continue
        
        anno_texts = doc_annotations['texts']
        anno_labels = doc_annotations['labels']
        
        doc_indices = data.index[data['doc_name'] == doc_name].tolist()
        paragraphs = []
        
        for idx in doc_indices:
            paragraphs.append(data.iloc[idx]['text'])
            
        with multiprocessing.Pool(30) as p:
            match_labels = p.starmap(match_paragraph_with_annotation, [(paragraph, anno_texts, anno_labels) for paragraph in paragraphs])

        for i, idx in enumerate(doc_indices):
            data.iloc[idx, data.columns.get_loc("labels")] = ",".join(match_labels[i])
            for l in match_labels[i]:
                data.iloc[idx, data.columns.get_loc(l)] = 1
                    
    return

match_labels(data, annotations)

# Remove documents that cannot match the annotation
error_doc = []
for name in pd.unique(data['doc_name']):
    print(name)
    matched = 0
    doc_data = data[data["doc_name"] == name]
    for i, l in enumerate(annotations[name]['labels']):
        if doc_data[l].sum() > 0:
            matched += 1
    if matched <= len(annotations[name]['labels']) / 10:
        error_doc.append(name)

for name in error_doc:
    doc_indices = data.index[data['doc_name'] == name].tolist()
    data.drop(doc_indices, inplace=True)


data.to_pickle("processed_data.pkl")
