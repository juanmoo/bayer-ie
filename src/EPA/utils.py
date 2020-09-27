import os
import sys
import re
import pandas as pd


def clean_text(text):
    if type(text) is not str:
        return None
    text = text.strip().replace('||', '\n\n')
    return text

def parse_spreadsheet(paths):
    '''
    Parses spreadsheet located at given path and outputs a dictionary. This parser assumes the
    following spreadsheet format:
    Row 1: Title Row. The ith entry in this row will correspond to the title of the data located
    in the ith column.
    Row 2-L: Data Row
    Returns: Dictionary keyed by filenames containing two lists of equal length with names
    "paragraphs" and "labels". The list "paragraphs" contains text entries and the "labels"
    entries contain a list of lists of labels.
    '''
    annotations=dict()
    rationales=dict()

    data = []
    
    for path in paths:
        file_path = os.path.normpath(path)
        if not os.path.isfile(file_path):
            raise Exception('Unable to find file at %s.'%file_path)
        data.append(pd.read_excel(file_path, sheet_name=0))

    data = pd.concat(data, ignore_index=True)

    pdf_list = data['Link to label']
    text_list = data['Broad Concept Paragraph - Original OCR cut and paste']
    corrected_list = data['Broad Concept Paragraph - corrected']
    rationale_list = data['Rationale for broad concept']
    labels_list = data['Broad concept']

    for pdf, text, corrected, rationale, label in zip(pdf_list, text_list, corrected_list, rationale_list, labels_list):
        if type(pdf) is str and '.pdf' in pdf:
            text = clean_text(text)
            corrected = clean_text(corrected)
            if corrected:
                text = corrected
            if text is None or len(text) == 0:
                continue
                
            name = os.path.basename(pdf).split('.pdf')[0]
            if name not in annotations:
                annotations[name] = {'texts': [], 'labels': []}
            found = False
            for i, lb in enumerate(annotations[name]['labels']):
                if lb == label:
                    annotations[name]['texts'][i] += '\n\n' + text
                    found = True
                    break
            if not found:
                annotations[name]['texts'].append(text)
                annotations[name]['labels'].append(label)
            
            if label not in rationales:
                rationales[label] = []
            if type(rationale) is str:
                rationale = re.sub('\s+', ' ', rationale.strip())
                rationales[label] += rationale.split('||')
                
    for label in rationales:
        rationales[label] = list(set(rationales[label]))

    return annotations, rationales