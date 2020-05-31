'''
Extract significant finding portions through heuristics.
'''
import json, sys, os, re
import pandas as pd
import numpy as np
from pdf_parser import process_documents
from utils import load_parsed_file, tokenize_matches

class heuristicExtractor:
    def __init__(self):
        self.heuristics = [lambda x: False]

    def classify(self, paragraph):
        return any([f(paragraph) for f in self.heuristics])
        

class hepaticImpairmentExtractor(heuristicExtractor):

    def __init__(self):
        self.heuristics = []
        self.label = 'significant findings - hepatic impairment'

        '''
        Heuristic explanations:

        1. There are several variations of hepatic impairment used in the documents. This
        heuristic matches a paragraph if the header or subheader contain one of the
        variations that can be made out of a word from the list 'words1' and another from 
        'words2'

        2. In several documents, sections that deal with hepatic impairment are placed under
        a general 'populations' section. This heuristic matches a paragraph if they contain
        the word hepatic and are located in the population section.

        3. A few documents do not separate hepatic impairment sections with headers/sections,
        and only dedicate a short paragraph. This heuristic matches a paragraph if it has 
        four lines or less and explicitly mentions 'hepatic impair'.
        '''

        def h1(p):
            heads = [(p['header'] or '').lower(), (p['subheader'] or '').lower()]
            words1 = ['hepatic', 'liver']
            words2 = ['impair', 'failure', 'disorder', 'reactions']
            return any([(w1 in h) and (w2 in h) for w1 in words1 for w2 in words2 for h in heads])
        self.heuristics.append(h1)

        def h2(p):
            heads = [(p['header'] or '').lower(), (p['subheader'] or '').lower()]
            return any(['population' in h for h in heads]) and \
            ('hepatic' in (p['text'] or '').lower().split())
        self.heuristics.append(h2)

        def h3(p):
            return len([e for e in (p['header'] or '').split('\n') if len(e) > 0]) <= 4 and \
            ('hepatic impair' in re.sub('\s+|\n', ' ', (p['header'] or '').lower()))
        self.heuristics.append(h3)



class renalImpairmentExtractor(heuristicExtractor):
    
    def __init__(self):
        self.heuristics = []
        self.label = 'significant findings - renal impairment'

        '''
        Heuristic explanations:

        1. There are several variations of renal impairment used in the documents. This
        heuristic matches a paragraph if the header or subheader contain one of the
        variations that can be made out of a word from the list 'words1' and another from 
        'words2'.

        2. This sections are often located under 'population(s)' headers/subheaders. This
        heuristic matches a paragraph if they fall within a header/subheader that includes
        the word 'population' whose text talks about 'renal'. A common false negative for
        this heuristic is that of paragraphs dealing with 'elderly population' and as such
        those are explicitly excluded.

        3. A few documents do not separate hepatic impairment sections with headers/sections,
        and only dedicate a short paragraph. This heuristic matches a paragraph if it has 
        four lines or less and explicitly mentions 'renal impair'.

        '''

        def h1(p):
            heads = [(p['header'] or '').lower(), (p['subheader'] or '').lower()]
            words1 = ['renal']
            words2 = ['impair', 'failure', 'insufficiency', 'function']
            return any([(w1 in h) and (w2 in h) for w1 in words1 for w2 in words2 for h in heads])
        self.heuristics.append(h1)

        def h2(p):
            heads = [(p['header'] or '').lower(), (p['subheader'] or '').lower()]

            return any(['population' in h for h in heads]) and \
            not any(['elder' in h for h in heads]) and \
            ('renal' in (p['text'] or '').lower())
        self.heuristics.append(h2)

        def h3(p):
            return len([e for e in (p['header'] or '').split('\n') if len(e) > 0]) <= 4 and \
            ('renal impair' in re.sub('\s+|\n', ' ', (p['header'] or '')))
        self.heuristics.append(h3)


class pregnancyExtractor(heuristicExtractor):
    
    def __init__(self):
        self.heuristics = []
        self.label = 'significant findings - pregnancy'

        '''
        Heuristic explanations:

        1. Matches paragraphs that have a from of the word 'pregnancy' in the section/
        subsection and don't have a header or a subheader associated with it.

        2. Match paragraphs that contain a variation of the word pregnancy in their
        header/subheader.
        '''

        def h1(p):
            heads = [(p['header'] or '').lower(), (p['subheader'] or '').lower()]
            sects = [(p['section'] or '').lower(), (p['subsection'] or '').lower()]
            words = ['pregnancy', 'pregnant']
            return any([w in s for s in sects for w in words]) and \
            all([len(h) == 0 for h in heads])
        self.heuristics.append(h1)

        def h2(p):
            heads = [(p['header'] or '').lower(), (p['subheader'] or '').lower()]
            words = ['pregnancy', 'pregnant']
            return any([w in h for h in heads for w in words])
        self.heuristics.append(h2)

def extractSignificantFindings(data_path, output_path, output_dir=None, pool_workers=1):
    # Load data from PDFs or JSON
    if os.path.isdir(data_path): # From PDFs
        parsed_docs_path = os.path.join(output_dir, 'parsed_docs.json') if output_dir else None
        parsed = process_documents(data_path, output_path=parsed_docs_path, pool_workers=pool_workers)
        data = parsed_to_df(parsed)
    elif os.path.isfile(data_path) and data_path.lower().endswith('.json'): # From JSON
        data = load_parsed_file(data_path)
    else:
        raise Exception('Unable to load data from %d'%data_path)
    
    extractors = [hepaticImpairmentExtractor(), renalImpairmentExtractor(), pregnancyExtractor()]
    output = pd.DataFrame(columns=['document', 'label', 'text'])

    tok_data = tokenize_matches(pd.DataFrame(data))

    for e in extractors:
        match_i = np.array([e.classify(p) for p in tok_data.iloc]).reshape(-1)
        matches = data.loc[match_i]
        
        old_name = None
        section = None
        subsection = None
        header = None
        subheader = None
        cell_text = None

        for row in matches.iloc:
            i = row.name

            if cell_text is None:
                section = row['section']
                subsection = row['subsection']
                header = row['header']
                subheader = row['subheader']
                cell_text = row['text']
            
            elif (row.name - 1 == old_name) and (subheader == row['subheader']):
                cell_text += '\n' + row['text']

            elif i == list(matches.index)[-1]:
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
                    'label': e.label,
                    'text': cell
                }
                output = output.append(row_dict, ignore_index=True)

                # Start new row #
                section = row['section']
                subsection = row['subsection']
                header = row['header']
                subheader = row['subheader']
                cell_text = row['text']

            old_i = row.name

    output.to_excel(output_path)
    return output


if __name__ == '__main__':
    
    data_path = '/scratch/juanmoo1/bayer/VendorEMAforMIT/Labels/parsed.json'
    # data_path = '/scratch/juanmoo1/bayer/VendorEMAforMIT/newLabels/parsed.json'
    data_file = open(data_path, 'r')
    data = json.load(data_file)

    output_dir = '/scratch/juanmoo1/bayer/VendorEMAforMIT/Labels/hepaticImpairment/'
    # output_dir = '/scratch/juanmoo1/bayer/VendorEMAforMIT/Labels/renalImpairment/'
    # output_dir = '/scratch/juanmoo1/bayer/VendorEMAforMIT/Labels/pregnancy/'

    extractor = hepaticImpairmentExtractor()
    # extractor = renalImpairmentExtractor()
    # extractor = pregnancyExtractor()

    for doc_name in data:
        doc_data = data[doc_name]
        keep = [(i, p) for i, p in enumerate(doc_data) if extractor.classify(p)]

        with open(os.path.join(output_dir, doc_name + '.txt'), 'w+') as f:
            old_i = None
            for i, p in keep:
                if (i - 1 == old_i) and p['subheader'] == keep[i - 1][1]['subheader']:
                    f.write(p['text'])
                    f.write('\n')
                else:
                    f.write('-' * 50 + '\n')
                    f.write('Section: ' + (p['section'] if p['section'] else 'None') + '\n')
                    f.write('Subsection: ' + (p['subsection'] if p['subsection'] else 'None') + '\n')
                    f.write('Header: ' + (p['header'] if p['header'] else 'None') + '\n')
                    f.write('Subheader: ' + (p['subheader'] if p['subheader'] else 'None') + '\n')
                    f.write('\n')
                    f.write(p['text'])
                    f.write('\n')

