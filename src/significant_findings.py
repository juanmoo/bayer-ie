'''
Extract significant finding portions through heuristics.
'''
import json, sys, os, re

class heuristicExtractor:
    def __init__(self):
        self.heuristics = [lambda x: False]

    def classify(self, paragraph):
        return any([f(paragraph) for f in self.heuristics])
        

class hepaticImpairmentExtractor(heuristicExtractor):

    def __init__(self):
        self.heuristics = []

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
            ('hepatic impair' in re.sub('\s+|\n', ' ', (p['header'] or '')))
        self.heuristics.append(h3)



class renalImpairmentExtractor(heuristicExtractor):
    
    def __init__(self):
        self.heuristics = []

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

