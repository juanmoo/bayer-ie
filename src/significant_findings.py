'''
Extract significant finding portions through heuristics.
'''
import json, sys, os


class heuristicExtractor:
    def __init__(self):
        self.heuristics = [lambda x: False]

    def classify(self, paragraph):
        return any([f(paragraph) for f in self.heuristics])
        

class hepaticImpairmentExtractor(heuristicExtractor):

    def __init__(self):
        self.heuristics = []

        def h1(p):
            heads = [(p['header'] or '').lower(), (p['subheader'] or '').lower()]
            words1 = ['hepatic', 'liver']
            words2 = ['impair', 'failure']
            return any([(w1 in h) and (w2 in h) for w1 in words1 for w2 in words2 for h in heads])
        self.heuristics.append(h1)

        def h2(p):
            heads = [(p['header'] or '').lower(), (p['subheader'] or '').lower()]
            return any(['population' in h for h in heads]) and \
            ('hepatic' in p['text'])
        self.heuristics.append(h2)

        def h3(p):
            heads = [(p['header'] or '').lower(), (p['subheader'] or '').lower()]
            return any(['liver' in h for h in heads])
        self.heuristics.append(h3)


class renalImpairmentExtractor(heuristicExtractor):
    
    def __init__(self):
        self.heuristics = []


if __name__ == '__main__':
    
    data_path = '/scratch/juanmoo1/bayer/VendorEMAforMIT/Labels/parsed.json'
    data_file = open(data_path, 'r')
    data = json.load(data_file)

    output_dir = '/scratch/juanmoo1/bayer/VendorEMAforMIT/Labels/hepaticImpairment/'

    extractor = hepaticImpairmentExtractor()
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



