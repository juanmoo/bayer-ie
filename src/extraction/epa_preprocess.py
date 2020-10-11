import os
import sys
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import pandas as pd
from functools import reduce
from multiprocessing import Pool


def parse_xml(data):
    if not data:
        return None

    tmp_file = tempfile.TemporaryFile('w+b')
    tmp_file.write(data)
    tmp_file.seek(0)

    result = []
    tree = ET.parse(tmp_file)
    root = tree.getroot()
    b1list, b2list = [], []
    cur_text = []
    prev_top = "0"
    text, btext = "", ""
    for text_elem in root.iter('text'):
        _text = ''.join(text_elem.itertext()).strip()
        _btext = ' '.join([b.text for b in text_elem.iter(
            'b') if b.text is not None]).strip()
        top = text_elem.get("top")
        # Sometimes a line is splitted into several elements
        if abs(int(top)-int(prev_top)) <= 2:
            if _text:
                text += " " + _text
            if _btext:
                btext += " " + _btext
            continue
        prev_top = top
        text = text.strip()
        btext = btext.strip()
        # break at empty line or bolded line
        if text == '' or text == btext or len(cur_text) > 25:
            para_text = '\n'.join(cur_text)
            if para_text.strip() != '':
                result.append({
                    'text': para_text,
                    'b1': b1list[-1] if len(b1list) >= 1 else '',
                    'b2': b2list[-1] if len(b2list) >= 1 else ''
                })
            cur_text = []
        if text:
            cur_text.append(text)
        if btext:
            if text == btext:
                b1list.append(btext)
                b2list = []
            else:
                b2list.append(btext)
        text, btext = _text, _btext

    tmp_file.close()
    return result


def pdf_to_xml(pdf_path, PARSER_CMD="/usr/bin/pdftohtml"):
    if not (os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf')):
        raise Exception(f'No pdf file found at {pdf_path}')

    print(f'Parsing: {pdf_path}')

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_file_path = os.path.join(tmpdirname, 'tmp_file.xml')
        cmd = [PARSER_CMD, '-xml', '-f', '3', pdf_path, tmp_file_path]
        process = subprocess.run(cmd, stdout=subprocess.PIPE)
        fname = os.path.basename(pdf_path).lower().replace('.xml', '')
        if process.returncode != 0:
            print(f'Unable to parse file at {pdf_path}')
            return None
        
        with open(tmp_file_path, 'rb') as f:
            res = f.read()
            return res
    
    

def process_documents(docs_path, pool_workers=1):
    docs_path = os.path.realpath(os.path.normpath(docs_path))

    # Parameter Validation
    if not os.path.isdir(docs_path):
        raise Exception(f'Directory at \"{docs_path}\" could not be found.')
        
    # PDF -> XML
    pdfs = [f for f in os.listdir(docs_path) if f.lower().endswith('.pdf')]
    paths = [os.path.join(docs_path, fname) for fname in pdfs]

    with Pool(pool_workers) as pool:
        xml_strs = pool.map(pdf_to_xml, paths)
    
    # XML -> Dict
    with Pool(pool_workers) as pool:
        data = pool.map(parse_xml, xml_strs)

    fnames = [f[:-4].lower() for f in pdfs]
    data = dict([e for e in zip(fnames, data) if e[1]]) # Filter unsuccesful parsings

    # Dict -> DataFrame
    entries = [[{**{'file': f}, **e} for e in data[f]] for f in data.keys()]
    entries = list(reduce(lambda a, b: a + b, entries))
    data = pd.DataFrame(entries)

    # Empty Labels Column
    data['labels'] = ''

    return data



if __name__ == '__main__':
    argv = sys.argv[1:]
    path = argv[0] # First argument is path to PDFs
    pool_workers = int(argv[1]) # Second argument num of workers

    data = process_documents(path, pool_workers=pool_workers)
    data.to_excel('tmp.xlsx', index=False)