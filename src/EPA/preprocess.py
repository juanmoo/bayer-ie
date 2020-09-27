import os
import subprocess
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

# Parse Command Line Args #
argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--pdf_path', default='/data/rsg/nlp/yujieq/data/bayer/VendorEPAforMIT/Labels', help='Path of folder where PDFs are placed')
args = parser.parse_args(argv)

pdf_path = args.pdf_path
for pdf_name in sorted(os.listdir(pdf_path)):
    name = pdf_name.split('.')[0]
    xml_path = "xmls/" + name
    if not os.path.exists(xml_path):
        os.mkdir(xml_path)
    cmd = '/usr/bin/pdftohtml -xml -f 3 ' + pdf_path+'/'+pdf_name +' '+ xml_path+'/output.xml'
    os.system(cmd)

def parse_document(name):
    result = []
    tree = ET.parse(name)
    root = tree.getroot()
    b1list, b2list = [], []
    cur_text = []
    prev_top = "0"
    text, btext = "", ""
    for text_elem in root.iter('text'):
        _text = ''.join(text_elem.itertext()).strip()
        _btext = ' '.join([b.text for b in text_elem.iter('b') if b.text is not None]).strip()
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
    return result
        
    
json_obj = {}

for name in tqdm(sorted(os.listdir('xmls'))):
    xml_path = 'xmls/'+name+'/output.xml'
    if not os.path.exists(xml_path):
        print(name, 'not exists.')
        continue
    res = parse_document(xml_path)
    if len(res) != 0:
        json_obj[name] = res
        
        
with open('parsed_EPA.json', 'w') as f:
    json.dump(json_obj, f)