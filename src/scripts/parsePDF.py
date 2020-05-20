#! /usr/bin/python3

import os, sys, subprocess, tempfile
import json, re, argparse
from lxml import etree as ET
from multiprocessing import Pool

# Constants
paragraph_skip_threshold = 13
left_start_paragraph_treshold = 71


''' --------------------  UTILITY FUNCTIONS -------------------- ''' 

'''
Creates dictionary from html element arguments in the form:
    "key:val; key2:val2; ... ;keyN:valN;"
'''
def parse_element_arguments(args):
    return dict([e.strip().split(':') for e in args.split(';') if len(e.strip()) > 0])

'''
Parse <style> element of a page's html file. Returns a key of the style-id to 
text type using the following criteria:

1. If the text-weight is bold, it is a section/subsection.
2. If the text-style is italic, it is a subheader.
3. If none of the above categories fit, assume that it is regular text.
'''
def parse_style_key(sk):
    style_lines = [e.strip() for e in sk.strip().split('\n') if e.strip()[0] == '#']
    key = {}
    for sl in style_lines:
        i = sl.find(' ')
        style_type = sl[1:i]
        style_def = re.sub('\{|\}', '', sl[i + 1:]).strip()
        style = parse_element_arguments(style_def)

        if style.get('font-weight', None) == 'bold':
            key[style_type] = 'section'
        elif style.get('font-style', None) == 'italic':
            key[style_type] = 'subheader'
        else:
            key[style_type] = 'text'
    return key
        
'''
Parse single page from generated HTML files using the given state. Given the path to the 
desired HTML file, this function will generate a 2D array containing elements of the form
[section, subsection, header, subheader, text] for each text line and 'None' to mark the
end of a paragraph.

At the end of the run, a dictionary with the end-state will be returned along with the
created 2D-array.
'''
def parse_page(path, section=None, subsection=None, header=None, subheader=None, current_top=None):
    output = [] #section, subsection, header, subheader, text || None

    parser = ET.HTMLParser()
    root = ET.parse(path, parser).getroot()

    # Get mapping of style-id to text type
    style_key = next(root.iter('style')).text
    style_key = parse_style_key(style_key)

    # Parse each textline available
    body = root.find('body')
    for div in body.iter('div'):
        if div.get('class') != 'txt':
            continue

        # Test if the paragraph ended by comparing vertical change
        div_attr = parse_element_arguments(div.get('style'))
        top = int(div_attr.get('top', '0px').replace('px', ''))
        left = int(div_attr.get('left', '0px').replace('px', ''))
        if left > left_start_paragraph_treshold:
            continue

        elif (current_top is None) or (top - current_top > paragraph_skip_threshold) and \
            (len(output) > 0) and (output[-1] is not None):
            output.append(None)
        current_top=top
        
        #Get all line's text
        spans = list(div.iter('span'))
        text = ' '.join([e.text for e in spans])

        # Determine line type based on styling. In case of multiple, default
        # to 'text'.
        types = [style_key[e.get('id')] for e in spans]
        ids = [e.get('id') for e in spans]
        type = types[0] if all([e == types[0] for e in types]) else 'text'

        # Text styling for sections and subsections is the same. Differentiate
        # them by searching for #.# or #. pattern
        if type == 'section' and re.match('\d\.\d', text) is not None:
            type = 'subsection'

        # Update Headers/Sections if applicable
        if type == 'section':
            section = text
            subsection = None
            header = None
            subheader = None

        elif type == 'subsection':
            subsection = text
            header = None
            subheader = None
        
        elif type == 'header':
            header = text
            subheader = None

        elif type == 'subheader':
            subheader = text
        
        else:
            output.append([section, subsection, header, subheader, text])

    state = {
                'section': section, 
                'subsection': subsection, 
                'header': header,
                'subheader': subheader
            }

    return output, state

'''
Parse PDF document in <input_file> and return JSON-like output.
'''
def parse_document(input_file):
    with tempfile.TemporaryDirectory() as temp_root:

        # Run pdftohtml command
        temp_dir = os.path.join(temp_root, 'tmp')
        cmd = 'pdftohtml {file} {out_dir}'.format(file=input_file, out_dir=temp_dir)
        return_code = subprocess.run(cmd.split(), stderr=subprocess.DEVNULL).returncode

        if return_code != 0:
            raise Exception('Unable to execute pdftohtml for given args. \n Attempted Command: %s'%(cmd))

        # Page HTML files ordered by number
        files = [f for f in os.listdir(temp_dir) if re.sub('\d', '', f) == 'page.html']
        files.sort(key=lambda s: int(re.sub('page|\.html', '', s)))

        # Parse page by page and concatenate results
        output = []
        state = {}
        for page in files:
            page_file = os.path.join(temp_dir, page)
            page_output, state = parse_page(page_file, **state)
            output.extend(page_output)

        # Restructure output
        paragraphs = []
        current_line = 0
        while output[current_line] is None:
            current_line += 1
        while current_line < len(output):
            # New Paragraph
            par_lines = []
            while current_line < len(output) and output[current_line] is not None:
                par_lines.append(output[current_line])
                current_line += 1
            current_line += 1

            if len(par_lines) > 0:
                paragraph = {
                    'section': par_lines[0][0],
                    'subsection': par_lines[0][1],
                    'header': par_lines[0][2],
                    'subheader': par_lines[0][3],
                    'text': '\n'.join(l[4] for l in par_lines)
                }
                paragraphs.append(paragraph)
        return paragraphs

if __name__ == '__main__':
    # Parse Command Line Args #
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Path to PDF file/folder with PDF file(s)')
    parser.add_argument('--dest', help='Desired path to output file')
    parser.add_argument('--pool-workers', help='Number of workers to be used to process documents simultaneously')
    args = parser.parse_args(argv)

    # Verify Args #
    source_path = os.path.realpath(args.source)
    source_dir, source_base = os.path.split(source_path)
    if not os.path.isdir(source_path) and not os.path.isdir(source_dir):
        raise Exception('Source path is invalid.')
    if os.path.isdir(source_path):
        source_dir = source_path

    dest_path = os.path.realpath(args.dest)
    dest_dir, dest_base = os.path.split(dest_path)
    if not os.path.isdir(dest_dir):
        raise Exception('Destination path is invalid.')
 
    if args.pool_workers is not None:
        if args.pool_workers.strip().isdigit():
            pool_workers = int(args.pool_workers)
        else:
            raise Exception("Invalid number of pool workers.")
    else:
        pool_workers = 1

    pdfs = [e for e in os.listdir(source_dir) if '.pdf' in e]
    
    def process_doc(e):
        name = e.replace('.pdf', '')
        print('Processing %s.'%name)
        full_path = os.path.join(source_dir, e)
        out = parse_document(full_path)
        print('Done processing %s!'%name)
        return (name, out)

    with Pool(pool_workers) as p:
        docs = p.map(process_doc, pdfs)
    
    docs = dict(docs)
    
    with open(dest_path, 'w') as f:
        f.write(json.dumps(docs, indent=4))

    