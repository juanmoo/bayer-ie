#! /usr/bin/env python

from PIL import Image
import numpy as np
import json, re, argparse
import os, sys, subprocess, tempfile
from lxml import etree as ET
from multiprocessing import Pool
from time import time as time

# Constants
paragraph_skip_threshold = 20


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
Extract location of lines and tables in a background image.
'''
def parse_image(image_path, height, width):
    th = 200
    img = Image.open(image_path)
    img = img.resize((width, height)).convert('L')
    img = np.array(img, dtype=np.uint8)
    img = np.where(img > th, 255, 0)

    def unvisited_black(x, y):
        return img[x, y] == 0 and (x,y) not in visited
    
    visited = set()
    lines, tables = [], []

    for i in range(height):
        for j in range(width):
            if unvisited_black(i, j):
                queue = [(i, j)]
                visited.add((i, j))
                k = 0
                while k < len(queue):
                    x, y = queue[k]
                    k += 1
                    for dx in [-1,0,1]:
                        for dy in [-1,0,1]:
                            if unvisited_black(x+dx, y+dy):
                                queue.append((x+dx, y+dy))
                                visited.add((x+dx, y+dy))
                a = np.array(queue)
                minx, miny = a.min(0)
                maxx, maxy = a.max(0)

                if maxx-minx < 3:
                    if maxy - miny > 10:
                        lines.append((minx, miny, maxx, maxy))
                else:
                    tables.append((minx, miny, maxx, maxy))
    
    return lines, tables

'''
Parse single page from generated HTML files using the given state. Given the path to the 
desired HTML file, this function will generate a 2D array containing elements of the form
[section, subsection, header, subheader, text] for each text line and 'None' to mark the
end of a paragraph.

At the end of the run, a dictionary with the end-state will be returned along with the
created 2D-array.
'''
def parse_page(path, img_path, section=None, subsection=None, header=None, subheader=None, current_top=None):
    output = [] #section, subsection, header, subheader, text || None
    parser = ET.HTMLParser()
    root = ET.parse(path, parser).getroot()
    
    body = root.find('body')
    img = body.find('img')
    lines, tables = parse_image(img_path, int(img.get('height')), int(img.get('width')))

    def check_in_table(x):
        for x1, y1, x2, y2 in tables:
            if x1 <= x and x <= x2:
                return True
        return False
    
    def check_underline(x, h):
        for x1, y1, x2, y2 in lines:
            if x <= x1 and x1 <= x+h+1:
                return True
        return False

    # Get mapping of style-id to text type
    style_key = next(root.iter('style')).text
    style_key = parse_style_key(style_key)

    # Parse each textline available
    for div in body.iter('div'):
        if div.get('class') != 'txt':
            continue

        # Test if the paragraph ended by comparing vertical change
        div_attr = parse_element_arguments(div.get('style'))
        top = int(div_attr.get('top', '0px').replace('px', ''))
        left = int(div_attr.get('left', '0px').replace('px', ''))

        if check_in_table(top):
            continue

        elif (current_top is None) or (top - current_top > paragraph_skip_threshold) and \
            (len(output) > 0) and (output[-1] is not None):
            output.append(None)
        current_top=top
        
        #Get all line's text
        spans = list(div.iter('span'))
        span_styles = [parse_element_arguments(s.attrib.get('style')) for s in spans]
        font_size = max([int(style.get('font-size', '0px').replace('px', '')) for style in span_styles])
        text = ' '.join([e.text if e.text else '' for e in spans])

        # Determine line type based on styling. In case of multiple, default
        # to 'text'.
        types = [style_key[e.get('id')] for e in spans]
        ids = [e.get('id') for e in spans]
        type = types[0] if all([e == types[0] for e in types]) else 'text'

        # Text styling for sections and subsections is the same. Differentiate
        # them by searching for #.# or #. pattern
        if type == 'section' and re.match('\d', text) is None:
            type = 'text'

        elif type == 'section' and re.match('\d\.\d', text) is not None:
            type = 'subsection'

        elif type in ['text', 'subheader'] and check_underline(top, font_size):
            type = 'header'

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
            image_file = os.path.join(temp_dir, page.replace('.html', '.png'))
            page_output, state = parse_page(page_file, image_file, **state)
            # ignore the parts past ANNEX II
            if any([out is not None and out[4] == 'ANNEX II' for out in page_output]):
                break
            output.extend(page_output)

        # Restructure output
        paragraphs = []
        current_line = 0
        while current_line < len(output) and output[current_line] is None:
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

def process_doc(path):
    start_time = time()
    dir_name, base_name = os.path.split(path)
    name = base_name.replace('.pdf', '')
    print('Processing %s.'%name)

    out = parse_document(path)
    tot_time = time() - start_time
    tot_time = int(tot_time * 100.0 + 0.5)/100.0
    print('Done processing %s in %f seconds!'%(name, tot_time))
    return (name, out)


def process_documents(source_path, output_path=None, pool_workers=1):
    source_path = os.path.realpath(source_path)
    output_path = os.path.realpath(output_path)
    # Verify Source Path #
    if os.path.isfile(source_path):
        # Parse single PDF file
        pdfs = [source_path]
    elif os.path.isdir(source_path):
        # Parse all PDFs in given directory
        pdfs = [os.path.join(source_path, e) for e in os.listdir(source_path) if e.endswith('.pdf')]
    else:
        raise Exception('Source path is invalid.')

    if output_path is not None:
        base_dir = os.path.dirname(output_path)
        if not os.path.isdir(base_dir):
            raise Exception('Destination path \'%s\' is invalid.'%output_path)
    
    # Process Documents in parallel
    with Pool(pool_workers) as p:
        docs = p.map(process_doc, pdfs)
    docs = dict(docs)
    
    # output JSON of parsed documents if path
    # was given
    if output_path is not None:
        with open(output_path, 'w') as f:
            f.write(json.dumps(docs, indent=4))

    return docs

if __name__ == '__main__':
    # Parse Command Line Args #
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Path to PDF file/folder with PDF file(s)')
    parser.add_argument('--dest', help='Desired path to output file')
    parser.add_argument('--pool-workers', type=int, default=1, help='Number of workers to be used to process documents simultaneously')
    args = parser.parse_args(argv)
    args = vars(args)

    source_path = args.get('source', None)
    dest_path = args.get('dest', None)
    pool_workers = args.get('pool_workers', '1')

    if pool_workers <= 0: 
        raise Exception('%d is not a valid number of pool workers.'%pool_workers)

    process_documents(source_path, output_path=dest_path, pool_workers=pool_workers)