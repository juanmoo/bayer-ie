'''
Define procedure to extract lines from background images produced from parsing PDF with pdftohtml
'''

import os, sys
from PIL import Image
import numpy as np

# Convert image to one-hot black/white array where 0 -> black & 1 -> white
def img_to_bw(img):
    threshold = 200 

    arr = np.array(img)
    rgb_to_lum = np.array([.2126, .7152, .0722]).reshape((3, 1))
    arr = np.array((arr@rgb_to_lum > threshold) * 1.0, dtype=np.uint8)
    arr = arr.reshape(arr.shape[:-1])

    return arr

'''
Find horizontal black lines in array <arr>, return dictionary
of lines in the form:
    lines = {
        row_i: [
            (start_j, stop_j),
            ...
        ],
        ...
    }


The function will only look for lines whose leftmost point falls
within (start, stop) and whose size falls within (min_line_size, max_line_size).
'''
def get_lines(arr, start=0, stop=None, min_line_size=10, max_line_size=None):

    row_num, col_num = arr.shape

    if stop is None:
        stop = arr.shape[1]

    if max_line_size is None:
        max_line_size = col_num//2

    # Row-keyed dictionary of (start, stop) tuples for lines
    lines = dict()

    for c in range(start, stop):
        for r in range(0, row_num):
            if arr[r, c] == 0:
                continue_line = False
                for i, (start, stop) in enumerate(lines.get(r, [])):
                    if c == stop + 1:
                        continue_line = True
                        lines[r][i] = (start, c)

                if not continue_line:
                    if r not in lines:
                        lines[r] = list()
                    lines[r].append((c, c))

    # Extend lines found
    for r in lines:
        for i, (start, stop) in enumerate(lines[r]):
            new_stop = stop
            while arr[r][new_stop + 1] == 0:
                new_stop += 1
            if new_stop > stop:
                lines[r][i] = (start, new_stop)

    # Filter based on line length
    for r in lines:
        lines[r] = [(start, stop) for (start, stop) in lines[r] if min_line_size <= stop - start + 1 <= max_line_size]

    # Keep non-empty row lists
    lines = {k: lines[k] for k in lines if len(lines[k]) > 0}

    return lines

'''
Get lines of png image located in <path>
'''
def get_lines_from_png(path, size, **kwargs):

    real_path = os.path.realpath(path)
    if not os.path.isfile(real_path):
        raise Exception('File in %s was not found.\n'%path)

    img = Image.open(path).resize((size))
    arr = img_to_bw(img)
    lines = get_lines(arr, **kwargs)
    return lines



if __name__ == '__main__':
    parsed_path = '/Users/juanortiz/Desktop/output'
    
    img_path = os.path.join(parsed_path, 'page2.png')

    img_shape = ((595, 842))
    lines = get_lines_from_png(img_path, img_shape, start=69, stop=80)

    print(lines)
