import os, json

# Vars
jsons_path = '/scratch/juanmoo1/jsons'
file_name = 'EMA_dump.json'
doc_name = 'elmiron-epar-product-information_en'

with open(os.path.join(jsons_path, file_name), 'r') as f:
    doc_data = json.load(f)
    doc = doc_data[doc_name]


output = ''
output2 = ''

labels = set(doc['element_tag'])

for tag, text in zip(doc['element_tag'], doc['element_text']):

    output += tag + ' => ' + text + '\n'
    output2 += ('head' if tag == 'head' else 'text') + ' => ' + text + '\n'

with open('./all_headers.txt', 'w') as f:
    f.write(output)

with open('./only_head.txt', 'w') as f:
   f.write(output2)


