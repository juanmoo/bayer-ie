#! /usr/bin/python3

import os, sys
import requests

args = sys.argv[1:]

if len(args) < 1:
    raise Exception(
        "No input files were given."
    )

url = "http://cloud.science-miner.com/grobid/api/processFulltextDocument"
#url = "http://localhost:8070/grobid/api/processFulltextDocument"
for s in args:
    file_path = os.path.realpath(s)

    r = requests.post(url, files = {"input": open(file_path, "rb")})
    xml_name = os.path.realpath('.') + '/xmls/' + s.split('.')[0] + ".xml"
    res = r.text
    with open(xml_name, 'w') as f:
        f.write(res)


