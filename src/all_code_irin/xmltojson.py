import xmltodict
import json
s= '/Users/iringhosh/Desktop/NLP_UROP/Spring2020UROP/reviews-xml/VendorFDAforMIT/ADYNOVATE_95299.xml'
output = '/Users/iringhosh/Desktop/NLP_UROP/Spring2020UROP/reviews-json/VendorFDAforMIT/ADYNOVATE_95299.json'
with open(s) as in_file:
    xml = in_file.read()
    with open(output, 'w') as out_file:
        json.dump(xmltodict.parse(xml), out_file)
        print("dumped!")
