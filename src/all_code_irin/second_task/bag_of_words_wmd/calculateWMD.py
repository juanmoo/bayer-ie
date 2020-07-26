import spacy
import wmd
import pandas as pd 
import os
import json


path = os.getcwd()
directory = "/Desktop/Spring2020UROP"
data_type = "VendorEMAforMIT"

def get_drug_name(drug_file_name):
    drug_name = ""
    for e in drug_file_name[1 :]:
        if e=="-":
            break
        drug_name += e
    return drug_name    

file_name = path + directory+"/data/bayer/"+data_type+"/BayerLabelAnnotationWorksheet_Genpact_MasterEMA.xlsx"
xl_file = pd.ExcelFile(file_name)
dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}

df = dfs["Main Worksheet"]

broad_concepts = list(df['Broad Concept Paragraph'])
assessment = list(df['Text in assessment report 1'])
file_names = list(df['Link to label (or filename if using files sent by file transfer)'])

positive_examples = [(broad_concepts[i], assessment[i], file_names[i]) for i in range(len(broad_concepts ))]
positive_examples = list(filter(lambda t: float not in map(type, t),  positive_examples)) #removing NaNs


positive_examples_dictionary = {}
for concept, review, f in positive_examples:
    drug_name = get_drug_name(f)
    positive_examples_dictionary[drug_name] = (concept, review)   
    
#positive_examples_dictionary is filename: (broad_concept, supporting_review)    
    
negative_examples_dictionary = {}

review_files_xls = os.listdir(path + directory + "/reviews-xls/"+data_type)


    
for i,drug_name in enumerate(positive_examples_dictionary):
    broad_concept = positive_examples_dictionary[drug_name]
    try:
        drug_name = "abasria"
        file_path = list(filter(lambda f: f[7:7+len(drug_name)]==drug_name, review_files_xls))[0]
        with open(path + directory + "/reviews-xls/"+data_type+"/"+file_path) as f:
            data = json.load(f)
            included_paragraphs = []
            for paragraph in data:
                included_paragraphs.append(paragraph)
            negative_examples_dictionary[drug_name] = (broad_concept, included_paragraphs)     
    except:
        pass
        
#       
        
        
#    for 
#        if :
#            negative_examples.append((concept, not_supporting))
#nlp = spacy.load('en', create_pipeline=wmd.WMD.create_spacy_pipeline)
#doc1 = nlp("Politician speaks to the media in Illinois.")
#doc2 = nlp("The president greets the press in Chicago.")
#print(doc1.similarity(doc2))

