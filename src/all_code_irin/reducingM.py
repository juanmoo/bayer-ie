import sys
import os
import pandas as pd
from datetime import datetime
from nltk.tokenize import sent_tokenize
#
#corpus = api.load('text8')
#model = api.load("glove-wiki-gigaword-50")
#model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)

path = os.getcwd()
directory = "/Desktop/Spring2020UROP"
data_type = "VendorEMAforMIT"
filename = sys.argv[1] 

def get_drug_name(drug_file_name):
    drug_name = ""
    for e in drug_file_name[1 :]:
        if e=="-":
            break
        drug_name += e
    return drug_name 

    
file_name = path + directory+"/output_xls_files/"+filename
xls_file = pd.ExcelFile(file_name)
dfs = {sheet_name: xls_file.parse(sheet_name) 
          for sheet_name in xls_file.sheet_names}
    

    
df = dfs[list(dfs.keys())[0]]
print(list(df.keys()))

broad_concepts = list(df['Broad Concept Paragraph'])
assessment = list(df['Text in assessment report 1'])
file_names = list(df['Link to label (or filename if using files sent by file transfer)'])
predicted_paragraphs = list(df['Paragraphs'])



file_concept = [(file_names[i], broad_concepts[i], predicted_paragraphs[i], assessment[i]) for i in range(len(file_names))
                                                    if type(file_names[i])!=float 
                                                    if type(broad_concepts[i])!=float]


output1 = []
output2 = []

review_files_xls = os.listdir(path + directory + "/reviews-json/"+data_type) 

M1 , M2 = 5, 1
for i, (file_name, broad_concept, prediction, assessment) in enumerate(file_concept):
    try:
        prediction_paragraphs1 = sent_tokenize(prediction)
        prediction_paragraphs2 = sent_tokenize(prediction)
        
        print(len(prediction_paragraphs1))
    
        try:
            prediction_paragraphs1 = prediction_paragraphs1[: M1]
        except:
            pass
        try:
            prediction_paragraphs2 = prediction_paragraphs2[: M2]
        except:
            pass
    
        prediction1 = "\n".join(prediction_paragraphs1)
        prediction2 = "\n".join(prediction_paragraphs2)
        
        output1.append((file_name, broad_concept, prediction1,  assessment))  
        output2.append((file_name, broad_concept, prediction2, assessment))
        
        print("i=", str(i))
    except:
        pass
          
           
        
df1 = pd.DataFrame(output1, columns = ['Link to label (or filename if using files sent by file transfer)', 
                                     'Broad Concept Paragraph','Paragraphs','Text in assessment report 1'])
timestring = datetime.now().strftime("%H:%M:%S")
writer = pd.ExcelWriter(path + directory + '/output_xls_files/testEMA-ntkl-prf1-M='+str(M1)+"-"+timestring+'.xlsx', engine='xlsxwriter')
df1.to_excel(writer,sheet_name='test',index=False)
writer.save() 

df2 = pd.DataFrame(output2, columns = ['Link to label (or filename if using files sent by file transfer)', 
                                     'Broad Concept Paragraph','Paragraphs','Text in assessment report 1'])
writer = pd.ExcelWriter(path + directory + '/output_xls_files/testEMA-ntkl-prf1-M='+str(M2)+"-"+timestring+'.xlsx', engine='xlsxwriter')
df2.to_excel(writer,sheet_name='test',index=False)
writer.save()    
