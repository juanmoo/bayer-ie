import os
import sys
import pandas as pd
from datetime import datetime

filename = sys.argv[1] 

xls_file = pd.ExcelFile(filename)
dfs = {sheet_name: xls_file.parse(sheet_name) 
          for sheet_name in xls_file.sheet_names}   
df = dfs["Main Worksheet"]

df_output = pd.DataFrame(columns=['Broad concept','Number of rows','Precision@1', 'Precision@5', 
                                  'Precision@10', 'Recall@1', 'Recall@5', 'Recall@10'])
path = os.getcwd()
concept_wise = {}
for index, row in df.iterrows():
    try:
        concept = row['Broad concept'].lower()
    except:
        print("exception ", concept) 
        continue
    precision_one, precision_five, precision_ten = row['Precision@1'], row['Precision@5'], row['Precision@10']
    recall_one, recall_five, recall_ten = row['Recall@1'], row['Recall@5'], row['Recall@10']
    if concept not in concept_wise:
        precision_recall = {k:[] for k in ['p1', 'p5', 'p10', 'r1', 'r5', 'r10']}
        concept_wise[concept] = precision_recall
    concept_wise[concept]['p1'].append(precision_one)
    concept_wise[concept]['p5'].append(precision_five)
    concept_wise[concept]['p10'].append(precision_ten)
    concept_wise[concept]['r1'].append(recall_one)
    concept_wise[concept]['r5'].append(recall_five)
    concept_wise[concept]['r10'].append(recall_ten)
    
for k in list(concept_wise.keys()):
    print("concept = ", k)
    print("number of rows =", len(concept_wise[k]['p1']))
    print("Precision@1=", sum(concept_wise[k]['p1'])/len(concept_wise[k]['p1']))
    print("Precision@5=", sum(concept_wise[k]['p5'])/len(concept_wise[k]['p5']))
    print("Precision@10=", sum(concept_wise[k]['p10'])/len(concept_wise[k]['p10']))
    print("Recall@1=", sum(concept_wise[k]['r1'])/len(concept_wise[k]['r1']))
    print("Recall@5=", sum(concept_wise[k]['r5'])/len(concept_wise[k]['r5']))
    print("Recall@10=", sum(concept_wise[k]['r10'])/len(concept_wise[k]['r10']))
    print("-----")
    
    df_output = df_output.append({'Broad concept': str(k),
                      'Number of rows': str(len(concept_wise[k]['p1'])),
                      'Precision@1': str(sum(concept_wise[k]['p1'])/len(concept_wise[k]['p1'])), 
                      'Precision@5': str(sum(concept_wise[k]['p5'])/len(concept_wise[k]['p5'])), 
                      'Precision@10': str(sum(concept_wise[k]['p10'])/len(concept_wise[k]['p10'])), 
                      'Recall@1': str(sum(concept_wise[k]['r1'])/len(concept_wise[k]['r1'])), 
                      'Recall@5': str(sum(concept_wise[k]['r5'])/len(concept_wise[k]['r5'])), 
                      'Recall@10': str(sum(concept_wise[k]['r10'])/len(concept_wise[k]['r10']))
                      }, ignore_index=True) 

timestring = datetime.now().strftime("%H:%M:%S")
writer = pd.ExcelWriter(path + '/output_xls_files/testFDA_wmd_average_metric_wrt_concept'+
                                "_"+timestring+'.xlsx', engine='xlsxwriter')
df_output.to_excel(writer,sheet_name='Main Worksheet',index=False)
writer.save()       
