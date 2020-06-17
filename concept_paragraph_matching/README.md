## Script : wmd_model.py

This script takes 3 arguments 

1. First argument should be the path of the xlsx file which has the broad concept-paragarph matching and and related information. 

2. Second argument should be the directory which has all the json files (The XML files can be converted to JSON files with the parseXML.py script)

3. Third argument should be the path where we want the output xlsx file with the top predictions to be put   

## Notes

The 'Predicted paragraphs' column of the output xlsx file will have the top 10 predicted paragraphs where the paragraphs are separated by two newline characters ('\n\n')


