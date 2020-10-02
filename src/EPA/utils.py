import os
import sys
import re
import pandas as pd


old2new = {
    "138_Application_Instructions": "Application Instructions",
    "133_Agriculture_Use_Requirements": "Application Instructions",
    "174_Aerial_Applications": "Application Instructions",
    "24_Terrestrial_Use": "Application instructions",
    "175_Ground_Applications": "Application Instructions",
    "166_Ground_Application_Broadcast": "Application Instructions",
    "165_Ground_Application_Banding": "Application Instructions",
    "181_Uniform_Water_Distribution_and_System_Calibration": "Application Instructions",
    "190_Quantity_of_water_per_acre_to_achieve_efficacy": "Application Instructions",
    "173_Airblast_Air_Assist_Applications_for_Tree_Crops_and_Vineyards": "Application Instructions",
    "1474_Cover_Crops": "Application Instructions",
    "1493_Rotational_Crop_Table": "Application Instructions",
    "182_Chemigation_Monitoring": "Chemigation",
    "168_Chemigation": "Chemigation",
    "184_Use_for_Sprinkler_or_Drip_trickle_Chemigation": "Chemigation",
    "187_Injection_for_Chemigation": "Chemigation",
    "25_Non-Target_Organisms": "Env warning - species",
    "30_Endangered_Species": "Env warning - species",
    "26_Surface_Water_Advisory": "Env warning - water",
    "27_Ground_Water_Advisory": "Env warning - water",
    "180_Types_of_Irrigation_Systems": "Irrigation",
    "186_Using_Water_from_Public_Water_Systems": "Irrigation",
    "189_Solid_Set_and_Manually_Controlled_Linear_Systems": "Irrigation",
    "188_Center-Pivot_and_Automatic-Move_Linear_Systems": "Irrigation",
    "177_Order_of_Mixing": "Mixing",
    "167_Spray_Additives": "Mixing",
    "176_Compatibility": "Mixing",
    "178_Re-Suspending_WG_Products_in_Spray_Solution": "Mixing",
    "161_Compatibility_Testing_and_Tank_Mix_Partners": "Mixing",
    "18_Hazards_to_Humans_and_Domestic_Animals": "18_Hazards_to_Humans_and_Domestic_Animals",
    "172_Sensitive_Areas": "172_Sensitive_Areas",
    "154_Pesticide_Storage": "154_Pesticide_Storage",
    "93_Referral_Statement": "93_Referral_Statement",
    "134_Non-agriculture_Use_Requirements": "134_Non-agriculture_Use_Requirements",
    "32_Physical_and_Chemical_Hazards": "32_Physical_and_Chemical_Hazards",
    "135_Product_Information": "135_Product_Information",
    "160_Spray_Drift_Management": "Off Target Movement",
    "169_Droplet_Size": "Off Target Movement",
    "170_Wind_Speed": "Off Target Movement",
    "29_Run_Off_Management": "Off Target Movement",
    "171_Temperature_Inversions": "Off Target Movement",
    "31_Pollinator_Protection": "Pollinator",
    "293_Pollinator_Language": "Pollinator",
    "19_Personal_Protective_Equipment_PPE": "PPE",
    "233_Applicators_and_other_handlers_must_wear": "PPE",
    "179_Equipment_Cleanup_Procedures": "Safety Procedures",
    "183_Required_System_Safety_Devices": "Safety Procedures",
    "191_Flushing_and_Cleaning_the_Chemical_Injection_System": "Safety Procedures",
    "136_Use_Restrictions": "Use Restrictions",
    "137_Use_Precautions": "Use Restrictions",
    "56_Rotational_Crops": "Use Restrictions"
}


def clean_text(text):
    if type(text) is not str:
        return None
    text = text.strip().replace('||', '\n\n')
    return text

def parse_spreadsheet(paths, merge_labels=False):
    '''
    Parses spreadsheet located at given path and outputs a dictionary. This parser assumes the
    following spreadsheet format:
    Row 1: Title Row. The ith entry in this row will correspond to the title of the data located
    in the ith column.
    Row 2-L: Data Row
    Returns: Dictionary keyed by filenames containing two lists of equal length with names
    "paragraphs" and "labels". The list "paragraphs" contains text entries and the "labels"
    entries contain a list of lists of labels.
    '''
    annotations=dict()
    rationales=dict()

    data = []
    
    for path in paths:
        file_path = os.path.normpath(path)
        if not os.path.isfile(file_path):
            raise Exception('Unable to find file at %s.'%file_path)
        data.append(pd.read_excel(file_path, sheet_name=0))

    data = pd.concat(data, ignore_index=True)

    pdf_list = data['Link to label']
    text_list = data['Broad Concept Paragraph - Original OCR cut and paste']
    corrected_list = data['Broad Concept Paragraph - corrected']
    rationale_list = data['Rationale for broad concept']
    labels_list = data['Broad concept']
    
    if merge_labels:
        labels_list = [old2new.get(lb, lb) for lb in labels_list]

    for pdf, text, corrected, rationale, label in zip(pdf_list, text_list, corrected_list, rationale_list, labels_list):
        if type(pdf) is str and '.pdf' in pdf:
            text = clean_text(text)
            corrected = clean_text(corrected)
            if corrected:
                text = corrected
            if text is None or len(text) == 0:
                continue
                
            name = os.path.basename(pdf).split('.pdf')[0]
            if name not in annotations:
                annotations[name] = {'texts': [], 'labels': []}
            found = False
            for i, lb in enumerate(annotations[name]['labels']):
                if lb == label:
                    annotations[name]['texts'][i] += '\n\n' + text
                    found = True
                    break
            if not found:
                annotations[name]['texts'].append(text)
                annotations[name]['labels'].append(label)
            
            if label not in rationales:
                rationales[label] = []
            if type(rationale) is str:
                rationale = re.sub('\s+', ' ', rationale.strip())
                rationales[label] += rationale.split('||')
                
    for label in rationales:
        rationales[label] = list(set(rationales[label]))

    return annotations, rationales