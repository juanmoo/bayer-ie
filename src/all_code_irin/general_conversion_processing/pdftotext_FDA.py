import os
import io
import json
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage



def extract_text_from_pdf(pdf_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            
        text = fake_file_handle.getvalue()
    
    # close open handles
    converter.close()
    fake_file_handle.close()
    
    if text:
        return text

path_so_far = "/Users/iringhosh/Desktop/NLP_UROP/Spring2020UROP/reviews-json/VendorFDAforMIT/"
   
if __name__ == '__main__':
    all_pdfs = os.listdir()
    all_pdfs_processed = []
    for pdf in all_pdfs:
        if pdf[0] == ".":
            pdf = pdf[1 :]
        if pdf[-7 :] == ".icloud":
            pdf = pdf[: -7]
        all_pdfs_processed.append(pdf)        
    print("all_pdfs = ", all_pdfs_processed)
    for i, pdf in enumerate(all_pdfs_processed):
        pdf_path = os.path.join(os.getcwd() , pdf)
        print("pdf_path=", pdf_path)
        text = extract_text_from_pdf(pdf_path)
        json_object = json.dumps(text) 
  
        # Writing to sample.json 
        final_path = path_so_far + "/"+ pdf[: -4]+".json"
        print("check path =", final_path )
        with open(final_path, "w") as outfile: 
            outfile.write(json_object) 
       
