# CSAIL NLP Bayer Project

This is a python-based project to be used to parse and extract relevant portions of EMA/FDA/EPA PDF documents.
The modules in this project are setup up so that they can be used through a Command-line interface (CLI).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the packages specified in `requirements.py`.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The mechanism to process native PDF files is dependent on [xpdf](https://www.xpdfreader.com/)'s implementation of `pdftohtml` (version 0.62.0). In order for the modules in this project to work properly, make sure to download `pdftohtml` from [here](https://www.xpdfreader.com/download.html) and add it to your PATH.

The reviews PDFs for the retrieval task are processed with [Grobid](https://grobid.readthedocs.io/en/latest/).

## Pretrained models

Due to GitHub/GitLab's file size constraints, the pretrained models have been compressed using gunzip. To utilize them, they must first be decompressed using the following commands:

```bash
gzip -d <path to compressed EMA model>
gzip -d <path to compressed FDA model>
gzip -d <path to compressed EPA model>
```

## Usage - Extraction

Codes are located in [`src/extraction`](src/extraction).

Currently, there are four available commands. These can be accessed through the `main.py` module and are as follows:

- segment

  - Positional Arguments:

    - `source`: Data source (EMA or FDA)
    - `dir`: Path to directory containing input documents (PDFs or XMLs)
    - `output_dir`: Path to desired output file(s)

  - Optional Arguments:
    - `--pool-workers`: Number of pool workers to be used.
    - `--separate-documents`: Separate segmentation in a per-document basis.

- mapLabels:
  - Positional Arguments:
    - `data_dir`: Path to segmented files.
    - `output_dir`: Path to ouput directory.
    - `mapping_file`: Path to json file with label mappings.
  - Optional Arguments:
    - `--separate-documents`: Separate segmentations in a per-document basis.

- train

  - Positional Arguments:
    - `source`: Data source (EMA or FDA).
    - `data_dir`: Path to segmented file(s).
    - `output_dir`: Path to desired output directory.
  - Optional Arguments:
    - `--rationales_path`: Path to rationales file.


- predict

  - Positional Arguments:
    - `source`: Data source (EMA or FDA).
    - `data_dir`: Path to segmented file(s).
    - `models_path`: Path to trained models file.
    - `output_dir`: Path to desired output directory.
  - Optional Arguments:
    - `--separate_documents`: Separate segmentations in a per-document basis.

- xvalidate
  - Positional Arguments:
    - `source`: Data source (EMA or FDA).
    - `data_dir`: Path to segmented file(s).
    - `output_dir`: Path to desired output directory.
    - `num_folds`: Number of folds to use in cross validation.
  - Optional Arguments:
    - `--rationales_path`: Path to rationales file.

## Usage - Retrieval

Codes are located in [`src/retrieval`](src/retrieval).

`pdfToXML.py` converts the PDFs to XMLs. Grobid server needs to be launched beforehand. Examples can be found at [`example_data/retrieval`](example_data/retrieval).

`main.py` runs and evaluates the algorithm. Use the command `--method [BOW|WMD]` to specify the method. The predictions and evaluation results are saved to an Excel file.

```
python src/retrieval/main.py --method [BOW|WMD] --excel_path [path to the xlsx file] \
        --xml_path [path to the XML folder] --output_path [output xlsx]
```
