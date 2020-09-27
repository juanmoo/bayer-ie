# CSAIL NLP Bayer Project
This is a python-based project to be used to parse and extract relevant portions of EMA-styled PDF documents.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the packages specified in ```requirements.py```.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The mechanism to process native PDF files is dependent on [xpdf](https://www.xpdfreader.com/)'s implementation of ```pdftohtml``` (version 0.62.0). In order for the modules in this project to work properly, make sure to download ```pdftohtml``` from [here](https://www.xpdfreader.com/download.html) and add it to your PATH.

## EMA&FDA Usage
The modules in this project are setup up so that they can be used through a Command-line interface (CLI).

Currently, there are three available commands. These can be accessed through the ``main.py`` module located in [src](src/) and are as follows:

* segment
    * Positional Arguments:
        * ```source```: Data source (EMA or FDA)
        * ```dir```: Path to directory containing input documents (PDFs or XMLs)
        * ```output-dir```: Path to desired output file(s)

    * Optional Arguments:
        * ```--pool-workers```: Number of pool workers to be used.
        * ```--separate-documents```:  Separate segmentation in a per-document basis.

* train
    * Positional Arguments:
        * ```Source```: Data source (EMA or FDA).
        * ```data_dir```: Path to segmented file(s).
        * ```rationales_path```:  Path to rationales file.
        * ```output_dir```: Path to desired output directory.

* predict
    * Positional Arguments:
        * ```Source```: Data source (EMA or FDA).
        * ```data_dir```: Path to segmented file(s).
        * ```rationales_path```:  Path to rationales file.
        * ```models_path```:  Path to trained models file.
        * ```output_dir```: Path to desired output directory.
