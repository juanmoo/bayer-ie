# CSAIL NLP Bayer Project
This is a python-based project to be used to parse and extract relevant portions of EMA-styled PDF documents.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the packages specified in ```requirements.py```.

```bash
pip install -r requirements.txt
```

The mechanism to process native PDF files is dependent on [xpdf](https://www.xpdfreader.com/)'s implementation of ```pdftohtml```. In order for the modules in this project to work properly, make sure to download ```pdftohtml``` from [here](https://www.xpdfreader.com/download.html) and add it to your PATH.

## Usage
The modules in this project are setup up so that they can be used through a Command-line interface (CLI).

Currently, there are three available commands. These can be accessed through the ``main.py`` module located in [src](src/) and are as follows:

* trainModel
    * Required Arguments
        * ```data```: Path to directory containing PDFs or JSON document containing a parsed version of the PDFs.
        * ```annotations```: Path to spreadsheet containing the annotations to be used to train the model. In addition to the labels, this document should also contain a rationale column with pertinent phrases for the concept.
        * ```output_dir```: Directory where the JSON-encoded representations of PDFs are to be placed if ```data``` is a directory containing PDFs.
    * Optional Arguments
        * ```pool-workers```: In the case that the input data is a directory with PDFs, this option specifies the number of threads to be used simmultaneously to process the documents.
        * ```exact-match```: This option specifies whether or not an exact-match method should be used to match the given annotations to the parsed documents. By default, this option is set to false and a  Levenshtein Distance based fuzzy-matching method is used.

Example usage:
```
<path to project>/src/main.py trainModel --pool-workers=1 --exact-match <data-path> <annotations-path> <output-dir-path>
```

* extractSections
    * Required Parameters
        * ```models```: Path to serialized model file created using ```trainModel```.
        * ```data```: Path to directory containing PDFs or JSON document containing a parsed version of the PDFs. 
        * ```output_path```: Desired path to output spreadsheed containing the extracted sections.
    * Optional Parameters
        * ```checkpoint_dir```: Directory where the JSON-encoded representations of PDFs are to be placed if ```data``` is a directory containing PDFs.
        * ```pool-workers```: In the case that the input data is a directory with PDFs, this option specifies the number of threads to be used simmultaneously to process the documents.
        * ```exact-match```: This option specifies whether or not an exact-match method should be used to match the given annotations to the parsed documents. By default, this option is set to false and a  Levenshtein Distance based fuzzy-matching method is used.

Example Usage:
```
<path to project>/src/main.py extractSections --checkpoint-dir=<checkpoint-dir path> --pool-workers=1 --exact-match <models-path> <data-path> <output-path>
```

* extractSignificant
    * Required Parameters
        * ```data```: Path to directory containing PDFs or JSON document containing a parsed version of the PDFs. 
        * ```output_path```: Desired path to output spreadsheed containing the extracted sections.
    * Optional Arguments
        * ```pool-workers```: In the case that the input data is a directory with PDFs, this option specifies the number of threads to be used simmultaneously to process the documents.
        * ```exact-match```: This option specifies whether or not an exact-match method should be used to match the given annotations to the parsed documents. By default, this option is set to false and a  Levenshtein Distance based fuzzy-matching method is used.

Example Usage:
```
<path to project>/src/main.py extractSignificant --pool-workers=1 --exact-match <data-path> <output-path>
```

* crossValidate
    * Required Parameters
        * ```data```: Path to directory containing PDFs or JSON document containing a parsed version of the PDFs. 
        * ```annotations```: Path to spreadsheet containing the annotations to be used to train the model. In addition to the labels, this document should also contain a rationale column with pertinent phrases for the concept.
        * ```output_dir```: Directory where the JSON-encoded representations of PDFs are to be placed if ```data``` is a directory containing PDFs.
        * ```num_folds```: Number of folds to use during cross validation.
    * Optional Parameters
        * ```pool-workers```: In the case that the input data is a directory with PDFs, this option specifies the number of threads to be used simmultaneously to process the documents.
        * ```exact-match```: This option specifies whether or not an exact-match method should be used to match the given annotations to the parsed documents. By default, this option is set to false and a  Levenshtein Distance based fuzzy-matching method is used.

Example Usage:
```
<path to project>/src/main.py crossValidate --pool-workers=1 --exact-match <data-path> <annotations-path> <output-dir-path> num_folds
```