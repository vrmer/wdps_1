#!/bin/sh
echo "Downloading and Installing Packages ..."
pip3 install --upgrade pip3
pip3 install -r requirements.txt
echo "Downloading and Installing English Spacy Model 'en-core-web-md' ..."
python3 -m spacy download en_core_web_md
echo "Downloading NLTK datasets ..."
python3 ./import_nltk_sets.py
echo "Running script on default values"
echo "Therefore, all the warc files will be read and saved and all ElasticSearch results will be generated and saved within the 'outputs' folder"
echo "If this script is ran once without error, you can manually call the command"
echo "To do this, set -p and -s to 'False', change -l based on your preference"
echo "Change -m to your preferred model and provide the corresponding txt document for -p."
python3 ./starter_code.py -p True -fp "warc_file_names.txt" -s True -f True -m 'popularity'
