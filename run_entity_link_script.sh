#!/bin/sh
echo "Downloading and Installing Packages ..."
pip3 install --upgrade pip3
pip3 install -r requirements.txt
echo "Downloading and Installing English Spacy Model 'en-core-web-md' ..."
python3 -m spacy download en_core_web_md
echo "Downloading NLTK datasets ..."
python3 ./import_nltk_sets.py
echo "------------"
echo "Running script on default values"
echo "Therefore, all the warc files will be read and saved and all ElasticSearch results will be generated and saved within the 'outputs' folder"
echo "Please adjust the warc_archives argument in the config.ini file to the filepath or glob pattern you are trying to process though"
echo "If this script is ran once without error, you can keep on adjusting the config.ini file"
echo "If the WARC archives are already processed, set process_warc to False"
echo "If the candidate results are already saved, set save_es_results to False"
echo "If you want to test another model besides the prominence-based one, put in lesk, glove or bert"
echo "------------"
python3 ./starter_code.py
