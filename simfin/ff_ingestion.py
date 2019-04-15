# -*- coding: utf-8 -*-
"""
Fama-French Ingestion

04/15/2019
Jared Berry
"""

import requests
import zipfile
import io

def get_fama_french():
    """
    Pull the research factors csv file (zipped); unzip and extract
    """
    r = requests.get('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    
if __name__ == '__main__':
    get_fama_french()
