#!/usr/bin/env python3
"""
This script takes as input a pickle dictionary with {efoid:listofstudies} and download them in a directory
"""

# =====================
# Imports
# =====================
import argparse
import pandas as pd
import pickle as pkl
import requests
import os
from multiprocessing import Pool, cpu_count
from sklearn.utils import resample
import subprocess



def download_harmonised_gcst(
    url,
    outdir,
    resume=True,
    quiet=False,
):
    """
    Download only the 'harmonised' folder for a given GCST study
    from the EBI GWAS summary statistics FTP.
    """

    os.makedirs(outdir, exist_ok=True)


    cmd = [
        "wget",
        "-r",
        "-np",
        "-nH",
        "--cut-dirs=7"
    ]

    if resume:
        cmd.append("--continue")

    if quiet:
        cmd.append("--no-verbose")

    cmd.append(url)

    subprocess.run(
        cmd,
        cwd=outdir,
        check=True
    )


def download_studies(study_dictionary, outfolder_path):
    for efo, studies in study_dictionary.items():
        efolder_path = os.path.join(outfolder_path, efo)
        os.makedirs(efolder_path, exist_ok=True)

        for study in studies:
            url = f"https://www.ebi.ac.uk/gwas/rest/api/v2/studies/{study}"
            out_file = os.path.join(efolder_path, f"{study}.json")
            out_folder = os.path.join(efolder_path,study)
            # Skip if already downloaded
            if os.path.exists(out_file):
                print(f"Skipping {study}, already exists")
                continue

            study_info = requests.get(url).json()
     
            ftp_link = study_info['full_summary_stats']
            ftp_link = ftp_link.replace('http://','ftp://')
            
            
            print("[DOWNLOADING STUDY]")

            download_harmonised_gcst(ftp_link,out_folder)

def main():
    efodict_path = '../data/efo_dict.pkl'
    with open(efodict_path,'rb') as f:
        efodict = pkl.load(f)
    download_studies(efodict,'../efos/')
    


# =====================
# Entry point
# =====================
if __name__ == "__main__":
    main()
