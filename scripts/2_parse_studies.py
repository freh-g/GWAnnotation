#!/usr/bin/env python3
"""
This script loads GWAS Catalogue studies and will create a vcf with updated positions on Grch38 and target based on a certain pvalue threshold.
"""

# =====================
# Import
# =====================
import argparse
import pandas as pd
import pickle
import os
import requests
import subprocess
import yaml


# =====================
# Funzioni principali
# =====================
def load_gwas_study(input_path):
    print(f"Loading GWAS Catalogue studies from {input_path}")
    return pd.read_csv(input_path, sep="\t")


def extract_sample_sizes(metadata_path):
    """
    Extract sample sizes (total and per ancestry) from a GWAS-SSF metadata file.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    samples = metadata.get("samples", [])
    by_ancestry = {}
    total = 0

    for sample in samples:
        ancestries = sample.get("sample_ancestry_category", [])
        size = sample.get("sample_size", 0)
        total += size
        for anc in ancestries:
            by_ancestry[anc] = by_ancestry.get(anc, 0) + size

    return {"total_sample_size": total, "by_ancestry": by_ancestry}


def main():
    full_sumstat_path = '../efos/'

    for efo_folder in os.listdir(full_sumstat_path):
        if 'EFO' in efo_folder:
            print(efo_folder)
            efo_path = os.path.join(full_sumstat_path, efo_folder)
            if not os.path.isdir(efo_path):
                continue
    
            for study in os.listdir(efo_path):
                study_path = os.path.join(efo_path, study)
                if not os.path.isdir(study_path):
                    continue
    
                study_df = None
                sample_size = None
    
                for data in os.listdir(study_path):
                    data_path = os.path.join(study_path, data)
    
                    # --- Read summary stats ---
                    if (
                        "check" not in data
                        and ".h" in data
                        and "tbi" not in data
                        and "meta" not in data
                        and os.path.isfile(data_path)
                    ):
                        print(f"file_read: {data}")
                        df = pd.read_csv(data_path, sep="\t")
    
                        # Dynamically select relevant columns
                        rsid_columns   = [c for c in df.columns if "rsid" in c.lower()]
                        # pos_columns    = [c for c in df.columns if 'pos' in c.lower() or 'base_pair' in c.lower()]
                        allele_columns = [c for c in df.columns if c=="effect_allele"]
    
                        needed_cols = rsid_columns + allele_columns + ["beta", "p_value"]
    
                        # Only keep columns that actually exist in the file
                        needed_cols = [c for c in needed_cols if c in df.columns]
    
                        df = df[needed_cols]
                        df["study_id"] = study
                        study_df = df
    
                    # --- Extract sample size from metadata ---
                    elif (
                        "check" not in data
                        and ".h" in data
                        and "yaml" in data
                        and "meta" in data
                        and os.path.isfile(data_path)
                    ):
                        print(f"metadata_read: {data}")
                        sample_size = extract_sample_sizes(data_path)["total_sample_size"]
    
                    # else:
                    #     print(f"file_NOT_read: {data}")
    
                # --- If both sumstats and sample size exist ---
                if study_df is not None:
                    study_df["sample_size"] = sample_size if sample_size is not None else pd.NA
    
                    # --- Save the parsed study ---
                    out_dir = f'../parsed_studies/{efo_folder}/'
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    study_df['effect_allele'] = study_df['effect_allele'].astype(str).apply(lambda x: x.upper())
                    study_df = study_df.dropna()
                    study_df.to_csv(f'../parsed_studies/{efo_folder}/{study}.tsv',sep = '\t',index = None)


# =====================
# Entry point
# =====================
if __name__ == "__main__":
    main()