#!/usr/bin/env python3
"""
This script loads GWAS Catalogue studies and creates a VCF-like table with
updated GRCh38 positions. SNPs are lifted using dbSNP chunks and merged
back to the original GWAS associations on rsID and effect allele.
"""

# =====================
# Imports
# =====================
import argparse
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from sklearn.utils import resample


# =====================
# Argument parsing
# =====================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", "-i",
        required=False,
        default="../parsed_studies",
        help="Study folder path"
    )

    parser.add_argument(
        "--pvalue", "-p",
        required=False,
        default=5e-8,
        help="P-value threshold"
    )

    return parser.parse_args()


# =====================
# Utility functions
# =====================
def load_parsed_associations(input_path):
    return pd.read_csv(input_path, sep='\t')


# =====================
# Multiprocessing worker
# =====================
def process_file(args):
    """
    Read one dbSNP chunk and return rows overlapping the SNP set.
    """
    file, snps, path_dbsnp = args

    if "check" in file:
        return None

    file_path = os.path.join(path_dbsnp, file)

    try:
        tmp = pd.read_csv(file_path, sep="\t", header=None)
    except Exception as e:
        print(f"[WARNING] Failed reading {file}: {e}")
        return None

    tmp = tmp[tmp[2].isin(snps)]
    return tmp if not tmp.empty else None


# =====================
# Core logic
# =====================

def lift_database(study_df, path_dbsnp):
    """
    Lift GWAS SNPs using dbSNP chunks and merge lifted coordinates back to
    the study dataframe. Matching is done on both hm_rsid and effect_allele.
    """

    df = study_df.copy()

    # ---- Ensure hm_rsid column exists ---- #
    if "hm_rsid" not in df.columns:
        if "rsid" in df.columns:
            df = df.rename(columns={"rsid": "hm_rsid"})
        else:
            raise ValueError("Neither 'hm_rsid' nor 'rsid' column found")

    # ---- Ensure effect_allele column exists ---- #
    if "effect_allele" not in df.columns:
        raise ValueError("No 'effect_allele' column found in study dataframe")

    # ---- Get SNP list ---- #
    snps = set(df["hm_rsid"].dropna().astype(str))
    files = os.listdir(path_dbsnp)

    if not files:
        raise ValueError("dbSNP directory is empty")

    tasks = [(f, snps, path_dbsnp) for f in files]

    num_workers = min(cpu_count(), len(tasks))
    with Pool(num_workers) as pool:
        results = pool.map(process_file, tasks)

    results = [r for r in results if r is not None and not r.empty]

    if not results:
        print("[WARNING] No SNPs found in dbSNP")
        return pd.DataFrame()

    # ---- Build lifted VCF-like table ---- #
    vcf_lifted = pd.concat(results, ignore_index=True)
    vcf_lifted.columns = ["chr", "pos", "hm_rsid", "ref", "alt"]

    # ---- Merge on rsid + effect_allele ---- #
    merged = pd.merge(
        vcf_lifted,
        df,
        left_on=["hm_rsid", "alt"],
        right_on=["hm_rsid", "effect_allele"],
        how="inner"
    )

    n_total   = len(df["hm_rsid"].dropna().unique())
    n_matched = merged["hm_rsid"].nunique()
    print(f"  rsIDs queried : {n_total}")
    print(f"  rsIDs matched : {n_matched}  ({100 * n_matched / max(n_total, 1):.1f}%)")

    return merged


def balanced_negative_sampling(lifted_df, pval_thresh=5e-8, negative_thresh = 0.5, col="chr", ):
    """
    Sample negatives (p > threshold) in a stratified way following the
    chromosome distribution of positives (p <= threshold).
    """

    df = lifted_df.copy()

    # ---- Ensure p_value column exists ---- #
    if "p_value" not in df.columns:
        if "hm_p_value" in df.columns:
            df = df.rename(columns={"hm_p_value": "p_value"})
        else:
            raise ValueError("No p_value or hm_p_value column found")

    # ---- Ensure numeric p-values ---- #
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")

    # ---- Split classes ---- #
    minordf = df[(df["p_value"] <= pval_thresh) & (df["beta"] > 0)].copy()
    print(f'Number of positives {minordf.shape}, beta values mean = {minordf["beta"].mean()}')
    majordf = df[df["p_value"] > negative_thresh].copy()

    minordf['Target'] = 1
    majordf['Target'] = 0

    if minordf.empty:
        raise ValueError("No positive (minority) samples found")

    if majordf.empty:
        raise ValueError("No negative (majority) samples found")

    total_samples = len(minordf)

    # ---- Normalize chromosome column ---- #
    minordf[col] = minordf[col].astype(str)
    majordf[col] = majordf[col].astype(str)

    # ---- Chromosome proportions from positives ---- #
    class_proportions = minordf[col].value_counts(normalize=True)
    samples_per_class = (class_proportions * total_samples).round().astype(int)

    sampled = []

    for label, n_samples in samples_per_class.items():
        group = majordf[majordf[col] == label]

        if group.empty or n_samples == 0:
            continue

        sampled_group = resample(
            group,
            n_samples=n_samples,
            replace=n_samples > len(group),
            random_state=42
        )
        sampled.append(sampled_group)

    if not sampled:
        raise ValueError("Stratified sampling failed: no samples drawn")

    stratified_subset = pd.concat(sampled, ignore_index=True)
    outfile = pd.concat([stratified_subset, minordf])

    return outfile


def main():

    args = parse_args()

    dbsnp_path = "/home/francesco.gualdi/scratch/data/DBsnp/dbsnp_chunks/"
    studies_parsed_path = "../parsed_studies"
    outpath = "../lifted_vcfs/"

    os.makedirs(outpath, exist_ok=True)

    for efolder in os.listdir(studies_parsed_path):
        print(f"[WORKING ON] {efolder}")

        data = []
        efolder_path = os.path.join(studies_parsed_path, efolder)

        if not os.path.isdir(efolder_path):
            continue

        for study in os.listdir(efolder_path):
            if 'check' in study:
                continue
            study_path = os.path.join(efolder_path, study)

            print(f"[LOADING] {study}")
            study_df = load_parsed_associations(study_path)

            print(f"[LIFTING] {study}")
            lifted = lift_database(study_df, dbsnp_path)

            if lifted.empty:
                print(f"[WARNING] Empty output for {study}")
                continue

            print(f"[STRATIFYING] {study}")
            stratified_and_lifted = balanced_negative_sampling(lifted)

            data.append(stratified_and_lifted)

        if not data:
            print(f"[WARNING] No data for {efolder}, skipping")
            continue

        final = pd.concat(data)
        out_file = os.path.join(outpath, efolder + '_lifted_sampled.vcf')

        final = final[['chr', 'pos', 'ref', 'alt', 'hm_rsid', 'beta', 'p_value', 'study_id', 'sample_size', 'Target']]
        final = final.sample(frac=1)
        print(f'Number of positives {minordf.shape}, beta values mean = {minordf["beta"].mean()}')
        final.to_csv(out_file, index=False, sep='\t')
        print(f"[SAVED] {out_file}")


# =====================
# Entry point
# =====================
if __name__ == "__main__":
    main()
