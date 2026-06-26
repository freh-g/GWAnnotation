# Variant Annotation Pipeline for Complex Disorder Association Prediction

A scalable and reproducible pipeline for generating machine learning-ready datasets of genetic variants associated with complex disorders using **GWAS summary statistics** and **CADD functional annotations**.

The pipeline automatically retrieves GWAS studies, maps variants to the latest reference genome, performs balanced negative sampling, annotates variants with CADD features, and produces curated datasets suitable for downstream statistical and machine learning analyses.

---

## Features

- Retrieve GWAS studies from the GWAS Catalog using EFO terms
- Optional filtering by ancestry
- Map variants to the GRCh38 reference genome
- Prevent data leakage by assigning duplicated variants to a single study
- Chromosome-stratified negative sampling for balanced datasets
- Annotate variants using CADD
- Automatic preprocessing:
  - one-hot encoding of categorical features
  - missing value imputation
  - feature scaling
  - duplicate merging
- Generate machine learning-ready datasets

---

## Pipeline Overview

The pipeline consists of the following steps:

1. Retrieve GWAS summary statistics
2. Extract significant and non-significant variants
3. Map rsIDs to GRCh38 coordinates
4. Perform chromosome-balanced negative sampling
5. Annotate variants with CADD
6. Merge studies and disease labels
7. Handle missing values
8. Encode categorical variables
9. Scale numerical fxt


## Input

The pipeline expects a dictionary mapping **Experimental Factor Ontology (EFO)** identifiers to the corresponding **GWAS Catalog Study (GCST)** IDs.

Example:

```python
studies = {
    "EFO_0001645": [
        "GCST90132315",
        "GCST90132314",
        "GCST003116"
    ]
}
---

## Data Sources

- GWAS Catalog
- dbSNP
- CADD

---

## License

This project is released under the MIT License.

---

## Contact

Francesco Gualdi

Dalle Molle Institute for Artificial Intelligence (IDSIA), USI-SUPSI

Swiss Institute of Bioinformatics (SIB)

Email: francesco.gualdi@supsi.ch