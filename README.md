#  A Pipeline for Predicting Variant Associations with Complex Disorders via Functional Annotations

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
```
---

## Data Sources

- GWAS Catalog (https://www.ebi.ac.uk/gwas/)
- dbSNP (https://www.ncbi.nlm.nih.gov/snp/)
- CADD (https://cadd.bihealth.org/)

---

## License

This project is released under the MIT License.

---

## Contact

Francesco Gualdi

Dalle Molle Institute for Artificial Intelligence (IDSIA), USI-SUPSI

Swiss Institute of Bioinformatics (SIB)

Email: francesco.gualdi@supsi.ch