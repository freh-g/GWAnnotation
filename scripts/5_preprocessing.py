#!/usr/bin/env python3

"""
This script perform the preprocessing of the data. As input it takes the
directory containing the annotated efo tsv files. The output is the preprocessed
and merged data into a single csv file.

Usage:
    python preprocessing.py
    python preprocessing.py data=<annotated_efos_dir>
"""
import logging
from pathlib import Path
import json
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


log = logging.getLogger(__name__)
np.random.seed(42)

def DataConverter(data):
    """
        	This function takes as input the data not onehotencoded and converts the column types in the right ones
    
    """
    for col in data.columns.tolist():
        try:
            data[col] = data[col].astype(float)
        except:
            data[col] = data[col].astype(str)
    return data


def ColumnsRecognizer(data):
    """
        	Iterate over columns of the data and recognize the categorical and numerical columns
            
    """
    
    # Select categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    # Select numerical columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns

    return numerical_columns, categorical_columns

def Encoder(data,numerical_columns,categorical_columns):
    
    # Definizione dei trasformatori
    categorical_transformer = OneHotEncoder()
    continuous_transformer = StandardScaler()
    
    # Creazione del ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('cont', continuous_transformer, numerical_columns)
        ]
    )
    
    # Fit e trasformazione del dataset
    transformed_data = preprocessor.fit_transform(data)
    
    # Ottenimento dei nuovi nomi di colonne
    cat_columns = preprocessor.transformers_[0][1].get_feature_names_out(categorical_columns)
    cont_columns = numerical_columns
    all_columns = np.concatenate([cat_columns, cont_columns])
    
    # Creazione di un DataFrame con i dati trasformati
    # Convertiamo `transformed_data` in un array denso se è una matrice sparsa
    if hasattr(transformed_data, "toarray"):  # Controlla se `transformed_data` è una matrice sparsa
        transformed_data = transformed_data.toarray()
    
    transformed_df = pd.DataFrame(transformed_data, columns=all_columns)
    
    return transformed_df

def load_data(data_folder: str) -> pd.DataFrame:
    """
    From a folder containing tsv files for each efo, creates a single dataset
    and returns it.
    
    :param data_folder: Path to folder containing tsv for each EFO
    :type data_folder: str
    :return: Loaded data
    :rtype: DataFrame
    """
    log.info("Loading data ...")
    data_folder_path = Path(data_folder)
    df_efo_ls = []
    for file_path in data_folder_path.iterdir():
        if 'check' not in str(file_path):
            log.info(f"Loading file {file_path}")
            df = pd.read_csv(file_path, delimiter='\t')
            efo = "".join(file_path.name.split("_")[:2])
            df['condition'] = [efo]*len(df)
            if 'Unnamed: 0' in df.columns[0]:
                df = df.drop(columns=['Unnamed: 0'])
            df_efo_ls.append(df)
            log.info(f"Loaded {efo} dataset")
    return pd.concat(df_efo_ls)

def drop_missing(data: pd.DataFrame, miss_rate: float = 0.4) -> pd.DataFrame:
    """
    Drops features that have more than 'miss_rate' missing rate.

    :param data: dataset to drop missing features
    :type data: DataFrame
    :param miss_rate: threshold of missing rate
    :type miss_rate: float
    :return: Filtered dataset
    :rtype: DataFrame
    """
    new_data = pd.DataFrame()
    missing_ratios = (data.isna().sum() + data.isin(['nan']).sum()) / \
        data.shape[0]
    drop_mask = missing_ratios > miss_rate
    drop_features = data.columns[drop_mask]
    action = ["drop" if is_to_drop else "impute" for is_to_drop in drop_mask]
    new_data = data.drop(columns=drop_features)
    missing_ratio = pd.DataFrame({
        "feature": missing_ratios.index,
        "missing ratio": missing_ratios.values,
        "action": action
    })
    def format_ratio(value):
        res = "{:.1%}".format(value)
        res = res.replace("%","\\%")
        return res
    missing_ratio = missing_ratio[missing_ratio["missing ratio"] > 0]
    missing_ratio = missing_ratio.sort_values("missing ratio", ascending=False)
    latex_tab = missing_ratio.to_latex(escape=True, index=False, longtable=True,
    formatters={"missing ratio": format_ratio})
    log.info(f"Dropped {len(drop_features)} features")
    log.info(f"Features dropped: \n {drop_features.values}")
    latex_tab_path = Path(HydraConfig.get().runtime.output_dir, 
        "./missing_features.tex")
    with open(latex_tab_path, "w") as file:
        file.write(latex_tab)
    log.info(f"Saved missing features table in '{latex_tab_path.absolute()}'")
    return new_data

def impute(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing data. Continuos value are mean imputed while categorical have
    fixed imputation to 'undeifned' category.
    
    :param data: Data to impute
    :type data: pd.DataFrame
    :return: Imputed data
    :rtype: DataFrame
    """
    n_missing = (data.isna().sum() + data.isin(['nan']).sum()).sum()
    log.info(f"Missing data before imputation : {n_missing}")
    # impute missing categorical with 'undefined'
    rep_val = 'undefined'
    cat_dataset = data.select_dtypes(object)
    cat_missing_count = (cat_dataset.isna() | cat_dataset.isin(['nan'])).sum()
    log.info(f"Categorical missing count:\n {cat_missing_count}")
    cat_missing_cols = cat_missing_count.index[cat_missing_count > 0]
    cat_missing_dataset = cat_dataset[cat_missing_cols]
    cat_missing_dataset = cat_missing_dataset.replace(np.nan, rep_val)
    cat_missing_dataset = cat_missing_dataset.replace('nan', rep_val)
    data[cat_missing_dataset.columns] = cat_missing_dataset
    # impute missing continuos with the mean
    cont_dataset = data.select_dtypes(float)
    cont_missing_count = (cont_dataset.isna() | cont_dataset.isin(['nan'])).sum()
    cont_missing_cols = cont_missing_count.index[cont_missing_count > 0]
    log.info(f"Continuos missing features:\n {cont_missing_cols.values}")
    cont_missing_dataset = cont_dataset[cont_missing_cols]
    cont_missing_means = cont_missing_dataset.mean()
    imputed_dataset = pd.DataFrame()
    for col in cont_missing_dataset.columns:
        replaced_col = cont_missing_dataset[col].replace(np.nan, cont_missing_means[col])
        replaced_col = replaced_col.replace('nan', cont_missing_means[col])
        imputed_dataset[col] = replaced_col
    data[imputed_dataset.columns] = imputed_dataset
    n_missing = (data.isna().sum() + data.isin(['nan']).sum()).sum()
    log.info(f"Missing data after imputation : {n_missing}")
    return data





def one_hot_encode(data: pd.DataFrame, columns) -> pd.DataFrame:
    """
    Perform one-hot encoding on given data and set of feature's names.
    At the end, specified columns are dropped keeping only their one-hot encoded
    version. All other columns remain untouched.
    
    :param data: Data to perfrom one-hot encoding
    :type data: pd.DataFrame
    :param columns: Feature's names (columns) to perform one-hot encoding
    :return: Data with specified one-hot encoded columns
    :rtype: DataFrame
    """
    log.info("Saving study_to_condition dictionary")
    study_to_condition = {}
    for _, row in data.iterrows():
        if row['study_id'] not in study_to_condition.keys():
            study_to_condition[row['study_id']] = row['condition']
    with open(Path(HydraConfig.get().runtime.output_dir, 
        './study_to_condition.json'), "w") as file:
        json.dump(study_to_condition, file)
    log.info("Saving condition_to_studies dictionary")
    condition_to_studies = {}
    for study in study_to_condition:
        if study_to_condition[study] not in condition_to_studies.keys():
            condition_to_studies[study_to_condition[study]] = [study]
        else:
            condition_to_studies[study_to_condition[study]].append(study)
    with open(Path(HydraConfig.get().runtime.output_dir, 
        './conditions_to_studies.json'), "w") as file:
        json.dump(condition_to_studies, file)
    encoder = OneHotEncoder()
    to_encode_data = data[columns]
    # don't know why, but some categorical have mixed types int and str
    to_encode_data = to_encode_data.astype(str)
    encoded_array = encoder.fit_transform(to_encode_data)
    if (issparse(encoded_array)):
        encoded_array = encoded_array.toarray()
    feature_names = encoder.get_feature_names_out(to_encode_data.columns)
    encoded_data = pd.DataFrame(encoded_array, columns=feature_names, index=data.index)
    new_data = data.drop(columns=columns)
    new_data = pd.concat([new_data, encoded_data], axis=1)
    return new_data

# def scale(data: pd.DataFrame, columns) -> pd.DataFrame:
#     """
#     Scales the given columns of data with standard scaling approach.
#     All other columns remain untouched.

#     :param data: The data to scale
#     :type data: pd.DataFrame
#     :param columns: The columns that will be scaled in data
#     :return: The data with the specified columns scaled
#     :rtype: DataFrame
#     """
#     log.info("Scaling data with standard scaler")
#     scaler = StandardScaler()
#     to_scale_data = data[columns]
#     scaled_array = scaler.fit_transform(to_scale_data)
#     new_data = data.copy()
#     new_data[columns] = scaled_array
#     return new_data


# def scale(data: pd.DataFrame, columns, save_path='scaler.pkl') -> tuple[pd.DataFrame, StandardScaler]:
#     scaler = StandardScaler()
#     scaled_array = scaler.fit_transform(data[columns])
#     new_data = data.copy()
#     new_data[columns] = scaled_array
    
#     # save scaler
#     import joblib
#     joblib.dump(scaler, save_path)
    
#     return new_data, scaler

    
    
def one_hot_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes the target column. From a dataset with one-hot encoded
    columns that indicates the condtion for which the variant has been studied
    on, the target column will be one-hot encoded but assigned NaN when the 
    relative condition column has value 0.
    
    :param data: The data to one-hot encode the target
    :type data: pd.DataFrame
    :return: The data with the encoded target
    :rtype: DataFrame
    """
    key = [col for col in data.columns if str(col).startswith('chr_')]
    key.extend([col for col in data.columns if str(col).startswith('alt_')])
    key.extend([col for col in data.columns if str(col).startswith('ref_')])
    key.append('pos')
    n_duplicates = data[key].duplicated(keep='first').sum()
    log.info(f"Total duplicates due to presence in multiple studies and" +
        f" annotations duplication: {n_duplicates}")
    based_on = [col for col in data.columns if str(col).startswith('condition_')]
    new_data = pd.DataFrame()
    target_col = data['Target']
    checksum = data['Target'].sum()
    log.info("One hot of the targets")
    for on in tqdm(based_on):
        new_target_col = np.full(data.shape[0], np.nan, dtype=float)
        on_col = data[on]
        for i in range(len(on_col)):
            if on_col.iloc[i] == 1:
                new_target_col[i] = target_col.iloc[i]
        new_data[(on+'_Target')] = new_target_col
    new_data.index = data.index
    new_data = pd.concat([data.drop(columns=['Target']),new_data],axis=1)
    log.info(f"Target checksum before: {checksum}")
    after_checksum = new_data.iloc[:,-5:].sum().sum()
    log.info(f"Target checksum after: {after_checksum}")
    assert checksum == after_checksum
    return new_data

def merge(data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the duplicate variants. The variants have the same annotation except
    for the features 'Consuequence', 'study_id_*', 'condition_*', '*_Target'
    and 'Roulette-FILTER*' (all binary columns from one-hot encoding).
    This values in a columns for the duplicate entries are merged into a single 
    value by following the rules:
        - if is not a binary column to be merged, then return the first value;
            since all values in the column are the same 
        - assign 1 if there is at least a 1
        - assign 0 if there is no 1 but at least a 0
        - assign Nan if there are only Nan
    Nan are present in the target columns when the variant has no target for
    that condition, meaning that the relative condition column is 0. 
    
    :param data: The data to be merged
    :type data: pd.DataFrame
    :return: The merged data
    :rtype: DataFrame
    """
    log.info("Merging data over the rows with same key")
    key = [col for col in data.columns if str(col).startswith('chr_')]
    key.extend([col for col in data.columns if str(col).startswith('alt_')])
    key.extend([col for col in data.columns if str(col).startswith('ref_')])
    key.append('pos')
    log.info(f"Merge key that identifies one variation: {key}")
    to_merge = [col for col in data.columns if str(col).startswith('Consequence_')]
    to_merge.extend([col for col in data.columns if str(col).startswith('study_id_')])
    to_merge.extend([col for col in data.columns if str(col).startswith('condition_')])
    to_merge.extend([col for col in data.columns if str(col).endswith('_Target')])
    to_merge.extend([col for col in data.columns if str(col).startswith('Roulette-FILTER')])
    log.info(f"Features that will be merged: {to_merge}")
    dupli_mask = data[key].duplicated(keep=False)
    agg_dict = dict()
    for c in data.columns.values:
        agg_dict[c] = 'first'
    def merge_fnc(x):
        not_nan_x = [xi for xi in x if not np.isnan(xi)]
        if len(not_nan_x) > 0:
            if np.sum(not_nan_x) > 0:
                return 1
            else:
                return 0
        else:
            return np.nan
    for c in to_merge:
        agg_dict[c] = merge_fnc
    log.info("Merging ... (will take a while)")
    merged = data[dupli_mask].groupby(key).agg(agg_dict)
    log.info(f'Number of obtained merged entries: {len(merged)}')
    data = data.drop_duplicates(key,keep=False)
    data = pd.concat([data, merged], axis=0)
    data.index = range(len(data))
    return data

def save(data: pd.DataFrame):
    """
    Saves the dataframe into the output directory of Hydra in a single csv file.
    
    :param data: Data to save
    :type data: pd.DataFrame
    """
    path = '../preprocessed_gwas.csv'
    log.info(f"Saving data into '{path}'")
    data.to_csv(path, index=False)

@hydra.main(config_path="conf", config_name="preprocessing.yaml", 
    version_base="1.2")
def main(cfg: DictConfig):
    data = load_data(cfg['data'])
    data['chr'] = data['chr'].astype(str)
    print(data.shape)
    data = drop_missing(data=data, miss_rate=0.4)
    data = impute(data=data)
    # dropping ConsScore, ConsDetail, beta, p_value and sample_size
    data = data.drop(columns=['ConsScore','ConsDetail','beta','p_value','sample_size'])
    try:
        data = data.drop(columns=['GeneID']) # Maybe was dropped with the missing data part

    except:
        pass
    cont, cat = ColumnsRecognizer(data)
    # removing Target, Pos from continuos features to be scaled
    cont = cont.drop(['Target', 'pos'])
    # remove rsid, FeatureID  from categorical features to be one hot encoded
    cat = cat.drop(['hm_rsid','FeatureID'])
    data = one_hot_encode(data=data, columns=cat)
    # data = scale(data=data, columns=cont)
    data = one_hot_target(data=data)
    data = merge(data=data)
    save(data=data)
    log.info("Preprocess completed succesfully")

if __name__ == "__main__":
    main()