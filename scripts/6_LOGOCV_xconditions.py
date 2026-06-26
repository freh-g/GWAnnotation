#!/usr/bin/env python3
import pickle
import sys
import hydra
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import numpy as np
import os
import random
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import accuracy_score, roc_curve, auc
import pickle as pkl

log = logging.getLogger(__name__)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Data helpers (unchanged from xgboost script)
# ---------------------------------------------------------------------------

def get_X(df: pd.DataFrame, feature_cols: list):
    """Return a numeric matrix for the given feature columns."""
    return df[feature_cols].to_numpy(dtype=np.float32)


def get_Y(df: pd.DataFrame, condition: str):
    return df[(condition + '_Target')]


def rm_intracondition_dupli(row: pd.Series, condition_cols: list):
    if row[condition_cols].sum() > 1:
        idxs = np.where(row[condition_cols].values == 1)[0]
        idx_condition_keep = np.random.choice(idxs)
        row[condition_cols] = 0
        row[condition_cols[idx_condition_keep]] = 1
    return row


def load_data(path: str) -> dict:
    log.info(f"Loading dataset '{path}'")
    data = pd.read_csv(path)
    print([a for a in data.columns.tolist() if 'study' in a])
    log.info(f"Loaded dataset of shape {data.shape}, memory usage "
             f"{data.memory_usage().sum() / 1024 ** 3:.2f}GB")
    is_condition_column = lambda c: str(c).startswith("condition_") and \
                                    not str(c).endswith("_Target")
    conditions = data.columns[data.columns.map(is_condition_column)]
    print(conditions)
    msg = f"Splitting dataset into {len(conditions)} conditions:"
    for condition in conditions:
        msg += f"\n\t- {condition}"
    log.info(msg)
    data = data.apply(rm_intracondition_dupli, axis=1, args=[conditions])
    assert data[conditions].sum(axis=1).sum() == data.shape[0]
    dfs = {}
    for condition in conditions:
        dfs[condition] = data[data[condition] == 1]
    log.info("Split completed")
    return dfs


def group_by_study(condition, df_condition: pd.DataFrame, mapping_dict: dict):
    study_cols = [col for col in df_condition.columns if col.startswith('study_id_')]
    study_groups = {}
    for col in study_cols:
        is_match = col.split('_')[-1] in list(mapping_dict[condition])
        if not is_match:
            continue
        col_data = df_condition[col]
        if col_data.sum() == 0:
            continue
        study_groups[col] = col_data[col_data == 1].index
    return study_groups


def remove_duplicates(study_groups: dict, df_condition: pd.DataFrame):
    """
    Resolve multiple study assignments per sample by randomly keeping one.
    Here we will assign randomly a variant that is in multiple studies to one of them, 
    so that we can then do a clean leave-one-study-out split.
    """
    study_cols = study_groups.keys()
    summed = df_condition[study_cols].sum(axis=1)
    mask = summed > 1
    to_drop = summed[mask]
    for i in to_drop.index:
        dupli_cols = np.array(list(study_cols))[df_condition[study_cols].loc[i] == 1]
        rnd_col = np.random.choice(dupli_cols, 1)[0]
        for col in dupli_cols:
            if col != rnd_col:
                df_condition.loc[i, col] = 0
    return df_condition


def study_leave_one_out(df: pd.DataFrame, condition: str, study_groups: dict):
    """
    Generate nested study-based leave-one-out splits.
    This function constructs a cross-validation scheme where studies are used
    as grouping units. For each study, the samples belonging to that study are
    used as the test set. Among the remaining studies, one study at a time is
    used as a validation set, while all other studies form the training set.
    """
    test_splits = []
    for k_test in study_groups:
        test_idxs = study_groups[k_test]
        if df.loc[test_idxs, f'{condition}_Target'].sum() < 50:
            continue
        val_splits = []
        for k_val in study_groups:
            if k_val == k_test:
                continue
            val_idxs = study_groups[k_val]
            tr_idxs_list = []
            for k_tr in study_groups:
                if k_tr == k_test or k_tr == k_val:
                    continue
                tr_idxs_list.extend(study_groups[k_tr])
            val_splits.append((tr_idxs_list, val_idxs))
        test_splits.append((test_idxs, val_splits))
    return test_splits


# ---------------------------------------------------------------------------
# Generic train / train_final functions (model-agnostic)
# ---------------------------------------------------------------------------

def train(dfs, dfs_splits, clf, condition: str, test_split: int, X_cols: list,
          auc_eval: bool = False):
    df = dfs[condition]
    scores = []
    fpr_ls, tpr_ls, thresholds_ls = [], [], []

    for split in dfs_splits[condition][test_split][1]:
        tr_idxs, val_idxs = split

        tr_X = get_X(df.loc[tr_idxs], X_cols)
        tr_y = get_Y(df.loc[tr_idxs], condition)
        tr_X, tr_y = np.array(tr_X), np.array(tr_y)
        clf.fit(tr_X, tr_y)

        val_X = get_X(df.loc[val_idxs], X_cols)
        val_y = get_Y(df.loc[val_idxs], condition)
        val_X = np.array(val_X)

        if auc_eval:
            pred_y = clf.predict_proba(val_X)
            if hasattr(pred_y, "get"):
                pred_y = pred_y.get()
            elif not isinstance(pred_y, np.ndarray):
                pred_y = pred_y.asnumpy()
            fpr, tpr, thresholds = roc_curve(val_y, pred_y[:, 1])
            score = auc(fpr, tpr)
            fpr_ls.append(fpr)
            tpr_ls.append(tpr)
            thresholds_ls.append(thresholds)
        else:
            pred_y = clf.predict(val_X)
            if hasattr(pred_y, "get"):
                pred_y = pred_y.get()
            score = accuracy_score(val_y, pred_y)

        scores.append(score)

    if auc_eval:
        return scores, fpr_ls, tpr_ls, thresholds_ls
    return scores


def train_final(dfs, dfs_splits, clf, condition: str, i_model: int, X_cols: list):
    df = dfs[condition]
    ts_idxs = dfs_splits[condition][i_model][0]
    tr_X = get_X(df.drop(index=ts_idxs), X_cols)
    tr_y = get_Y(df.drop(index=ts_idxs), condition)
    ts_X = get_X(df.loc[ts_idxs], X_cols)
    ts_y = get_Y(df.loc[ts_idxs], condition)

    tr_X, tr_y = np.array(tr_X), np.array(tr_y)
    clf.fit(tr_X, tr_y)

    ts_X = np.array(ts_X)
    pred_y = clf.predict_proba(ts_X)
    if not isinstance(pred_y, np.ndarray):
        pred_y = pred_y.asnumpy()

    fpr, tpr, thresholds = roc_curve(ts_y, pred_y[:, 1])
    score = auc(fpr, tpr)
    return score, fpr, tpr, thresholds


def train_all_studies(dfs_scaled, best_params, X_cols, model_factory):
    """
    Train one final model per condition using ALL available data (no held-out test).
    This is the model to use for external/new sample prediction.
    """
    final_models = {}
    for condition, params_list in best_params.items():
        df = dfs_scaled[condition]
        
        # Average hyperparams across folds for the final model
        # (or just use fold 0 — your choice)
        # Here we use the params from the fold with best AUC
        best_fold_params = params_list[0]  # or pick by best score
        
        clf = model_factory(best_fold_params)
        tr_X = get_X(df, X_cols)
        tr_y = get_Y(df, condition)
        clf.fit(np.array(tr_X), np.array(tr_y))
        
        final_models[condition] = {
            'model':     clf,
            'X_cols':    X_cols,
            'condition': condition,
            'n_samples': len(df),
        }
        print(f"[final] {condition}: trained on {len(df)} samples")
    
    return final_models

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(
        gamma=params['gamma'],
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_lambda=params['reg_lambda'],
        reg_alpha=params['reg_alpha'],
        scale_pos_weight=params['scale_pos_weight'],
        random_state=42,
        device='cuda',
    )


def make_lr(params: dict) -> LogisticRegression:

    penalties = ['l1', 'l2']
    penalty = penalties[params['penalty']] if isinstance(params['penalty'], (int, np.integer)) else params['penalty']

    return LogisticRegression(
        C=params['C'],
        penalty=penalty,
        solver='saga',
        max_iter=1000,
        random_state=42
    )


def make_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']) if params['max_depth'] is not None else None,
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        max_features=params['max_features'],
        random_state=42,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Hyperparameter spaces
# ---------------------------------------------------------------------------

XGB_SPACE = {
    'gamma':            hp.uniform('gamma', 0, 5),
    'max_depth':        hp.uniformint('max_depth', 3, 20),
    'learning_rate':    hp.uniform('learning_rate', 0.01, 0.3),
    'subsample':        hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'reg_lambda':       hp.uniform('reg_lambda', 0, 1),
    'reg_alpha':        hp.uniform('reg_alpha', 0, 1),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 0, 5),
}

LR_SPACE = {
    'C':       hp.loguniform('C', np.log(1e-4), np.log(1e2)),
    'penalty': hp.choice('penalty', ['l1', 'l2']),
}

RF_SPACE = {
    'n_estimators':     hp.uniformint('n_estimators', 50, 500),
    'max_depth':        hp.uniformint('max_depth', 3, 30),
    'min_samples_split':hp.uniformint('min_samples_split', 2, 20),
    'min_samples_leaf': hp.uniformint('min_samples_leaf', 1, 10),
    'max_features':     hp.choice('max_features', ['sqrt', 'log2']),
}


# ---------------------------------------------------------------------------
# Generic tuning loop
# ---------------------------------------------------------------------------

def make_objective(dfs, dfs_splits, X_cols, model_factory):
    """Return a hyperopt objective for any model family."""
    def objective(params):
        clf = model_factory(params)
        condition  = params['condition']
        test_split = params['test_split']
        scores = train(dfs, dfs_splits, clf, condition, test_split, X_cols)
        return {'loss': -np.mean(scores), 'status': STATUS_OK}
    return objective


def tuning(dfs, dfs_splits, space, X_cols, model_factory, max_evals: int = 10):
    best_params  = {}
    trials_logs  = {}
    obj = make_objective(dfs, dfs_splits, X_cols, model_factory)

    for k in dfs:
        k_best_params = []
        k_trials_logs = []
        for i_model in range(len(dfs_splits[k])):
            trials = Trials()
            run_space = dict(space)          # shallow copy so we don't pollute shared space
            run_space['condition']  = k
            run_space['test_split'] = i_model
            best = fmin(
                fn=obj,
                space=run_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.default_rng(42),
            )
            best = space_eval(run_space, best)
            print(f"[{k}] fold {i_model}: {best}")
            k_best_params.append(best)
            k_trials_logs.append(trials)
        best_params[k] = k_best_params
        trials_logs[k] = k_trials_logs

    return best_params, trials_logs


# ---------------------------------------------------------------------------
# Final evaluation loop (shared across all model types)
# ---------------------------------------------------------------------------

def evaluate_model(dfs, dfs_splits, best_params, X_cols, model_factory):
    """Train final models on all (train+val) folds and evaluate on held-out test."""
    metrics = {}
    for condition, models_params in best_params.items():
        scores, fpr_ls, tpr_ls, thresholds_ls = [], [], [], []
        condition_models = []
        for i_model, params in enumerate(models_params):
            clf = model_factory(params)
            score, fpr, tpr, thresholds = train_final(
                dfs, dfs_splits, clf, condition, i_model, X_cols
            )
            scores.append(score)
            fpr_ls.append(fpr)
            tpr_ls.append(tpr)
            thresholds_ls.append(thresholds)
            condition_models.append(clf)
            print(f"{condition} - fold {i_model}  AUC: {score:.4f}")
        metrics[condition] = {
            'scores':        scores,
            'fpr_ls':        fpr_ls,
            'tpr_ls':        tpr_ls,
            'thresholds_ls': thresholds_ls,
            'models':        condition_models,
            # 'study_names':   study_names,  

        }
    return metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="xgboost.yaml", version_base="1.2")
def main(cfg: DictConfig):
    os.makedirs('../outputs', exist_ok=True)
    with open('../data/efo_dict.pkl', 'rb') as f:
        efo_dict = pkl.load(f)
    efo_dict_new = {'condition_' + k.replace('_', ''): v for k, v in efo_dict.items()}

    # ---- Load & split data ----
    dfs = load_data(cfg["dataset"])

    dfs_study_groups = {}
    for k, dat in dfs.items():
        dfs_study_groups[k] = group_by_study(k, dat, efo_dict_new)

    for k in dfs:
        dfs[k] = remove_duplicates(dfs_study_groups[k], dfs[k])

    for k, dat in dfs.items():
        dfs_study_groups[k] = group_by_study(k, dat, efo_dict_new)
        lengths = [len(dfs_study_groups[k][j]) for j in dfs_study_groups[k]]
        print(f"{k}  length: {len(dfs[k])}  groups sum: {np.sum(lengths)}")

    dfs_splits = {k: study_leave_one_out(dfs[k], k, dfs_study_groups[k]) for k in dfs}

    # ---- Feature columns ----
    all_cols = dfs[next(iter(dfs))].columns
    to_drop = (
        [c for c in all_cols if c.startswith("ref_")]
        + [c for c in all_cols if c.startswith("alt_")]
        + [c for c in all_cols if c.startswith("study_id_")]
        # + [c for c in all_cols if c.startswith("chr_")]
        + [c for c in all_cols if c.startswith("condition_")]
        + ["hm_rsid", "FeatureID", "Length","Type_SNV", "RawScore","PHRED"]
    )
    X_cols = all_cols.drop(to_drop).tolist()
    print("Feature columns:", X_cols)

    # ====================================================================
    # 1.  XGBoost
    # ====================================================================
    print("\n" + "=" * 60)
    print("  XGBoost")
    print("=" * 60)
    xgb_best_params, xgb_logs = tuning(
        dfs, dfs_splits, XGB_SPACE, X_cols,
        model_factory=make_xgb,
        max_evals=10,
    )
    with open('../outputs/xgb_best_params.pkl', 'wb') as f:
        pkl.dump(xgb_best_params, f)
    with open('../outputs/xgb_trials_logs.pkl', 'wb') as f:
        pkl.dump(xgb_logs, f)

    xgb_metrics = evaluate_model(dfs, dfs_splits, xgb_best_params, X_cols, make_xgb)
    with open('../outputs/xgb_metrics.pkl', 'wb') as f:
        pkl.dump(xgb_metrics, f)

    # ====================================================================
    # 2.  Logistic Regression
    # ====================================================================
    print("\n" + "=" * 60)
    print("  Logistic Regression")
    print("=" * 60)
    lr_best_params, lr_logs = tuning(
        dfs, dfs_splits, LR_SPACE, X_cols,
        model_factory=make_lr,
        max_evals=10,
    )
    with open('../outputs/lr_best_params.pkl', 'wb') as f:
        pkl.dump(lr_best_params, f)
    with open('../outputs/lr_trials_logs.pkl', 'wb') as f:
        pkl.dump(lr_logs, f)

    lr_metrics = evaluate_model(dfs, dfs_splits, lr_best_params, X_cols, make_lr)
    with open('../outputs/lr_metrics.pkl', 'wb') as f:
        pkl.dump(lr_metrics, f)

    # ====================================================================
    # 3.  Random Forest
    # ====================================================================
    print("\n" + "=" * 60)
    print("  Random Forest")
    print("=" * 60)
    rf_best_params, rf_logs = tuning(
        dfs, dfs_splits, RF_SPACE, X_cols,
        model_factory=make_rf,
        max_evals=10,
    )
    with open('../outputs/rf_best_params.pkl', 'wb') as f:
        pkl.dump(rf_best_params, f)
    with open('../outputs/rf_trials_logs.pkl', 'wb') as f:
        pkl.dump(rf_logs, f)

    rf_metrics = evaluate_model(dfs, dfs_splits, rf_best_params, X_cols, make_rf)
    with open('../outputs/rf_metrics.pkl', 'wb') as f:
        pkl.dump(rf_metrics, f)

    with open('../outputs/logo_groups.pkl', 'wb') as f:
        pkl.dump(dfs_study_groups, f)
    
    with open('../outputs/dfs_splits.pkl', 'wb') as f:
        pkl.dump(dfs_splits, f)
    
    with open('../outputs/xcodition_matrix_index.pkl', 'wb') as f:
        pkl.dump(list(dfs.keys()), f)

    with open('../outputs/dfs_dict.pkl', 'wb') as f:
        pkl.dump(dfs, f)

    # ====================================================================
    # Cross condition evaluation
    # ====================================================================
    print("\n" + "=" * 60)
    print("  Evaluating across conditions")
    print("=" * 60)
    conditions = list(dfs.keys())
    n_conditions = len(conditions)

    mean_mtx = np.zeros((n_conditions, n_conditions), dtype=np.float32)
    std_mtx  = np.zeros((n_conditions, n_conditions), dtype=np.float32)

    metrics = xgb_metrics  # or lr_metrics / rf_metrics

    for tr_i, tr_k in enumerate(conditions):
        trained_models = metrics[tr_k]['models']  
        for ts_i, ts_k in enumerate(conditions):
            df_ts = dfs[ts_k]  
            aucs = []
            for i_model, model in enumerate(trained_models):
                for i_ts_split, ts_split in enumerate(dfs_splits[ts_k]):
                    if tr_i == ts_i and i_model != i_ts_split:
                        continue

                    ts_idxs = ts_split[0]
                    ts_X = get_X(df_ts.loc[ts_idxs], X_cols)
                    ts_y = get_Y(df_ts.loc[ts_idxs], ts_k)

                    pred_y = model.predict_proba(ts_X)
                    if not isinstance(pred_y, np.ndarray):
                        pred_y = np.array(pred_y)

                    fpr, tpr, thresholds = roc_curve(ts_y, pred_y[:,1])
                    auc_score = auc(fpr, tpr)
                    aucs.append(auc_score)

            mean_mtx[tr_i, ts_i] = np.mean(aucs)
            std_mtx[tr_i, ts_i]  = np.std(aucs)

    mean_df = pd.DataFrame(mean_mtx, index=conditions, columns=conditions)
    std_df  = pd.DataFrame(std_mtx, index=conditions, columns=conditions)

    mean_df.to_csv('../outputs/cross_condition_mean_auc.csv')
    std_df.to_csv('../outputs/cross_condition_std_auc.csv')
        
    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 60)
    print("  Summary (mean AUC per condition)")
    print("=" * 60)
    for label, metrics in [("XGB", xgb_metrics), ("LR", lr_metrics), ("RF", rf_metrics)]:
        for condition, m in metrics.items():
            mean_auc = np.mean(m['scores'])
            std_auc  = np.std(m['scores'])
            print(f"  [{label}]  {condition:<40}  AUC = {mean_auc:.4f} ± {std_auc:.4f}")


if __name__ == "__main__":
    main()