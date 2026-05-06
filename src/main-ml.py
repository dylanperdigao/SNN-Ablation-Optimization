import argparse
import pandas as pd
import torch
import time
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import imblearn
import os
import sys
import warnings
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, svm, neighbors, naive_bayes, tree, ensemble
from modules.datasets import BAF
from modules.metrics import metrics_performance_5fpr, metrics_fairness

PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(PATH, os.path.pardir)))
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

EPSILON = 1e-3
MODELS_SUPERVISED = ["LR", "DT", "RF", "XGB", "LGB", "CAT", "ADA", "NB", "SVM", "KNN"]
MODELS_UNSUPERVISED = ["KMeans", "GMM", "DBSCAN", "IF", "OCSVM"]
MODELS = MODELS_SUPERVISED + MODELS_UNSUPERVISED
DATASETS = ["BAF-Base", "BAF-TypeI", "BAF-TypeII", "BAF-TypeIII", "BAF-TypeIV", "BAF-TypeV"]

def load_dataset(dataset: str, root='./data', validation=False):
    if dataset == 'baf':
        variant = 'Base'
    elif 'baf' in dataset.lower():
        variant = dataset.split('-')[-1]
    else:
        raise ValueError("Invalid dataset")
    train_dataset = BAF(variant=variant, root=f"{root}/BAF", train=True, mode='train', validation=validation)
    test_dataset = BAF(variant=variant, root=f"{root}/BAF", train=False, mode='test', validation=validation)
    return train_dataset, test_dataset

def main(trial: optuna.Trial, dataset: str, model_name: str, undersampling: bool, seed: int) -> tuple:
    # ---------------------------------------------------------
    # Hyperparameter Suggestion Space
    # ---------------------------------------------------------
    if model_name == "LR":
        c = trial.suggest_float("C", 1e-4, 1e4, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga", "liblinear", "newton-cg", "newton-cholesky", "sag"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        model = linear_model.LogisticRegression(C=c, solver=solver, class_weight=class_weight, random_state=seed, max_iter=1000)
    elif model_name == "DT":
        max_depth = trial.suggest_int("max_depth", 2, 32)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        model = tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, class_weight=class_weight, random_state=seed)
    elif model_name == "RF":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])
        model = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, class_weight=class_weight, random_state=seed)
    elif model_name == "IF":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        model = ensemble.IsolationForest(n_estimators=n_estimators, random_state=seed)
    elif model_name == "XGB":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 100.0, log=True)
        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=scale_pos_weight, random_state=seed, eval_metric="logloss")
    elif model_name == "LGB":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 100.0, log=True)
        model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=scale_pos_weight, random_state=seed)
    elif model_name == "CAT":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        auto_class_weights = trial.suggest_categorical("auto_class_weights", ["None", "Balanced", "SqrtBalanced"])
        cw_param = None if auto_class_weights == "None" else auto_class_weights
        model = cb.CatBoostClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, auto_class_weights=cw_param, random_state=seed, verbose=0)
    elif model_name == "ADA":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1.0, log=True)
        model = ensemble.AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=seed)
    elif model_name == "NB":
        var_smoothing = trial.suggest_float("var_smoothing", 1e-10, 1e-3, log=True)
        model = naive_bayes.GaussianNB(var_smoothing=var_smoothing)
    elif model_name == "OCSVM":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        nu = trial.suggest_float("nu", 0.01, 0.5)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        model = svm.OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    elif model_name == "SVM":
        c = trial.suggest_float("C", 1e-3, 1e3, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True) if kernel in ["rbf", "poly"] else "scale"
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
        model = svm.SVC(C=c, kernel=kernel, gamma=gamma, class_weight=class_weight, random_state=seed, probability=True)
    elif model_name == "KNN":
        n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_int("p", 1, 2)
        model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    # ---------------------------------------------------------
    # Data Loading & Preprocessing
    # ---------------------------------------------------------
    train_dataset, test_dataset = load_dataset(dataset, validation=False)
    x_train = train_dataset.data.squeeze(1).numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.squeeze(1).numpy()
    y_test = test_dataset.targets.numpy()
    if undersampling:
        sampler = imblearn.under_sampling.RandomUnderSampler(random_state=seed, sampling_strategy='majority')
        x_train, y_train = sampler.fit_resample(x_train, y_train)
    df_test = pd.DataFrame(x_test, columns=test_dataset.features)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    # ---------------------------------------------------------
    # Training & Prediction
    # ---------------------------------------------------------
    time_train_start = time.time()
    if model_name in MODELS_UNSUPERVISED:
        model.fit(x_train_scaled)
    else:
        model.fit(x_train_scaled, y_train)
    time_train_end = time.time() 
    time_test_start = time.time()
    if model_name in ["IF", "OCSVM"]:
        y_hat = -model.decision_function(x_test_scaled)
    else:
        y_hat = model.predict_proba(x_test_scaled)[:, 1]
    time_test_end = time.time() 
    perf_metrics = metrics_performance_5fpr(y_test, y_hat)
    fair_age_metrics = metrics_fairness(df_test, y_test, y_hat, "customer_age", 50)
    fair_income_metrics = metrics_fairness(df_test, y_test, y_hat, "income", 0.5)
    fair_employment_metrics = metrics_fairness(df_test, y_test, y_hat, "employment_status", 3)
    trial.set_user_attr("@time train", time_train_end - time_train_start)
    trial.set_user_attr("@time test", time_test_end - time_test_start)
    trial.set_user_attr("@perf accuracy", perf_metrics["accuracy"])
    trial.set_user_attr("@perf precision", perf_metrics["precision"])
    trial.set_user_attr("@perf recall", perf_metrics["recall"])
    trial.set_user_attr("@perf f1_score", perf_metrics["f1_score"])
    trial.set_user_attr("@perf fpr", perf_metrics["fpr"])
    trial.set_user_attr("@perf tnr", perf_metrics["tnr"])
    trial.set_user_attr("@perf roc_auc", perf_metrics["roc_auc"])
    trial.set_user_attr("@perf pr_auc", perf_metrics["pr_auc"])
    trial.set_user_attr("@auc roc_fprs", str(perf_metrics["roc_fprs"]))
    trial.set_user_attr("@auc roc_tprs", str(perf_metrics["roc_tprs"]))
    trial.set_user_attr("@auc roc_thresholds", str(perf_metrics["roc_thresholds"]))
    trial.set_user_attr("@auc pr_precisions", str(perf_metrics["pr_precisions"]))
    trial.set_user_attr("@auc pr_recalls", str(perf_metrics["pr_recalls"]))
    trial.set_user_attr("@auc pr_thresholds", str(perf_metrics["pr_thresholds"]))
    trial.set_user_attr("@fair fpr_ratio_age", fair_age_metrics["fpr_ratio"])
    trial.set_user_attr("@fair fnr_ratio_age", fair_age_metrics["fnr_ratio"])
    trial.set_user_attr("@fair fpr_ratio_income", fair_income_metrics["fpr_ratio"])
    trial.set_user_attr("@fair fnr_ratio_income", fair_income_metrics["fnr_ratio"])
    trial.set_user_attr("@fair fpr_ratio_employment", fair_employment_metrics["fpr_ratio"])
    trial.set_user_attr("@fair fnr_ratio_employment", fair_employment_metrics["fnr_ratio"])
    return perf_metrics["recall"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize SNN on Neuromorphic Dataset")
    parser.add_argument("--model", type=str, required=True, choices=MODELS, help=f"Models to use for training; available: {MODELS}")
    parser.add_argument("--dataset", type=str, default="all", choices=DATASETS+["all"], help=f"Dataset to use for training; available: {DATASETS}")
    parser.add_argument("--undersampling", action="store_true", help="Whether to use undersampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset != "all" else DATASETS
    print(f"Optimizing {args.model} on datasets: {datasets} with undersampling={args.undersampling}")
    for dataset in datasets:
        study = optuna.create_study(
            directions=["maximize"],
            sampler=optuna.samplers.TPESampler(multivariate=True, seed=args.seed, n_startup_trials=5),
            study_name=f"{dataset}-{args.model}",
            storage=f"sqlite:///{PATH}/results/results.db",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: 
                main(trial, dataset, args.model.upper(), args.undersampling, args.seed), 
                    n_jobs=4,
                    n_trials=100,
                    callbacks=[MaxTrialsCallback(100, states=(TrialState.COMPLETE,))],
                    show_progress_bar=True
        )

"""
python src/main-ml.py --model LR
python src/main-ml.py --model DT
python src/main-ml.py --model RF 
python src/main-ml.py --model IF
python src/main-ml.py --model XGB 
python src/main-ml.py --model LGB 
python src/main-ml.py --model CAT 
python src/main-ml.py --model ADA
python src/main-ml.py --model NB 
python src/main-ml.py --model OCSVM
python src/main-ml.py --model SVM
python src/main-ml.py --model KNN
"""