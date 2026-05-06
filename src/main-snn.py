import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import optuna
import imblearn
import warnings
import os
import sys
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime
from modules.networks import CSNN, FFSNN
from modules.datasets import BAF
from modules.metrics import metrics_performance_5fpr, metrics_fairness

PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(PATH, os.path.pardir)))
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

DATASETS = ["BAF-Base", "BAF-TypeI", "BAF-TypeII", "BAF-TypeIII", "BAF-TypeIV", "BAF-TypeV"]
MODELS = ["CSNN", "FFSNN"]

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

def main(trial: optuna.Trial, dataset: str, model_name: str, epochs: int, steps: int, batch_size: int, slope: int, fixparams: bool, optweights: bool, optslopes: bool, undersampling: bool, seed: int, time: str, device_name: str, learn_betas: bool, learn_thresholds: bool) -> float:
    METRIC = "recall"
    MODEL_FILENAME = f"{PATH}/models/{time}-{trial.study.study_name}-t{int(trial.number)}.pth"
    # ---------------------------------------------------------
    # Hyperparameter Suggestion Space
    # ---------------------------------------------------------
    layers = 4 if model_name == "CSNN" else 3 if model_name == "FFSNN" else 0
    if optslopes:
        slopes = [trial.suggest_int(f"slope_{i}", 5, 50, step=1) for i in range(1, layers + 1)]
    else:
        slopes = [slope] * layers
    if not fixparams:
        thresholds = [trial.suggest_float(f"threshold_{i}", 0.1, 10.0, step=0.01) for i in range(1, layers + 1)]
        betas = [trial.suggest_float(f"beta_{i}", 0.1, 1.0, step=0.01) for i in range(1, layers + 1)]
    else:
        thresholds = [1.0] * layers
        betas = [0.9] * layers  
    if optweights:
        minority_class_weight = trial.suggest_float("minority_class_weight", 0.98, 0.99, log=True)
    # ---------------------------------------------------------
    # Data Loading and Preprocessing
    # ---------------------------------------------------------
    train_dataset, test_dataset = load_dataset(dataset, validation=False)
    x_train = train_dataset.data.squeeze(1).numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.squeeze(1).numpy()
    y_test = test_dataset.targets.numpy()
    df_test = pd.DataFrame(x_test, columns=test_dataset.features)
    if undersampling:
        sampler = imblearn.under_sampling.RandomUnderSampler(random_state=seed, sampling_strategy='majority')
        x_train, y_train = sampler.fit_resample(x_train, y_train)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    # ---------------------------------------------------------
    # Training and Prediction 
    # ---------------------------------------------------------
    train_tensor = TensorDataset(torch.FloatTensor(x_train_scaled), torch.LongTensor(y_train))
    test_tensor = TensorDataset(torch.FloatTensor(x_test_scaled), torch.LongTensor(y_test))
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    input_size = x_train_scaled.shape[1]
    device = torch.device(device_name)
    if optweights:
        class_weights = torch.tensor([1-minority_class_weight, minority_class_weight], dtype=torch.float32).to(device)
    else:
        num_fraud_samples = np.sum(y_train == 1)
        num_normal_samples = np.sum(y_train == 0)
        weight_normal = 1.0 / num_normal_samples if num_normal_samples > 0 else 1.0
        weight_fraud = 1.0 / num_fraud_samples if num_fraud_samples > 0 else 1.0
        class_weights = torch.tensor([weight_normal, weight_fraud], dtype=torch.float32).to(device)
        class_weights = class_weights / class_weights.sum() 
    if model_name == "CSNN":
        model = CSNN(input_size, betas=betas, thresholds=thresholds, slopes=slopes, learn_betas=learn_betas, learn_thresholds=learn_thresholds, device=device).to(device)
    elif model_name == "FFSNN":
        model = FFSNN(input_size, betas=betas, thresholds=thresholds, slopes=slopes, learn_betas=learn_betas, learn_thresholds=learn_thresholds, device=device).to(device)
    else:
        raise ValueError("Invalid model name")
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    patience = 5  
    epochs_no_improve = 0
    best_recall = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    min_epochs = 5
    max_epochs = epochs
    for epoch in range(max_epochs):
        model.train()
        losses = []
        train_p_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for data, target in train_p_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            spk_rec, _ = model(data, steps)
            pred_spikes = spk_rec.sum(dim=0) 
            loss = criterion(pred_spikes, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            train_p_bar.set_postfix(Loss=np.mean(losses)) 
        model.eval()
        y_hat = []
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(device), target.to(device)
                spk_rec, _ = model(data, steps)
                spike_counts = spk_rec.sum(dim=0)
                probabilities = F.softmax(spike_counts, dim=1)
                y_proba = probabilities[:, 1]
                y_hat.extend(y_proba.cpu().numpy())
            perf_metrics = metrics_performance_5fpr(y_test, y_hat)
            recall = perf_metrics["recall"]
            trial.report(recall, step=epoch)
            if recall > best_recall:
                best_recall = recall
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epoch >= min_epochs and epochs_no_improve >= patience:
                break
            if trial.should_prune():
                raise optuna.TrialPruned()
    trial.set_user_attr("best_epoch", epoch + 1 - epochs_no_improve if best_recall > 0 else 0)
    model.load_state_dict(best_model_state)
    model.eval()
    y_hat = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc="Final Best Model Inference"):
            data = data.to(device)
            spk_rec, _ = model(data, steps)
            spike_counts = spk_rec.sum(dim=0)
            probabilities = F.softmax(spike_counts, dim=1)
            y_proba = probabilities[:, 1]
            y_hat.extend(y_proba.cpu().numpy())
    perf_metrics = metrics_performance_5fpr(y_test, y_hat)
    fair_age_metrics = metrics_fairness(df_test, y_test, y_hat, "customer_age", 50)
    fair_income_metrics = metrics_fairness(df_test, y_test, y_hat, "income", 0.5)
    fair_employment_metrics = metrics_fairness(df_test, y_test, y_hat, "employment_status", 3)
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
    try:
        best_v = trial.study.best_value
        print(f"Best {METRIC} so far: {best_v*100:.2f}%")
        print(f"Current trial {METRIC}: {best_recall*100:.2f}%")
    except ValueError:
        print("First trial: No best value yet.")
        best_v = -1.0 
    if best_recall > best_v:
        os.makedirs(os.path.dirname(MODEL_FILENAME), exist_ok=True)
        torch.save(best_model_state, MODEL_FILENAME)
    return best_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize SNN on Neuromorphic Dataset")
    parser.add_argument("--exp", type=str, required=True, choices=["A", "B", "C", "D"], help="Experiment to run")
    parser.add_argument("--dataset", type=str, default="BAF-Base", choices=DATASETS, help=f"Dataset to use for training")
    parser.add_argument("--model", type=str, default="CSNN", choices=MODELS, help="Model to use for training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--slope", type=int, default=25, help="Fixed surrogate gradient slope value (used if --optslopes is False)")
    parser.add_argument("--fixparams", action="store_true", help="Fix betas and thresholds instead of letting Optuna optimize them")
    parser.add_argument("--optweights", action="store_true", help="Class weighting optimization")
    parser.add_argument("--optslopes", action="store_true", help="Whether to let Optuna optimize slopes of the spiking function")
    parser.add_argument("--undersampling", action="store_true", help="Apply undersampling to address class imbalance")
    parser.add_argument("--learn_betas", action="store_true", help="Let PyTorch learn betas via gradient descent")
    parser.add_argument("--learn_thresholds", action="store_true", help="Let PyTorch learn thresholds via gradient descent")
    parser.add_argument("--jobs", type=int, default=10, help="Number of parallel jobs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y%m%d"), help="Timestamp")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=args.seed, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=(
            f"{args.dataset}-{args.model}"
            f"{'-UNDRSMPL' if args.undersampling else ''}"
            f"-E{args.epochs}-B{args.batch}-T{args.steps}"
            f"{'-OPTs' if args.optslopes else f'-S{args.slope}'}"
            f"{'-OPTw' if args.optweights else ''}"
            f"{'-FIXP' if args.fixparams else ''}"
            f"{'-LB' if args.learn_betas else ''}"
            f"{'-LT' if args.learn_thresholds else ''}"
        ),         
        storage=f"sqlite:///{PATH}/results/results.db",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: main(
            trial, args.dataset, args.model, args.epochs, args.steps, args.batch, args.slope,
            args.fixparams, args.optweights, args.optslopes, args.undersampling, 
            args.seed, args.date, args.device, args.learn_betas, args.learn_thresholds
        ), 
        n_jobs=args.jobs,
        n_trials=100, 
        show_progress_bar=True
    )

######################
#=== EXPERIMENT A ===#
######################
"""
"""
###### A1
"""
python src/main-snn.py --exp A --model CSNN --fixparams
python src/main-snn.py --exp A --model FFSNN --fixparams
"""
###### A2
"""
python src/main-snn.py --exp A --model CSNN --fixparams --learn_betas --learn_thresholds
python src/main-snn.py --exp A --model FFSNN --fixparams --learn_betas --learn_thresholds
"""
###### A3
"""
python src/main-snn.py --exp A --model CSNN --fixparams --learn_betas
python src/main-snn.py --exp A --model FFSNN --fixparams --learn_betas
"""
###### A4
"""
python src/main-snn.py --exp A --model CSNN --fixparams --learn_thresholds
python src/main-snn.py --exp A --model FFSNN --fixparams --learn_thresholds
"""
###### A5
"""
python src/main-snn.py --exp A --model CSNN --device cuda:1
python src/main-snn.py --exp A --model FFSNN --device cuda:1
"""
###### A6
"""
python src/main-snn.py --exp A --model CSNN --learn_betas --learn_thresholds
python src/main-snn.py --exp A --model FFSNN --learn_betas --learn_thresholds 
"""
###### A7
"""
python src/main-snn.py --exp A --model CSNN --learn_betas
python src/main-snn.py --exp A --model FFSNN --learn_betas
"""
###### A8
"""
python src/main-snn.py --exp A --model CSNN --learn_thresholds
python src/main-snn.py --exp A --model FFSNN --learn_thresholds
"""
######################
#=== EXPERIMENT B ===#
######################
"""
python src/main-snn.py --exp B --model CSNN --fixparams --steps 1
python src/main-snn.py --exp B --model FFSNN --fixparams --steps 1 
python src/main-snn.py --exp B --model CSNN --fixparams --steps 5
python src/main-snn.py --exp B --model FFSNN --fixparams --steps 5 
python src/main-snn.py --exp B --model CSNN --fixparams --steps 10
python src/main-snn.py --exp B --model FFSNN --fixparams --steps 10 
python src/main-snn.py --exp B --model CSNN --fixparams --steps 20 
python src/main-snn.py --exp B --model FFSNN --fixparams --steps 20 
python src/main-snn.py --exp B --model CSNN --fixparams --steps 30 
python src/main-snn.py --exp B --model FFSNN --fixparams --steps 30 
python src/main-snn.py --exp B --model CSNN --fixparams --steps 40 
python src/main-snn.py --exp B --model FFSNN --fixparams --steps 40 
python src/main-snn.py --exp B --model CSNN --fixparams --steps 50 
python src/main-snn.py --exp B --model FFSNN --fixparams --steps 50 
"""
######################
#=== EXPERIMENT C ===#
######################
"""
python src/main-snn.py --exp C --model CSNN --fixparams --slope 5
python src/main-snn.py --exp C --model FFSNN --fixparams --slope 5
python src/main-snn.py --exp C --model CSNN --fixparams --slope 10
python src/main-snn.py --exp C --model FFSNN --fixparams --slope 10
python src/main-snn.py --exp C --model CSNN --fixparams --slope 20
python src/main-snn.py --exp C --model FFSNN --fixparams --slope 20
python src/main-snn.py --exp C --model CSNN --fixparams --slope 30
python src/main-snn.py --exp C --model FFSNN --fixparams --slope 30 
python src/main-snn.py --exp C --model CSNN --fixparams --slope 40
python src/main-snn.py --exp C --model FFSNN --fixparams --slope 40 
python src/main-snn.py --exp C --model CSNN --fixparams --slope 50 
python src/main-snn.py --exp C --model FFSNN --fixparams --slope 50 
"""
######################
#=== EXPERIMENT D ===#
######################
"""
python src/main-snn.py --exp D --model CSNN --fixparams --slope 5 --steps 10 --learn_thresholds --dataset BAF-Base
python src/main-snn.py --exp D --model CSNN --fixparams --slope 5 --steps 10 --learn_thresholds --dataset BAF-TypeI
python src/main-snn.py --exp D --model CSNN --fixparams --slope 5 --steps 10 --learn_thresholds --dataset BAF-TypeII
python src/main-snn.py --exp D --model CSNN --fixparams --slope 5 --steps 10 --learn_thresholds --dataset BAF-TypeIII
python src/main-snn.py --exp D --model CSNN --fixparams --slope 5 --steps 10 --learn_thresholds --dataset BAF-TypeIV
python src/main-snn.py --exp D --model CSNN --fixparams --slope 5 --steps 10 --learn_thresholds --dataset BAF-TypeV
python src/main-snn.py --exp D --model FFSNN --fixparams --slope 30 --steps 10 --learn_thresholds --dataset BAF-Base
python src/main-snn.py --exp D --model FFSNN --fixparams --slope 30 --steps 10 --learn_thresholds --dataset BAF-TypeI
python src/main-snn.py --exp D --model FFSNN --fixparams --slope 30 --steps 10 --learn_thresholds --dataset BAF-TypeII
python src/main-snn.py --exp D --model FFSNN --fixparams --slope 30 --steps 10 --learn_thresholds --dataset BAF-TypeIII
python src/main-snn.py --exp D --model FFSNN --fixparams --slope 30 --steps 10 --learn_thresholds --dataset BAF-TypeIV
python src/main-snn.py --exp D --model FFSNN --fixparams --slope 30 --steps 10 -—learn_thresholds -—dataset BAF-TypeV
"""