"""Metrics"""
import torch


def get_precision(PR, GT, threshold=0.5):
    PR = PR > threshold
    GT = GT == torch.max(GT)
    TP = ((PR == 1) & (GT == 1))
    FP = ((PR == 1) & (GT == 0))
    Precision = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FP)) + 1e-6)
    return Precision


def get_recall(PR, GT, threshold=0.5):
    PR = PR > threshold
    GT = GT == torch.max(GT)
    TP = ((PR == 1) & (GT == 1))
    FN = ((PR == 0) & (GT == 1))
    Recall = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FN)) + 1e-6)
    return Recall


def get_fbetascore(PR, GT, threshold=0.5, beta=0.5):
    Recall = get_recall(PR, GT, threshold)
    Precision = get_precision(PR, GT, threshold)
    FBetaScore = (1 + beta**2)*(Precision*Recall)/(Precision * beta**2 + Recall + 1e-6)
    return FBetaScore
