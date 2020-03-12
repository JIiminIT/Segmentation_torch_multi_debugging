"""Image segmentation evaluation 시에 사용되는 함수들의 모듈"""
import pandas as pd
import numpy as np


__all__ = ["get_eval_metrics", "report_metrics"]


def get_eval_metrics(pred, label, metrics="dicejaccaccr"):
    """Calculate segmentation evaluation metrics
    
    Args:
        pred (np.ndarray): prediction to be measured
        label (np.ndarray): ground truth
        metrics (str, optional): calculates 'dice', 'jacc', 'accr' if included.
    
    Returns:
        tuple: dice, jaccard, accuracy
    """

    intersection = np.sum(pred * label)
    union = np.sum(pred) + np.sum(label)
    eps = 0.00000001

    result = {}

    if "dice" in metrics:
        dice = 2 * intersection / (union + eps)
        if not union:  # Let's calculate dice as 1 if there is no union
            dice = 1
        result["dice"] = dice

    if "jacc" in metrics:
        jaccard = intersection / (union - intersection + eps)
        if not union:  # Let's calculate jaccard as 1 if there is no union
            jaccard = 1
        result["jacc"] = jaccard

    if "accr" in metrics:
        accuracy = np.sum(pred == label) / (pred.size + eps)
        result["accr"] = accuracy

    return result


def report_metrics(pred, label, data_id, metrics="dicejaccaccr", n_classes=9):
    """Makes a pandas dataframe object of multi-class
    segmentation evaluation metrics.
    
    Args:
        pred (np.ndarray): [description]
        label (np.ndarray): [description]
        data_id (str): filename or subject id used to identify
        metrics (str, optional): calculates 'dice', 'jacc', 'accr' if included.
        n_classes (int, optional): Defaults to 9. number of classes
    
    Returns:
        pandas.DataFrame: concatenated DataFrame object
    """

    dice_dict = {"total": 0}
    jacc_dict = {"total": 0}
    accr_dict = {"total": 0}
    col_list = ["total"]

    # Check if two arrays have the same shape
    assert pred.shape == label.shape
    assert metrics  # Check if metrics is not empty

    for class_n in range(n_classes):
        result = get_eval_metrics(
            (pred == class_n), (label == class_n), metrics=metrics
        )
        name_col = "label_%d" % class_n
        col_list.append(name_col)

        if "dice" in metrics:
            dice_dict[name_col] = result["dice"]
            dice_dict["total"] += result["dice"] / n_classes
        if "jacc" in metrics:
            jacc_dict[name_col] = result["jacc"]
            jacc_dict["total"] += result["jacc"] / n_classes
        if "accr" in metrics:
            accr_dict[name_col] = result["accr"]

    # Label 별로 따로 계산하면 배경(너무 많은 비중을 차지)을 포함하기 때문에 전체에서 계산하는게 맞다.
    if "accr" in metrics:
        accr_dict["total"] += np.sum(pred == label) / pred.size

    # TODO: 매 번 새 데이터 프레임들을 만들고 concatenate하는 과정이 느림. 속도 개선이 필요할 것으로 보임.
    dfs = []

    if "dice" in metrics:
        dfs.append(
            pd.DataFrame(dice_dict, index=[[data_id], ["dice"]], columns=col_list)
        )
    if "jacc" in metrics:
        dfs.append(
            pd.DataFrame(jacc_dict, index=[[data_id], ["jaccard"]], columns=col_list)
        )
    if "accr" in metrics:
        dfs.append(
            pd.DataFrame(accr_dict, index=[[data_id], ["accuracy"]], columns=col_list)
        )

    total_df = pd.concat(dfs)

    return total_df
