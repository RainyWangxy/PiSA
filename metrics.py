import numpy as np
from sklearn.metrics import average_precision_score

def check_inputs(targs, preds):
    
    '''
    Helper function for input validation.
    '''
    assert (np.shape(preds) == np.shape(targs))
    assert type(preds) is np.ndarray
    assert type(targs) is np.ndarray
    assert (np.max(preds) <= 1.0) and (np.min(preds) >= 0.0)
    assert (np.max(targs) <= 1.0) and (np.min(targs) >= 0.0)
    assert (len(np.unique(targs)) <= 2)

def compute_avg_precision(targs, preds):
    
    '''
    Compute average precision.
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    '''
    
    check_inputs(targs,preds)
    
    if np.all(targs == 0):
        # If a class has zero true positives, we define average precision to be zero.
        metric_value = 0.0
    else:
        metric_value = average_precision_score(targs, preds)
    
    return metric_value

def compute_precision_at_k(targs, preds, k):
    
    '''
    Compute precision@k. 
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    k: Number of predictions to consider.
    '''
    
    check_inputs(targs, preds)
    
    classes_rel = np.flatnonzero(targs == 1)
    if len(classes_rel) == 0:
        return 0.0
    
    top_k_pred = np.argsort(preds)[::-1][:k]
    
    metric_value = float(len(np.intersect1d(top_k_pred, classes_rel))) / k
    
    return metric_value

def compute_recall_at_k(targs, preds, k):
    
    '''
    Compute recall@k. 
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    k: Number of predictions to consider.
    '''
    
    check_inputs(targs,preds)
    
    classes_rel = np.flatnonzero(targs == 1)
    if len(classes_rel) == 0:
        return 0.0
    
    top_k_pred = np.argsort(preds)[::-1][:k]
    
    metric_value = float(len(np.intersect1d(top_k_pred, classes_rel))) / len(classes_rel)
    
    return metric_value

def compute_metrics_all(y_pred, y_true):
    # Given predictions and labels, compute a few metrics.
    num_examples, num_classes = np.shape(y_true)
    results = {}
    average_precision_list = []
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true = np.array(y_true == 1, dtype=np.float32)  # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        average_precision_list.append(compute_avg_precision(y_true[:, j], y_pred[:, j]))
    results['map'] = 100.0 * float(np.mean(average_precision_list))
    targets = y_true
    preds = (y_pred > 0.5) * 1
    tp = np.zeros(targets.shape[1])
    fp = np.zeros(targets.shape[1])
    fn = np.zeros(targets.shape[1])
    tn = np.zeros(targets.shape[1])
    # measure accuracy and record loss
    for idx in range(targets.shape[0]):
        pred = preds[idx]
        target = targets[idx]
        tp += ((pred + target) == 2) * 1
        fp += ((pred - target) == 1) * 1
        fn += ((pred - target) == -1) * 1
        tn += ((pred + target) == 0) * 1
        # measure elapsed time
    p_c = [(tp[i] / (tp[i] + fp[i])) * 100.0 if tp[i] > 0 else 0.0
           for i in range(len(tp))]
    r_c = [(tp[i] / (tp[i] + fn[i])) * 100.0 if tp[i] > 0 else 0.0
           for i in range(len(tp))]
    p_c_mean = sum(p_c) / targets.shape[1]
    r_c_mean = sum(r_c) / targets.shape[1]
    # f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for i in range(len(tp))]
    if p_c_mean + r_c_mean == 0:
        f_c_mean = 0
    else:
        f_c_mean = 2 * p_c_mean * r_c_mean / (p_c_mean + r_c_mean)
    p_o = sum(tp) / sum(tp + fp) * 100.0
    r_o = sum(tp) / sum(tp + fn) * 100.0
    if p_o + r_o == 0:
        f_o = 0
    else:
        f_o = 2 * p_o * r_o / (p_o + r_o)
    results['of'] = f_o
    results['cf'] = f_c_mean
    return results


def compute_ap_divide(y_pred, y_true):
    
    """
    returns the metrics for each class separately.
    """
    num_examples, num_classes = np.shape(y_true)
    results = {}
    average_precision_list = []
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true = np.array(y_true == 1, dtype=np.float32)  # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        average_precision_list.append(compute_avg_precision(y_true[:, j], y_pred[:, j])*100.0)
    return average_precision_list
        # measure elapsed time
def select_nozero_class(targets):
    """
    处理样本数为0的类别
    """
    # 计算每个类别的样本数
    class_counts = np.sum(targets, axis=0)
    # 找到样本数为0的类别索引
    nozero_sample_classes = np.where(class_counts != 0)[0]
    # all_classes = np.arange(targets.shape[1])
    return nozero_sample_classes