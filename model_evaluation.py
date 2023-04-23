def evaluate(model, X, y):
    """
    Evaluate the performance of a classification model using various metrics.
    
    Args:
    model: A trained classification model object with `predict()` and `predict_proba()` methods.
    X: Input features for prediction.
    y: Target variable for prediction.
    
    Returns:
    A dictionary with evaluation metrics including accuracy, precision, recall, f1-score, AUC-ROC, AUC-PRC, and confusion matrix.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    prc_auc = average_precision_score(y, y_prob)
    conf_matrix = confusion_matrix(y, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'AUC-ROC': roc_auc,
        'AUC-PRC': prc_auc,
        'confusion_matrix': conf_matrix
    }
    
    return metrics