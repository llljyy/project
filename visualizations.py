def confusion_matrix(y_true, y_pred):
    """
    Plots a confusion matrix using seaborn.

    Parameters:
    y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred: array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    Returns:
    None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix


    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='Greens')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    plt.show()


def roc_curve(y_true, y_pred):
    """
    Plots the ROC curve for binary classification given the true labels and predicted probabilities.

    Parameters:
    y_true: array-like, shape (n_samples,)
        True binary labels in {0, 1}

    y_pred: array-like, shape (n_samples,)
        Target scores, can either be probability estimates of the positive class or confidence values.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    # Calculate the false positive rate (fpr), true positive rate (tpr), and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Calculate the area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()


def precision_recall_curve(y_true, y_pred):
    """
    Plots a Precision-Recall curve based on the true and predicted labels.
    
    Parameters:
    y_true: array-like of shape (n_samples,), true binary labels.
    y_pred: array-like of shape (n_samples,), the predicted probabilities or binary labels.
    
    Returns:
    None
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve


    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


def lagged_scatter_plot(data, variable, lag=1):
    """
    Plots a lagged scatter plot of a variable with its lagged values.

    Parameters:
    data (pandas.DataFrame): The input data.
    variable (str): The name of the variable to plot.
    lag (int): The lag to use for plotting. Default is 1.

    Returns:
    None
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(data[variable], data[variable].shift(lag))
    ax.set_xlabel(variable)
    ax.set_ylabel(variable + f" (lag={lag})")
    ax.set_title(f"Lagged Scatter Plot of {variable} (lag={lag})")
    plt.show()


def lag_correlation(series):
    """
    Description: This function generates an autocorrelation and partial autocorrelation plot for a given time series.

    Parameters:
    series: pandas Series representing the time series to be plotted.

    Returns:
    None
    """
    import matplotlib.pyplot as plt
    import statsmodels.api as sm


    fig, ax = plt.subplots(nrows=2, figsize=(10, 8))

    # Autocorrelation plot
    sm.graphics.tsa.plot_acf(series, lags=50, ax=ax[0])
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('Autocorrelation')
    ax[0].set_title('Autocorrelation Plot')

    # Partial autocorrelation plot
    sm.graphics.tsa.plot_pacf(series, lags=50, ax=ax[1])
    ax[1].set_xlabel('Lag')
    ax


def seasonal_decompose_plot(data, freq):
    """
    Description: This function takes in a time series data and the frequency of the seasonality and decomposes it into seasonal, trend, and residual components.
    
    Parameters:
    data: pandas Series or DataFrame with datetime index containing the time series data
    freq: integer representing the frequency of the seasonality
    
    Returns:
    A matplotlib figure object containing the seasonal decomposition plot
    """
    import matplotlib.pyplot as plt
    import statsmodels.api as sm


    # Decompose the time series
    decomposition = sm.tsa.seasonal_decompose(data, model='additive', period=freq)
    
    # Plot the decomposition
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(12,8))
    plt.subplots_adjust(hspace=0.3)
    
    axes[0].set_title('Observed')
    decomposition.observed.plot(ax=axes[0], legend=False)
    
    axes[1].set_title('Trend')
    decomposition.trend.plot(ax=axes[1], legend=False)
    
    axes[2].set_title('Seasonal')
    decomposition.seasonal.plot(ax=axes[2], legend=False)
    
    axes[3].set_title('Residual')
    decomposition.resid.plot(ax=axes[3], legend=False)
    
    plt.tight_layout()
    
    return fig


def plot_forecast(actual, forecast, xlabel='', ylabel='', title=''):
    """
    Plots a time series forecasting plot showing the actual and forecasted values.
    
    Parameters:
    actual (pandas.Series): Actual time series data.
    forecast (pandas.Series): Forecasted time series data.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    title (str): Title for the plot.
    """
    import matplotlib.pyplot as plt


    plt.plot(actual.index, actual.values, label='Actual')
    plt.plot(forecast.index, forecast.values, label='Forecast')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_residuals(y_true, y_pred):
    """
    Plots a residual plot to visualize the difference between predicted and actual values.

    Parameters:
    y_true: array-like, true values of the target variable
    y_pred: array-like, predicted values of the target variable

    Returns:
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np


    residuals = y_true - y_pred
    plt.figure(figsize=(10,6))
    plt.scatter(np.arange(len(residuals)), residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title("Residual Plot")
    plt.xlabel("Observations")
    plt.ylabel("Residuals")
    plt.show()


def plot_spectral_density(data, sampling_rate):
    """
    Plots a spectral density plot of the given data.
    
    Parameters:
    data (pandas.Series): Time series data to plot.
    sampling_rate (int): Sampling rate of the data.
    
    Returns:
    None (displays plot in console)
    """
    import matplotlib.pyplot as plt
    from scipy import signal
    
    freq, psd = signal.welch(data, fs=sampling_rate)
    plt.figure(figsize=(8, 4))
    plt.plot(freq, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title('Spectral Density Plot')
    plt.show()


def plot_cluster(X, y, title=''):
    """
    Description: This function creates a cluster plot to visualize the clusters formed by unsupervised learning algorithms.

    Parameters:

    X: numpy array or pandas DataFrame containing the data to be plotted.
    y: numpy array or pandas Series containing the cluster labels for each data point.
    title: string representing the title of the plot (optional).

    Returns:

    A seaborn cluster plot object.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Convert pandas objects to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    # Create the plot
    sns.clustermap(X, row_cluster=False, col_cluster=False, cmap='viridis', robust=True, yticklabels=y, figsize=(10,10))
    plt.title(title)
    plt.show()


def plot_pca(df, target_col=None):
    """
    Plots a scatter plot using Principal Component Analysis (PCA) for dimensionality reduction.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data to be plotted.
    target_col (str, optional): Name of the target column. If provided, the plot will show different colors for different target values.

    Returns:
    None
    """

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import pandas as pd

    # Apply PCA to the data
    pca = PCA(n_components=2)
    pca_df = pd.DataFrame(pca.fit_transform(df.drop(target_col, axis=1) if target_col else df))
    pca_df.columns = ['PC1', 'PC2']

    # Add target column to the PCA DataFrame if provided
    if target_col:
        pca_df['target'] = df[target_col]

    # Plot the PCA scatter plot
    fig, ax = plt.subplots()
    if target_col:
        for target_val in df[target_col].unique():
            ix = pca_df['target'] == target_val
            ax.scatter(pca_df.loc[ix, 'PC1'], pca_df.loc[ix, 'PC2'], label=target_val)
    else:
        ax.scatter(pca_df['PC1'], pca_df['PC2'])

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    plt.show()


def silhouette(X, y_pred):

    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_samples, silhouette_score
    import numpy as np

    cluster_labels = np.unique(y_pred)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_pred)
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_pred == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = plt.cm.Spectral(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = silhouette_score(X, y_pred)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette Coefficient')
    plt.show()


def dendrogram(X):
    '''
    Plots a dendrogram based on hierarchical clustering of the data X.

    Parameters:
    X (numpy array): data to be clustered
    
    Returns:
    None
    '''
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    # Calculate linkage matrix
    Z = linkage(X, 'ward')

    # Plot dendrogram
    plt.figure(figsize=(10, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data points')
    plt.ylabel('Distance')
    dendrogram(Z)
    plt.show()


def plot_elbow(X, max_k=10):
    """
    Plots the elbow method curve to help determine the optimal number of clusters for k-means clustering.

    Parameters:
        X (numpy.ndarray): Array of shape (n_samples, n_features) containing the data to be clustered.
        max_k (int): Maximum number of clusters to test. Default is 10.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    sse = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        sse.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.plot(range(1, max_k+1), sse, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method Curve')
    plt.show()





