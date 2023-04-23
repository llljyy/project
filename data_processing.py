class FeatureEngineering:

    import pandas as pd

    def __init__(self, data : pd.DataFrame):
        self.data = data


    def scale(self, method ='standard'):
        """
        Scales the data using either the StandardScaler or MinMaxScaler method.
        """

        import pandas as pd
        from sklearn.preprocessing import StandardScaler, MinMaxScaler


        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'standard' or 'minmax'.")
            
        scaled_data = self.scaler.fit_transform(self.data)

        self.scaled_data =  pd.DataFrame(scaled_data, columns=self.data.columns)
        return self.scaled_data
    

    def normalize(self):
        """
        Normalizes the data using L2 normalization.
        """
        import numpy as np

        self.normalized_data = self.data.apply(lambda x: x / np.linalg.norm(x))
        return self.normalized_data
    

    def encode(self, columns = None, one_hot=True):

        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder

        if columns:
            if one_hot:
                self.encoder = OneHotEncoder()
                encoded_data = self.encoder.fit_transform(self.data[columns])
                new_columns = self.encoder.get_feature_names_out(columns)
                encoded_data = pd.DataFrame(encoded_data.toarray(), columns=new_columns)
                self.encoded_data = pd.concat([self.data, encoded_data], axis=1).drop(columns=columns)
            else:
                dummies = pd.get_dummies(self.data[columns])
                self.encoded_data = pd.concat([self.data, dummies], axis=1).drop(columns=columns)
        else:
            columns = [col for col in self.data.columns if self.data[col].dtype == 'object']
            if one_hot:
                self.encoder = OneHotEncoder()
                encoded_data = self.encoder.fit_transform(self.data[columns])
                new_columns = self.encoder.get_feature_names_out(columns)
                encoded_data = pd.DataFrame(encoded_data.toarray(), columns=new_columns)
                self.encoded_data = pd.concat([self.data, encoded_data], axis=1).drop(columns=columns)
            else:
                dummies = pd.get_dummies(self.data[columns])
                self.encoded_data = pd.concat([self.data, dummies], axis=1).drop(columns=columns)
        return self.encoded_data
    

    def full_preprocess(self, method ='standard', columns = None, one_hot=True ):
        return FeatureEngineering(FeatureEngineering(FeatureEngineering(self.data).encode(columns,one_hot)).scale(method)).normalize()


    def split(self, y, test_size=0.2, random_state=None, shuffle=True, stratify=None):
        """
        Splits the data into training and testing sets.

        Parameters:
        y (numpy array or pandas series): Target variable
        test_size (float, optional): The proportion of the dataset to include in the test split (default=0.2)
        random_state (int, optional): Seed used by the random number generator (default=None)
        shuffle (bool, optional): Whether or not to shuffle the data before splitting (default=True)
        stratify (array-like, optional): If not None, data is split in a stratified fashion (default=None)

        Returns:
        X_train (numpy array or pandas dataframe): Training input features
        X_test (numpy array or pandas dataframe): Testing input features
        y_train (numpy array or pandas series): Training target variable
        y_test (numpy array or pandas series): Testing target variable
        """
        from sklearn.model_selection import train_test_split

        processed_data = self.full_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(processed_data.drop(y,axis=1), processed_data[y], test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        return X_train, X_test, y_train, y_test
    

def calculate_rolling_average(data, column_name, window=None):
    """
    Description: This function calculates the rolling average of a column in a pandas dataframe grouped by a specific column using a specified window size.
    
    Parameters:
    data: pandas DataFrame containing the data to be aggregated.
    column_name: string representing the name of the column to calculate the rolling average for.
    window: integer representing the size of the window to calculate the rolling average. If None, the window size is set to 3.
    
    Returns:
    A pandas DataFrame containing the rolling average values for each group and time window.
    """
    
    if window is None:
        window = 3
    
    rolling_average = data.groupby(column_name)[column_name].rolling(window).mean().reset_index()
    rolling_average = rolling_average.rename(columns={column_name: f'rolling_{window}_avg'})
    
    return rolling_average


def aggregate_data(data, groupby_cols, agg_cols, agg_funcs):
    """
    Description: This function aggregates data in a pandas DataFrame by grouping the data by specified columns and calculating summary statistics of specified columns.

    Parameters:
    data: pandas DataFrame containing the data to be aggregated.
    groupby_cols: a list of strings representing the column(s) to group by.
    agg_cols: a list of strings representing the column(s) to be summarized.
    agg_funcs: a list of functions to apply to each column in agg_cols. 

    Returns:
    A pandas DataFrame containing the aggregated data.
    """
    agg_dict = {col: agg_func for col, agg_func in zip(agg_cols, agg_funcs)}
    aggregated_data = data.groupby(groupby_cols).agg(agg_dict)
    aggregated_data = aggregated_data.reset_index()
    return aggregated_data


    

    