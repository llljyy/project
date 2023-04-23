import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from io import StringIO
from unittest.mock import patch
from ..modeling import *

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        data = {'age': [25, 45, 30, 50],
                'gender': ['male', 'female', 'male', 'female'],
                'income': [50000, 60000, 70000, 80000],
                'target': [0, 1, 0, 1]}
        self.df = pd.DataFrame(data)
        self.feature_engineering = FeatureEngineering(self.df)

    def test_scale_standard(self):
        scaled_data = self.feature_engineering.scale(method='standard')
        for col in scaled_data.columns:
            self.assertAlmostEqual(scaled_data[col].mean(), 0, delta=0.0001)
            self.assertAlmostEqual(scaled_data[col].std(), 1, delta=0.0001)

    def test_scale_minmax(self):
        scaled_data = self.feature_engineering.scale(method='minmax')
        for col in scaled_data.columns:
            self.assertAlmostEqual(scaled_data[col].min(), 0, delta=0.0001)
            self.assertAlmostEqual(scaled_data[col].max(), 1, delta=0.0001)

    def test_normalize(self):
        normalized_data = self.feature_engineering.normalize()
        for col in normalized_data.columns:
            self.assertAlmostEqual(np.linalg.norm(normalized_data[col]), 1, delta=0.0001)

    def test_encode(self):
        encoded_data = self.feature_engineering.encode(columns=['gender'])
        self.assertListEqual(list(encoded_data.columns), ['age', 'income', 'target', 'gender_female', 'gender_male'])

    def test_full_preprocess(self):
        preprocessed_data = self.feature_engineering.full_preprocess(method='standard', columns=['gender'], one_hot=True)
        for col in preprocessed_data.columns:
            self.assertAlmostEqual(preprocessed_data[col].mean(), 0, delta=0.0001)
            self.assertAlmostEqual(preprocessed_data[col].std(), 1, delta=0.0001)
        self.assertListEqual(list(preprocessed_data.columns), ['age', 'income', 'target', 'gender_female', 'gender_male'])

    def test_split(self):
        X_train, X_test, y_train, y_test = self.feature_engineering.split(y='target', test_size=0.25, random_state=42)
        self.assertEqual(len(X_train)+len(X_test), len(self.df))
        self.assertEqual(len(y_train)+len(y_test), len(self.df))
        self.assertAlmostEqual(len(X_test)/len(X_train), 0.25, delta=0.05)


class EvaluateTest(unittest.TestCase):
    def test_evaluate(self):
        # Create a binary classification dataset
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

        # Train a logistic regression model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Evaluate the model
        metrics = evaluate(model, X, y)

        # Check if the evaluation metrics are within expected range
        self.assertGreater(metrics['accuracy'], 0.8)
        self.assertGreater(metrics['precision'], 0.8)
        self.assertGreater(metrics['recall'], 0.8)
        self.assertGreater(metrics['f1-score'], 0.8)
        self.assertGreater(metrics['AUC-ROC'], 0.8)
        self.assertGreater(metrics['AUC-PRC'], 0.8)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0])
        self.y_pred = np.array([0, 0, 1, 1, 1, 1, 0, 0])
        self.data = pd.read_csv(StringIO('''
                time, value1, value2
                2019-01-01 00:00:00, 0.5, 1.0
                2019-01-01 01:00:00, 0.8, 2.0
                2019-01-01 02:00:00, 1.2, 3.0
                2019-01-01 03:00:00, 1.4, 4.0
                2019-01-01 04:00:00, 1.7, 5.0
                '''), index_col=0, parse_dates=True)

    def test_confusion_matrix(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            confusion_matrix(self.y_true, self.y_pred)
            self.assertRegex(fake_out.getvalue(), 'Confusion Matrix')

    def test_roc_curve(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            roc_curve(self.y_true, self.y_pred)
            self.assertRegex(fake_out.getvalue(), 'ROC curve')

    def test_precision_recall_curve(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            precision_recall_curve(self.y_true, self.y_pred)
            self.assertRegex(fake_out.getvalue(), 'Precision-Recall curve')

    def test_lagged_scatter_plot(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            lagged_scatter_plot(self.data, 'value1')
            self.assertRegex(fake_out.getvalue(), 'Lagged Scatter Plot')

    def test_lag_correlation(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            lag_correlation(self.data['value1'])
            self.assertRegex(fake_out.getvalue(), 'Autocorrelation Plot')

    def test_seasonal_decompose_plot(self):
        with patch('sys.stdout', new=StringIO()) as fake_out:
            seasonal_decompose_plot(self.data['value1'], 24)
            self.assertRegex(fake_out.getvalue(), 'Seasonal Decomposition of Time Series')


if __name__ == '__main__':
    unittest.main()

