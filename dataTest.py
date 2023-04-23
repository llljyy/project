import pandas as pd
import numpy as np
from ..data import *
import unittest

class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        # create a sample dataframe for testing
        self.df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5],
                                'B': [6, 7, 8, np.nan, 10],
                                'C': [11, 12, 13, 14, 15],
                                'D': [16, 17, 18, 19, 20],
                                'E': [21, np.nan, 23, 24, 25],
                                'F': [26, 27, np.nan, 29, 30],
                                'G': [31, 32, 33, 34, 35]})
                                
    def test_drop_missing_values(self):
        # test dropping missing values with default threshold and axis
        df_clean = drop_missing_values(self.df)
        self.assertEqual(df_clean.shape, (3, 7))

        # test dropping missing values with specified threshold and axis
        df_clean = drop_missing_values(self.df, threshold=0.3, axis=1)
        self.assertEqual(df_clean.shape, (5, 5))
        
        # test dropping missing values with specified subset
        df_clean = drop_missing_values(self.df, subset=['A', 'C', 'E'])
        self.assertEqual(df_clean.shape, (4, 7))

    def test_fill_missing_values(self):
        # test filling missing values with mean
        df_clean = fill_missing_values(self.df)
        self.assertEqual(df_clean.isna().sum().sum(), 0)
        
        # test filling missing values with median
        df_clean = fill_missing_values(self.df, method='median')
        self.assertEqual(df_clean.isna().sum().sum(), 0)

        # test filling missing values with mode
        df_clean = fill_missing_values(self.df, method='mode')
        self.assertEqual(df_clean.isna().sum().sum(), 0)

        # test filling missing values with a value
        df_clean = fill_missing_values(self.df, fill_value=999)
        self.assertEqual(df_clean.isna().sum().sum(), 0)

        # test filling missing values with specified subset
        df_clean = fill_missing_values(self.df, subset=['A', 'C', 'E'], method='median')
        self.assertEqual(df_clean.isna().sum().sum(), 0)

    def test_remove_outliers(self):
        # test removing outliers with default threshold
        df_clean = remove_outliers(self.df)
        self.assertEqual(df_clean.shape, (4, 7))

        # test removing outliers with specified threshold
        df_clean = remove_outliers(self.df, threshold=2)
        self.assertEqual(df_clean.shape, (5, 7))


class TestReadData(unittest.TestCase):

    def test_csv_file(self):
        file_path = "test.csv"
        df = read_data(file_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_excel_file(self):
        file_path = "test.xlsx"
        df = read_data(file_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_json_file(self):
        file_path = "test.json"
        df = read_data(file_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_text_file(self):
        file_path = "test.txt"
        df = read_data(file_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_sas_file(self):
        file_path = "test.sas"
        df = read_data(file_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_spss_file(self):
        file_path = "test.sav"
        df = read_data(file_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_parquet_file(self):
        file_path = "test.parquet"
        df = read_data(file_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_pickle_file(self):
        file_path = "test.pickle"
        df = read_data(file_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_unsupported_file_type(self):
        file_path = "test.unknown"
        with self.assertRaises(ValueError):
            read_data(file_path)


class TestReadImages(unittest.TestCase):

    def test_image_folder(self):
        folder_path = "test_images"
        image_size = (100, 100)
        images = read_images(folder_path, image_size)
        self.assertIsInstance(images, np.ndarray)
        self.assertEqual(images.shape, (2, 100, 100, 3))


class TestReadFromDatabase(unittest.TestCase):

    def test_database_connection(self):
        database_url = "postgresql://user:password@localhost/testdb"
        query = "SELECT * FROM test_table"
        df = read_from_database(database_url, query)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)


