import pandas as pd
import unittest
import os
from ..analysis import line, scatter, bar

class TestAnalysis(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [1, 4, 9, 16, 25],
            'color': ['a', 'a', 'b', 'b', 'b'],
            'size': [10, 20, 30, 40, 50]
        })

    def test_line(self):
        line(self.df, 'x', 'y', title='Line Plot Test', x_title='X-Axis', y_title='Y-Axis', color='green', filename='line_test.html')
        self.assertTrue(os.path.exists('line_test.html'))
        os.remove('line_test.html')

    def test_scatter(self):
        scatter(self.df, 'x', 'y', color_col='color', size_col='size', title='Scatter Plot Test', x_title='X-Axis', y_title='Y-Axis', marker=dict(color='red', size=8), filename='scatter_test.html')
        self.assertTrue(os.path.exists('scatter_test.html'))
        os.remove('scatter_test.html')

    def test_bar(self):
        bar(self.df, 'x', 'y', color_col='color', title='Bar Chart Test', x_title='X-Axis', y_title='Y-Axis', filename='bar_test.html')
        self.assertTrue(os.path.exists('bar_test.html'))
        os.remove('bar_test.html')
