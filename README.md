# MAna: Data Manipulation, Analysis, and Modeling Package

This package provides a collection of functions, class and moduls for data analysis, manipulation, and modeling. The main purpose of the package is to simplify and streamline the process of data analysis and modeling by providing a set of commonly used tools and techniques.

### Organization of the Package Code in the Repository

The package code is organized into several sub-packages:

data: for loading and cleaning data.
analysis: for data visualization.
modeling: for data preprocessing and model evaluation and visualization.


### Usage
To use this package, you can import the desired functions or classes from the relevant sub-packages. For example, to load and preprocess data, you can use the following code:
```
from data import data_io, data_cleaning

# Load data
data = data_io.read_data('path/to/data.csv')

# Preprocess data
preprocessed_data = data_cleaning.drop_missing_values(data)
```

To visualize data, you can use the following code:
```
from analysis import visualizations

# Plot histogram
visualizations.dist(data)

# Plot scatter plot
visualizations.scatter(data)
```
To evaluate models, you can use the following code:
```
from modeling import model_evaluation

# Evaluate model
evaluation_metrics = model_evaluation.evaluate(model, X_test, y_test)
```

This is just a small sample of what this package has to offer. For more detailed information on the available functions and classes, please refer to the documentation.