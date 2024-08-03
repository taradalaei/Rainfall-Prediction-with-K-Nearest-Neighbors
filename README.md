# Rainfall Prediction with K-Nearest Neighbors

This project demonstrates a machine learning pipeline for predicting rainfall using a K-Nearest Neighbors (K-NN) classifier. The dataset used is the BOM (Bureau of Meteorology) dataset, and the code includes data preprocessing, feature selection, model training, and evaluation. Additionally, the pipeline is applied to predict rainfall probabilities for a new dataset from Rasht.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection](#feature-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Predicting New Data](#predicting-new-data)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run the project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/rainfall-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd rainfall-prediction
    ```
3. Run the Python script:
    ```bash
    python rainfall_prediction.py
    ```

## Project Structure

- `rainfall_prediction.py`: The main script containing the entire pipeline from data loading to prediction.
- `README.md`: This file, providing an overview of the project.

## Data Preprocessing

1. **Loading Data**: The dataset is loaded from a CSV file.
2. **Handling Missing Values**: Missing values in numeric columns are filled with the median value for the corresponding month, and missing values in non-numeric columns are filled with the mode for the corresponding month.
3. **Feature Engineering**: New features such as Month, Day, and Year are extracted from the Date column.
4. **One-Hot Encoding**: Non-numeric columns are one-hot encoded to be used in the model.
5. **Data Splitting**: The dataset is split into training, validation, and test sets.

## Feature Selection

1. **Correlation Analysis**: Highly correlated features (correlation > 0.95) are identified and one feature from each pair is removed.
2. **Sequential Feature Selector**: A logistic regression-based sequential feature selector is used to select the top 10 features.

## Model Training and Evaluation

1. **Standardization**: Features are standardized using `StandardScaler`.
2. **K-NN Classifier**: A K-NN classifier is trained and evaluated using the validation set to find the best value of k (number of neighbors).
3. **Test Set Evaluation**: The best model is evaluated on the test set, and accuracy and classification report are generated.

## Predicting New Data

1. **New Data Loading**: A new dataset (e.g., from Rasht) is loaded and preprocessed similarly to the training data.
2. **Prediction**: The trained K-NN model is used to predict rainfall probabilities for the new dataset.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
