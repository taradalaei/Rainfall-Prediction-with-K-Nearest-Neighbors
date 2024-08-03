import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv('/content/drive/My Drive/P3/BOM.csv')

# Remove samples with unassigned target values
data = data[data['RainTomorrow'].notna()]

# Extract month from date
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Year'] = data["Date"].dt.year
print(data.head())

# Separate numeric and non-numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

# Fill missing values for numeric columns with median for the same month
for column in numeric_cols:
    data[column] = data.groupby('Month')[column].transform(lambda x: x.fillna(x.median()))

# Fill missing values for non-numeric columns with mode for the same month
for column in non_numeric_cols:
    data[column] = data.groupby('Month')[column].transform(lambda x: x.fillna(x.mode()[0]))

# Define features and target
X = data.drop(columns=['RainTomorrow', 'Date'])
y = data['RainTomorrow']

# One-hot encode non-numeric columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
non_numeric_cols.remove("RainTomorrow")
non_numeric_cols.remove("Date")
encoded_non_numeric = encoder.fit_transform(X[non_numeric_cols])

# Create a DataFrame with the encoded features
encoded_non_numeric_df = pd.DataFrame(encoded_non_numeric, columns=encoder.get_feature_names_out(non_numeric_cols))
print(encoded_non_numeric_df.head())

# Drop the original non-numeric columns and concatenate the encoded features
X = X.drop(columns=non_numeric_cols).reset_index(drop=True)
X = pd.concat([X, encoded_non_numeric_df], axis=1)
print(X.head())

# Split the dataset into train (60%), validation (20%), and test sets (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

##############################################

# Calculate the correlation matrix
correlation_matrix = X_train.corr()

# Identify pairs of highly correlated features (correlation > 0.95)
high_corr_pairs = []
for col1 in correlation_matrix.columns:
    for col2 in correlation_matrix.columns:
        if col1 != col2 and (col2, col1) not in high_corr_pairs:
            if abs(correlation_matrix.loc[col1, col2]) > 0.95:
                high_corr_pairs.append((col1, col2))

print(high_corr_pairs)

# Remove one feature from each highly correlated pair
features_to_remove = set([pair[1] for pair in high_corr_pairs])
print("features to remove: ", features_to_remove)
features_to_remove.remove('RainToday_Yes')
features_to_remove.add('RainToday_No')
X_train = X_train.drop(columns=features_to_remove)
X_val = X_val.drop(columns=features_to_remove)
X_test = X_test.drop(columns=features_to_remove)
print("the training set after removing one of correlated pairs:")
print(X_train.head())

####################################################
X.drop

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Sequential Feature Selector
sfs = SequentialFeatureSelector(LogisticRegression(), n_features_to_select=10, direction='forward')
sfs.fit(X_train_scaled, y_train)

# Get the selected features
selected_features = X_train.columns[sfs.get_support()]
print("selected features: ", selected_features)

################################################

# Train K-NN classifier
best_k = 0
best_accuracy = 0

for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled[:, sfs.get_support()], y_train)
    y_val_pred = knn.predict(X_val_scaled[:, sfs.get_support()])
    accuracy = accuracy_score(y_val, y_val_pred)

    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy

# Report the accuracy on the test set
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled[:, sfs.get_support()], y_train)
y_test_pred = knn.predict(X_test_scaled[:, sfs.get_support()])
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Best k: {best_k}')
print(f'Test Set Accuracy: {test_accuracy}')

report = classification_report(y_test, y_test_pred)
print(report)


#beacause of using one-hot encoding here, it is almost impossible to make the new record just like our training or test data. because it adds a value to the location feature and when using one-hot encoding, it adds a new column to our new dataset for rasht. thats why i couldnt predict this record.


# Load Rasht data
rasht_data = pd.read_csv('/content/Rasht.csv')

# Extract month and handle missing values similarly
rasht_data['Date'] = pd.to_datetime(rasht_data['Date'])
rasht_data['Month'] = rasht_data['Date'].dt.month

for column in rasht_data.columns:
    if rasht_data[column].isnull().sum() > 0:
        rasht_data[column].fillna(rasht_data.groupby('Month')[column].transform('median'), inplace=True)

# Define features and target
X = rasht_data.drop(columns=['RainTomorrow', 'Date'])
y = rasht_data['RainTomorrow']

# One-hot encode non-numeric columns
non_numeric_cols = rasht_data.select_dtypes(exclude=[np.number]).columns.tolist()
non_numeric_cols.remove("RainTomorrow")
non_numeric_cols.remove("Date")
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_non_numeric = encoder.fit_transform(X[non_numeric_cols])

# Create a DataFrame with the encoded features
encoded_non_numeric_df = pd.DataFrame(encoded_non_numeric, columns=encoder.get_feature_names_out(non_numeric_cols))
print(encoded_non_numeric_df.head())

# Drop the original non-numeric columns and concatenate the encoded features
X = X.drop(columns=non_numeric_cols).reset_index(drop=True)
X = pd.concat([X, encoded_non_numeric_df], axis=1)
X_rasht = X
print(X.head())

# Standardize the features
X_rasht_scaled = scaler.fit_transform(X_rasht)

# Predict using the trained K-NN model
rasht_predictions = knn.predict(X_rasht_scaled)

print('Predictions for Rasht:', rasht_predictions)

rasht_data = pd.read_csv('/content/Rasht.csv')
data = pd.read_csv('/content/drive/My Drive/P3/BOM.csv')

# Concatenate the two data frames vertically
data = pd.concat([data, rasht_data], ignore_index=True)

# Display the combined DataFrame
print(data)

# Extract month from date
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Year'] = data["Date"].dt.year
print(data.head())

# Separate numeric and non-numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

# Fill missing values for numeric columns with median for the same month
for column in numeric_cols:
    data[column] = data.groupby('Month')[column].transform(lambda x: x.fillna(x.median()))

# Fill missing values for non-numeric columns with mode for the same month
for column in non_numeric_cols:
    data[column] = data.groupby('Month')[column].transform(lambda x: x.fillna(x.mode()[0]))

# Define features and target
X = data.drop(columns=['RainTomorrow', 'Date'])
y = data['RainTomorrow']

# One-hot encode non-numeric columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
non_numeric_cols.remove("RainTomorrow")
non_numeric_cols.remove("Date")
encoded_non_numeric = encoder.fit_transform(X[non_numeric_cols])

# Create a DataFrame with the encoded features
encoded_non_numeric_df = pd.DataFrame(encoded_non_numeric, columns=encoder.get_feature_names_out(non_numeric_cols))
print(encoded_non_numeric_df.head())

# Drop the original non-numeric columns and concatenate the encoded features
X = X.drop(columns=non_numeric_cols).reset_index(drop=True)
X = pd.concat([X, encoded_non_numeric_df], axis=1)
print(X.head())

X = X.drop(columns=features_to_remove)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)

# Train K-NN classifier
best_k = 0
best_accuracy = 0

for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled[:, sfs.get_support()], y_train)
    y_val_pred = knn.predict(X_val_scaled[:, sfs.get_support()])
    accuracy = accuracy_score(y_val, y_val_pred)

    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy

# Report the accuracy on the test set
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled[:, sfs.get_support()], y_train)
y_test_pred = knn.predict(X_test_scaled[:, sfs.get_support()])
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Best k: {best_k}')
print(f'Test Set Accuracy: {test_accuracy}')

report = classification_report(y_test, y_test_pred)
print(report)

##################################################

# Predict probability of rainfall
rasht_probabilities = knn.predict_proba(X_rasht_scaled)
rain_probabilities = rasht_probabilities[:, 1]  # Probability of rain (class 1)

print('Rainfall probabilities for Rasht:', rain_probabilities)