# Capstone 4 Disease Prediction using Machine Learning
--------------------------------------------------------------------------
### Disease Prediction based on the symptom input by user.
--------------------------------------------------------------------------

[![Screenshot-2023-07-14-at-12-43-59-AM.png](https://i.postimg.cc/fL54xC0m/Screenshot-2023-07-14-at-12-43-59-AM.png)](https://postimg.cc/DJWYhq6w)

## Description
The Disease Prediction machine learning system is designed to predict diseases based on the input of symptoms. The system utilizes a dataset of prognosis containing symptom information to train multiple machine learning models, including RandomForestClassifier, GaussianNB, and SVC.

The system takes advantage of the Gradio library to create an interactive user interface where users can select three symptoms from dropdown lists. The selected symptoms are then passed to the predictDisease function, which processes the input and generates predictions using the trained models.

[Link to dataset]([https://data.gov.sg/dataset/resale-flat-prices](https://www.kaggle.com/code/ghaidalthobaity/disease-prediction-random-forest/notebook))

## Import Library

[![Screenshot-2023-07-14-at-12-52-58-AM.png](https://i.postimg.cc/RhgNb2NR/Screenshot-2023-07-14-at-12-52-58-AM.png)](https://postimg.cc/m1FLPX3z)
__________________________________________________________________________

When importing the libraries, the following actions are taking place:

1. `import pandas as pd`: This imports the Pandas library, which provides data manipulation and analysis tools. It allows for the loading, processing, and analysis of structured data in the form of data frames.

2. `from scipy.stats import mode`: This imports the `mode` function from the SciPy library's `stats` module. The `mode` function calculates the mode (the most frequently occurring value) of a dataset or array.

3. `import matplotlib.pyplot as plt`: This imports the `pyplot` module from the Matplotlib library, which provides a MATLAB-like plotting interface. It allows for creating various types of visualizations and plots.

4. `import seaborn as sns`: This imports the Seaborn library, which is a data visualization library based on Matplotlib. Seaborn provides a high-level interface for creating informative and visually appealing statistical graphics.

5. `from sklearn.model_selection import train_test_split`: This imports the `train_test_split` function from the scikit-learn (sklearn) library's `model_selection` module. This function is used to split a dataset into training and testing subsets for machine learning model evaluation.

6. `from sklearn import metrics`: This imports the `metrics` module from scikit-learn. The `metrics` module provides various functions for evaluating the performance of machine learning models, such as calculating accuracy, precision, recall, and F1 score.

7. `from sklearn.metrics import classification_report`: This imports the `classification_report` function from scikit-learn's `metrics` module. The `classification_report` function generates a detailed report with various metrics for evaluating the performance of a classification model.

8. `from sklearn.preprocessing import LabelEncoder`: This imports the `LabelEncoder` class from scikit-learn's `preprocessing` module. `LabelEncoder` is used for encoding categorical variables into numerical format, which is required by many machine learning algorithms.

9. `from sklearn.model_selection import train_test_split, cross_val_score`: This imports additional functions from scikit-learn's `model_selection` module. `cross_val_score` is used for performing cross-validation, and `train_test_split` is used for splitting data into training and testing sets.

10. `from sklearn.svm import SVC`: This imports the `SVC` class from scikit-learn's `svm` module. `SVC` stands for Support Vector Classifier, which is a machine learning algorithm used for classification tasks.

11. `from sklearn.naive_bayes import GaussianNB`: This imports the `GaussianNB` class from scikit-learn's `naive_bayes` module. `GaussianNB` is a popular naive Bayes classifier implementation based on the assumption of Gaussian (normal) distribution of features.

12. `from sklearn.ensemble import RandomForestClassifier`: This imports the `RandomForestClassifier` class from scikit-learn's `ensemble` module. `RandomForestClassifier` is an ensemble learning method that combines multiple decision trees to make predictions.

13. `from sklearn.metrics import accuracy_score, confusion_matrix`: This imports additional functions from scikit-learn's `metrics` module. `accuracy_score` calculates the accuracy of a classification model, and `confusion_matrix` creates a matrix to visualize the performance of a classification model by showing the counts of true positive, true negative, false positive, and false negative predictions.

14. `%matplotlib inline`: This is a Jupyter Notebook magic command that enables the inline display of Matplotlib plots in the notebook itself.

15. pd.set_option("display.max_rows", None): This code sets the maximum number of rows to be displayed when printing or displaying a Pandas DataFrame to None, which means there is no limitation on the number of rows displayed. By default, Pandas limits the number of rows displayed to a certain value, typically 10, to avoid overwhelming the output with a large number of rows.

16. pd.set_option("display.max_columns", None): This code sets the maximum number of columns to be displayed when printing or displaying a Pandas DataFrame to None, meaning there is no limitation on the number of columns displayed. Similar to the previous option, Pandas limits the number of columns displayed by default, typically to a value like 20, to ensure the output is readable and not too wide.

These imported libraries provide a range of functionalities for data manipulation, visualization, model evaluation, and implementation of machine learning algorithms for disease prediction tasks.

## Import Data
----------------------------------------------------------------------------
[![Screenshot-2023-07-14-at-1-03-02-AM.png](https://i.postimg.cc/SsVR1HPK/Screenshot-2023-07-14-at-1-03-02-AM.png)](https://postimg.cc/HccTVvdG)

1. The libraries `numpy`, `pandas`, `gradio`, and `os` are imported. These libraries provide functionality for numerical computations, data processing and analysis, creating user interfaces, and working with the operating system, respectively.

2. The current working directory is obtained using the `os.getcwd()` function and stored in the variable `Capstone_4_Python`.

3. The variable `path` is created by concatenating the current working directory (`Capstone_4_Python`) with the file name (`Testing.csv`).

4. The `pd.read_csv()` function is used to read the CSV file located at the specified path (`path`) and load it into a pandas DataFrame. The DataFrame is assigned to the variable `test`.

5. The `test.head()` function is called to display the first few rows of the `test` DataFrame, providing a preview of the imported data.

[![Screenshot-2023-07-14-at-1-03-17-AM.png](https://i.postimg.cc/66Lggtzc/Screenshot-2023-07-14-at-1-03-17-AM.png)](https://postimg.cc/Cd5cfy3n)

1. A new variable `path2` is created by concatenating the current working directory (`Capstone_4_Python`) with the file name (`Training.csv`).

2. The `pd.read_csv()` function is used to read the CSV file located at the specified path (`path2`) and load it into a pandas DataFrame. The DataFrame is assigned to the variable `train`.

3. The `train.head()` function is called to display the first few rows of the `train` DataFrame, providing a preview of the imported training data.

[![Screenshot-2023-07-14-at-1-15-36-AM.png](https://i.postimg.cc/90bxMPfM/Screenshot-2023-07-14-at-1-15-36-AM.png)](https://postimg.cc/YL4NDWYK)

The code train.shape and test.shape is used to determine the shape of the DataFrame train and test.

When train.shape and test.shape is executed, it returns a tuple that represents the dimensions of the DataFrame. The first element of the tuple represents the number of rows in the DataFrame, and the second element represents the number of columns.

By calling train.shape and test.shape, you can obtain the number of rows and columns in the train DataFrame and test DataFrame respectively, which provides information about the size of the dataset.

## Data Cleaning

[![Screenshot-2023-07-14-at-1-21-59-AM.png](https://i.postimg.cc/25phvqM7/Screenshot-2023-07-14-at-1-21-59-AM.png)](https://postimg.cc/Yv3vBSpv)

1. The code `train.isnull().any()` is used to check for null values in the DataFrame `train`.

When `train.isnull().any()` is executed, it returns a Series that indicates whether each column in the DataFrame contains any null values. If a column contains at least one null value, the corresponding value in the Series will be `True`; otherwise, it will be `False`.

By using `train.isnull().any()`, you can quickly identify which columns in the `train` DataFrame have missing values. This information is helpful for data cleaning and preprocessing steps, as you can decide how to handle or impute the missing values based on the specific column.

2. The code train.drop('Unnamed: 133', inplace=True, axis=1) is used to drop the column named 'Unnamed: 133' from the DataFrame train.

[Screenshot-2023-07-14-at-1-27-29-AM.png](https://postimg.cc/mhndsbd2)

Checking after dropping the 'Unnamed: 133' column

-------------------------------------------------------------------------------

[![Screenshot-2023-07-14-at-1-33-47-AM.png](https://i.postimg.cc/YCnRGfwG/Screenshot-2023-07-14-at-1-33-47-AM.png)](https://postimg.cc/4YcVjcys)

The code train['prognosis'].value_counts() is used to count the number of occurrences of each unique value in the 'prognosis' column of the train DataFrame.

The output of train['prognosis'].value_counts() will show the count of each unique value in the 'prognosis' column, indicating how many instances belong to each class. This information can be useful for assessing the class distribution and potential class imbalance in the dataset.

[![Screenshot-2023-07-14-at-1-39-27-AM.png](https://i.postimg.cc/65YRQWdY/Screenshot-2023-07-14-at-1-39-27-AM.png)](https://postimg.cc/fJSk2hq9)

1. `train.info()`: This command provides information about the DataFrame `train`. It displays a concise summary of the DataFrame, including the number of non-null values, the data type of each column, and memory usage. It is useful for quickly understanding the structure and properties of the dataset.

2. `train.describe()`: This command generates descriptive statistics of the numerical columns in the DataFrame `train`. It calculates various statistical measures such as count, mean, standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum. It provides insights into the distribution and summary statistics of the numerical data in the DataFrame.

3. `encoder = LabelEncoder()`: This command creates an instance of the `LabelEncoder` class from the scikit-learn library. The `LabelEncoder` is used to encode categorical labels into numerical values. It assigns a unique integer to each unique label in the target column.

4. `train["prognosis"] = encoder.fit_transform(train["prognosis"])`: This command applies the `fit_transform()` method of the `LabelEncoder` to the "prognosis" column in the `train` DataFrame. It fits the encoder to the unique values in the "prognosis" column and transforms the labels into encoded numerical values. The transformed values are then assigned back to the "prognosis" column in the DataFrame. This step is typically performed when the target variable is categorical, and machine learning models require numerical inputs.

[![Screenshot-2023-07-14-at-1-42-26-AM.png](https://i.postimg.cc/jShJ1sjz/Screenshot-2023-07-14-at-1-42-26-AM.png)](https://postimg.cc/rDKm02xm)

1. `A = train[["prognosis"]]`: This line creates a new DataFrame `A` containing only the "prognosis" column from the `train` DataFrame. It selects the "prognosis" column using double square brackets `[[...]]`, which returns a DataFrame instead of a Series.

2. `B = train.drop(["prognosis"],axis=1)`: This line creates a new DataFrame `B` by dropping the "prognosis" column from the `train` DataFrame. The `drop()` function is used to remove the specified column(s) along the specified axis (in this case, `axis=1` refers to columns).

3. `C = test.drop(["prognosis"],axis=1)`: This line creates a new DataFrame `C` by dropping the "prognosis" column from the `test` DataFrame. Similar to the previous line, it removes the specified column(s) along the specified axis.

4. `x = train.iloc[:,:-1]`: This line creates a DataFrame `x` containing all rows and all columns except the last column from the `train` DataFrame. The `iloc` function is used for integer-based indexing and slicing.

5. `y = train.iloc[:, -1]`: This line creates a Series `y` containing the last column from the `train` DataFrame. It selects only the last column using negative indexing.

6. `x_train, x_test, y_train, y_test = train_test_split(B,A,test_size=0.2, random_state=24)`: This line splits the data into training and testing sets. It uses the `train_test_split()` function from scikit-learn to split the `B` and `A` DataFrames into `x_train`, `x_test`, `y_train`, and `y_test` respectively. The `test_size` parameter specifies the proportion of the data that should be allocated for testing (in this case, 20% for testing and 80% for training). The `random_state` parameter ensures reproducibility of the random splitting.

7. `print(f"Train: {x_train.shape}, {y_train.shape}")` and `print(f"Test: {x_test.shape}, {y_test.shape}")`:
This line prints the shape of the training data. It displays the number of rows and columns in `x_train` and `y_train`, `x_test` and `y_test` using the `shape` attribute. The f-string formatting is used to include the shape values in the printed statement.

## Split Data

[![Screenshot-2023-07-14-at-1-57-36-AM.png](https://i.postimg.cc/6QVJnpBG/Screenshot-2023-07-14-at-1-57-36-AM.png)](https://postimg.cc/jWjFrKws)

k-fold cross-validation for multiple machine learning models.

1.	`cv_scoring(estimator, x, y)`: This function is defined to calculate the scoring metric (accuracy score in this case) for cross-validation. It takes an estimator (model), input features (x), and target labels (y) as input and returns the accuracy score.

2.	`models`: This dictionary defines the different models to be evaluated in the cross-validation. It contains instances of the Support Vector Classifier (SVC), Gaussian Naive Bayes (Gaussian NB), and Random Forest Classifier.

3.	Cross-validation loop: The code iterates over each model in the `models` dictionary. For each model, it performs k-fold cross-validation using the `cross_val_score` function. The input features `x` and target labels `y` are passed along with the specified number of folds (`cv=10`), number of parallel jobs (`n_jobs=-1`), and the scoring metric (`scoring=cv_scoring`). The result is a list of scores for each fold.

4.	Printing results: After performing cross-validation for each model, the code prints the model name, individual scores for each fold (`scores`), and the mean score (`np.mean(scores)`). This provides an evaluation of the model's performance across the different folds.

## Model Building
[![Screenshot-2023-07-14-at-2-00-15-AM.png](https://i.postimg.cc/Gm5JKFhW/Screenshot-2023-07-14-at-2-00-15-AM.png)](https://postimg.cc/rD5rFt4j)

The above image is training a Random Forest Classifier model and generating a classification report based on the predictions made by the model.

1.	`mod = RandomForestClassifier(n_estimators = 150, n_jobs = 8, criterion= 'entropy', random_state = 42)`: This line initializes a Random Forest Classifier model with specified parameters. `n_estimators` indicates the number of decision trees in the forest, `n_jobs` sets the number of parallel jobs to run for fitting the trees, `criterion` defines the impurity measure for splitting nodes ('entropy' in this case), and `random_state` sets the seed for random number generation.

2.	`mod = mod.fit(x_train, y_train.values.ravel())`: The model is trained using the training data (`x_train` as input features and `y_train` as target labels). The `values.ravel()` method is used to convert the target labels to a 1-dimensional array.

3.	`pred = mod.predict(x_test)`: The trained model is used to predict the target labels for the test data (`x_test`).

4.	`report = classification_report(y_test, pred, output_dict=True)`: The `classification_report` function is applied to compare the predicted labels (`pred`) with the actual labels (`y_test`) from the test set. The `output_dict=True` parameter returns the classification report as a dictionary.

5.	`pd.DataFrame(report).transpose()`: The classification report dictionary is converted into a pandas DataFrame and transposed to display the report with metrics such as precision, recall, F1-score, and support for each class. Each row corresponds to a class, and the columns represent the metrics.


[![Screenshot-2023-07-14-at-2-05-31-AM.png](https://i.postimg.cc/Y0dbwJM9/Screenshot-2023-07-14-at-2-05-31-AM.png)](https://postimg.cc/9wqGdg45)

1.	`metrics.accuracy_score(y_test, pred)`: This function takes two arguments, the true labels (`y_test`) and the predicted labels (`pred`), and computes the accuracy score, which is the proportion of correct predictions out of the total number of samples.
2.	The returned value represents the accuracy of the model's predictions on the test data.

A Support Vector Machine (SVM) classifier is trained and tested on the given data.

3.	`svm_model = SVC()`: An SVM classifier object is created.

4.	`svm_model.fit(x_train, y_train.values.reshape(-1))`: The SVM classifier is trained on the training data (`x_train` and `y_train`). The `reshape(-1)` is used to ensure that the shape of `y_train` is compatible with the SVM classifier.

5.	`preds = svm_model.predict(x_test)`: The SVM classifier makes predictions on the test data (`x_test`) and assigns the predicted labels to `preds`.

6.	`accuracy_score(y_train, svm_model.predict(x_train))`: The accuracy of the SVM classifier's predictions on the training data is calculated using the `accuracy_score` function. It compares the true labels (`y_train`) with the predicted labels on the training data (`svm_model.predict(x_train)`).

7.	`accuracy_score(y_test, preds)`: The accuracy of the SVM classifier's predictions on the test data is calculated using the `accuracy_score` function. It compares the true labels (`y_test`) with the predicted labels on the test data (`preds`).

8.	`cf_matrix = confusion_matrix(y_test, preds)`: The confusion matrix is computed using the `confusion_matrix` function. It compares the true labels (`y_test`) with the predicted labels on the test data (`preds`).

9.	`sns.heatmap(cf_matrix, annot=True, cmap='seismic')`: A heatmap visualization of the confusion matrix is created using the seaborn library.

10.	The accuracy scores on both the training and test data are printed, and the confusion matrix heatmap is displayed.

[![Screenshot-2023-07-14-at-2-10-47-AM.png](https://i.postimg.cc/x81BHGdX/Screenshot-2023-07-14-at-2-10-47-AM.png)](https://postimg.cc/LYGyF1LS)

A Naive Bayes classifier is trained and tested on the given data.

1.	`nb_model = GaussianNB()`: A Naive Bayes classifier object is created.

2.	`nb_model.fit(x_train, y_train.values.ravel())`: The Naive Bayes classifier is trained on the training data (`x_train` and `y_train`). The `ravel()` function is used to ensure that the shape of `y_train` is compatible with the Naive Bayes classifier.

3.	`preds = nb_model.predict(x_test)`: The Naive Bayes classifier makes predictions on the test data (`x_test`) and assigns the predicted labels to `preds`.

4.	`accuracy_score(y_train, nb_model.predict(x_train))`: The accuracy of the Naive Bayes classifier's predictions on the training data is calculated using the `accuracy_score` function. It compares the true labels (`y_train`) with the predicted labels on the training data (`nb_model.predict(x_train)`).

5.	`accuracy_score(y_test, preds)`: The accuracy of the Naive Bayes classifier's predictions on the test data is calculated using the `accuracy_score` function. It compares the true labels (`y_test`) with the predicted labels on the test data (`preds`).

6.	`cf_matrix = confusion_matrix(y_test, preds)`: The confusion matrix is computed using the `confusion_matrix` function. It compares the true labels (`y_test`) with the predicted labels on the test data (`preds`).

7.	`sns.heatmap(cf_matrix, annot=True, cmap='seismic')`: A heatmap visualization of the confusion matrix is created using the seaborn library.

8.	The accuracy scores on both the training and test data are printed, and the confusion matrix heatmap is displayed.


[![Screenshot-2023-07-14-at-2-12-40-AM.png](https://i.postimg.cc/NFZbMR2q/Screenshot-2023-07-14-at-2-12-40-AM.png)](https://postimg.cc/SY7c12GD)

A Random Forest classifier is trained and tested on the given data.

1.	`rf_model = RandomForestClassifier(random_state=18)`: A Random Forest classifier object is created with a specified random state for reproducibility.

2.	`rf_model.fit(x_train, y_train.values.ravel())`: The Random Forest classifier is trained on the training data (`x_train` and `y_train`). The `ravel()` function is used to ensure that the shape of `y_train` is compatible with the Random Forest classifier.

3.	`preds = rf_model.predict(x_test)`: The Random Forest classifier makes predictions on the test data (`x_test`) and assigns the predicted labels to `preds`.

4.	`accuracy_score(y_train, rf_model.predict(x_train))`: The accuracy of the Random Forest classifier's predictions on the training data is calculated using the `accuracy_score` function. It compares the true labels (`y_train`) with the predicted labels on the training data (`rf_model.predict(x_train)`).

5.	`accuracy_score(y_test, preds)`: The accuracy of the Random Forest classifier's predictions on the test data is calculated using the `accuracy_score` function. It compares the true labels (`y_test`) with the predicted labels on the test data (`preds`).

6.	`cf_matrix = confusion_matrix(y_test, preds)`: The confusion matrix is computed using the `confusion_matrix` function. It compares the true labels (`y_test`) with the predicted labels on the test data (`preds`).

7.	`sns.heatmap(cf_matrix, annot=True, cmap='seismic')`: A heatmap visualization of the confusion matrix is created using the seaborn library.

8.	The accuracy scores on both the training and test data are printed, and the confusion matrix heatmap is displayed.

## Validating Data

[![Screenshot-2023-07-14-at-2-14-47-AM.png](https://i.postimg.cc/pLcQXDw0/Screenshot-2023-07-14-at-2-14-47-AM.png)](https://postimg.cc/JsJHpB4J)

The final SVM, Naive Bayes, and Random Forest models are trained on the whole dataset (`x` and `y`).

1. `final_svm_model = SVC()`: The final SVM model is initialized.
2. `final_nb_model = GaussianNB()`: The final Naive Bayes model is initialized.
3. `final_rf_model = RandomForestClassifier(random_state=18)`: The final Random Forest model is initialized.
4. `final_svm_model.fit(x, y)`: The final SVM model is trained on the whole dataset.
5. `final_nb_model.fit(x, y)`: The final Naive Bayes model is trained on the whole dataset.
6. `final_rf_model.fit(x, y)`: The final Random Forest model is trained on the whole dataset.

The test dataset (`test`) is read and preprocessed.
7. `test_x` is assigned the feature values from the test dataset.
8. `test_y` is assigned the encoded target labels from the test dataset.

Predictions are made on the test dataset using the trained models:
9. `svm_preds` contains the predictions made by the final SVM model.
10. `nb_preds` contains the predictions made by the final Naive Bayes model.
11. `rf_preds` contains the predictions made by the final Random Forest model.

The final predictions are obtained by taking the mode of predictions made by all three classifiers (`svm_preds`, `nb_preds`, and `rf_preds`).
12. The accuracy of the combined model's predictions on the test dataset is calculated using `accuracy_score(test_y, final_preds)`.
13. The confusion matrix is computed using `confusion_matrix(test_y, final_preds)` and displayed as a heatmap using seaborn.

## Displaying Tree

[![Screenshot-2023-07-14-at-2-19-23-AM.png](https://i.postimg.cc/8zd82zxP/Screenshot-2023-07-14-at-2-19-23-AM.png)](https://postimg.cc/215XvrzP)

The decision tree visualization for one of the estimators from the Random Forest model is created and displayed using matplotlib.

1.	`from sklearn import tree`: The `tree` module from scikit-learn is imported.
2.	`plt.figure(figsize=(30,15))`: A figure with a specific size is created to plot the decision tree.
3.	`tree.plot_tree(mod.estimators_[8], filled=True)`: The `plot_tree` function is used to plot the decision tree of the estimator at index 8 from the `mod` (Random Forest) model. The `filled` parameter is set to `True` to color the tree nodes based on the majority class.

## Predicting Disease

[![Screenshot-2023-07-14-at-2-20-58-AM.png](https://i.postimg.cc/3rD3LYrW/Screenshot-2023-07-14-at-2-20-58-AM.png)](https://postimg.cc/NLBhFhHw)

In this code snippet, the following steps are performed:

1. The `symptoms` variable is assigned the column names of the `x` DataFrame, which represent the symptoms.
2. A dictionary called `symptom_index` is created to map each symptom to its corresponding index. The symptom names are capitalized and split by underscores if present.
3. The feature names of the model objects `final_rf_model`, `final_nb_model`, and `final_svm_model` are assigned as the `symptoms`.
4. The `data_dict` dictionary is created, containing the `symptom_index` and the `predictions_classes` from the `encoder` object.
5. The `predictDisease` function is defined. It takes a string of symptoms separated by commas as input.
6. The function processes the input symptoms, creating an input data array for the models. It sets the corresponding index to 1 for each symptom present.
7. The input data is reshaped and converted into a suitable format for model predictions.
8. Individual predictions are generated using the trained models (`final_rf_model`, `final_nb_model`, and `final_svm_model`).
9. The final prediction is made by taking the mode of all predictions.
10. The function returns a dictionary containing the predictions for each model and the final prediction.
11. The `predictDisease` function is tested by passing the input string "Itching,Skin Rash,Nodal Skin Eruptions". The predictions are printed.

### Using Gradio interface

[![Screenshot-2023-07-14-at-2-26-03-AM.png](https://i.postimg.cc/HkGGVybt/Screenshot-2023-07-14-at-2-26-03-AM.png)](https://postimg.cc/8jbXYsKF)

1. The necessary libraries are imported, including `gradio`, `numpy`, `pandas`, `scipy.stats`, `RandomForestClassifier`, `GaussianNB`, and `SVC`.
2. The `symptoms` variable is assigned the column names of the `train` DataFrame, excluding the last column (which represents the target variable).
3. Three models, namely `final_rf_model`, `final_nb_model`, and `final_svm_model`, are initialized (Random Forest, Naive Bayes, and SVM) and trained using the `x_train` and `y_train` data.
4. The `predictDisease` function is defined, which takes three symptoms as input.
5. Within the function, an input data array is created, with the presence of each symptom indicated by setting the corresponding index to 1.
6. The input data is reshaped and converted into a suitable format for model predictions.
7. Predictions are made using the trained models (`final_rf_model`, `final_nb_model`, and `final_svm_model`).
8. The final prediction is made by taking the mode of the predictions from the three models.
9. The function returns a dictionary containing the predictions for each model and the final prediction.
10. A `data_dict` dictionary is created, which contains the symptom index and the prediction classes.
11. The `test_x` variable is initialized with a zero-filled array, with dimensions (1, len(symptoms)).
12. Placeholder values are assigned to `final_rf_prediction`, `final_nb_prediction`, and `final_svm_prediction`.
13. Dropdown input interfaces (`symptom_dropdown1`, `symptom_dropdown2`, `symptom_dropdown3`) are created using `gradio`.
14. An output interface (`disease_output`) is defined as a label for displaying the predicted disease.
15. A Gradio interface is created, using the `predictDisease` function as the prediction function, the dropdown inputs and label output, and appropriate titles and descriptions.
16. The Gradio interface is launched and shared.

The overall purpose of this code is to create a Gradio interface that allows users to select three symptoms and obtain a predicted disease based on the selected symptoms using the trained models.

[![Screenshot-2023-07-14-at-2-41-30-AM.png](https://i.postimg.cc/28YcMNNC/Screenshot-2023-07-14-at-2-41-30-AM.png)](https://postimg.cc/grMHL507)
----------------------------------------------------------------------------
The overall objective of this project is to develop a disease prediction system using machine learning techniques. The project aims to utilize a dataset containing symptoms and corresponding diseases to train multiple models, including Random Forest, Naive Bayes, and Support Vector Machine (SVM). The trained models are then used to predict diseases based on input symptoms provided by users through a Gradio interface.

The project's goal is to provide a user-friendly and interactive platform where individuals can input their symptoms and receive predictions of potential diseases. By leveraging machine learning algorithms, the system aims to assist in early disease detection, provide insights for medical professionals, and raise awareness among users about potential health conditions associated with their symptoms.

Overall, the project contributes to the field of healthcare by applying machine learning techniques to aid in disease prediction and provide a convenient tool for individuals to assess their symptoms and seek appropriate medical advice.
The interactive dashboard allows me to compare the data, filter the data and focus on specific interest when need to. When we have big data file, interactive dashboard make it easy to read the data.

----------------------------------------------------------------------------
[Contact me via LinkedIn](https://www.linkedin.com/in/shafinabegum)







