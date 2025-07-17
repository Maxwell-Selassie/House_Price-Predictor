This project is a complete end-to-end machine learning pipeline built using the Kaggle dataset: "House Prices - Advanced Regression Techniques". It demonstrates a real-world approach to predicting housing prices using data preprocessing, feature engineering, model selection, and hyperparameter tuning.

The project reads both the training and test data files, applies log transformation to reduce skewness in the target variable, handles missing values using imputers, encodes categorical variables, scales numerical features, and generates polynomial features. A pipeline is used to combine all these steps efficiently.

The model used is Ridge Regression, which includes regularization to avoid overfitting. Hyperparameters such as polynomial degree and alpha (regularization strength) are tuned using GridSearchCV. After training and validation, predictions are made on the test dataset and saved to a CSV file in the correct format for Kaggle submission.

Key libraries used include pandas, NumPy, scikit-learn, matplotlib, and seaborn. The model evaluation includes R² score and mean squared error.

This project is a strong addition to my machine learning portfolio, showing my understanding of preprocessing, pipelines, hyperparameter tuning, and the ability to structure machine learning projects for real-world datasets.

To run this project, simply ensure the training and test CSV files are in the project folder and execute the Python script. The submission file will be generated automatically.
