import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def housing_prices(training_filename='training set.csv', test_filename='test.csv'):
    '''
    Machine learning experiment using the Kaggle housing price dataset.
    '''
    # Load datasets
    try:
        df = pd.read_csv(training_filename)
        test_df = pd.read_csv(test_filename)
        test_ids = test_df['Id']
    except FileNotFoundError:
        print(f"Error: One of the files ({training_filename}, {test_filename}) not found.")
        return

    # Log missing data
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_percent = (missing / len(df)) * 100
    print("Missing Data:\n", pd.DataFrame({'Missing Count': missing, 'Missing %': missing_percent}))

    # Transform target to reduce skew
    df['logSalePrice'] = np.log1p(df['SalePrice'])

    # Drop columns with high missingness
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Fence', 'MasVnrType']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Dropped columns: {cols_to_drop}")

    # Prepare features and target
    x = df.drop(columns=['SalePrice', 'logSalePrice'])
    test_x = test_df.copy()
    y = df['logSalePrice']

    # Verify test set columns
    missing_cols = set(x.columns) - set(test_x.columns)
    if missing_cols:
        print(f"Warning: Test set missing columns: {missing_cols}")

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Separate numerical and categorical features
    num_features = x.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = x.select_dtypes(include=['object']).columns.tolist()
    print(f"Numerical features: {len(num_features)}, Categorical features: {len(cat_features)}")

    # Preprocessing pipelines
    numerical_transformers = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42)),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(include_bias=False))
    ])
    categorical_transformers = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformers, num_features),
        ('cat', categorical_transformers, cat_features)
    ])

    # Model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])

    # Hyperparameter tuning
    grid_params = {
        'preprocessor__num__poly__degree': [1, 2],
        'model__alpha': [0.1, 1, 10, 20, 50, 100]
    }
    grid = GridSearchCV(model, grid_params, cv=5, scoring='neg_mean_squared_error')
    grid.fit(x_train, y_train)

    # Evaluate on test set
    y_pred = grid.predict(x_test)
    print('Best polynomial degree:', grid.best_params_['preprocessor__num__poly__degree'])
    print('Best regularization strength:', grid.best_params_['model__alpha'])
    print('Best Cross Validation MSE:', -grid.best_score_)
    print('R^2 score (test set):', r2_score(y_test, y_pred))
    print('Mean Squared Error (test set):', mean_squared_error(y_test, y_pred))

    # Predict on test set
    test_pred_log = grid.predict(test_x)
    final_pred = np.expm1(test_pred_log)

    # Save submission
    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_pred})
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved as 'submission.csv'")

    # Plot actual vs predicted
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel('Actual Log Sale Price')
    plt.ylabel('Predicted Log Sale Price')
    plt.title('Actual vs Predicted Log Sale Prices')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    housing_prices()