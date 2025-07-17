import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy.stats import skew,kurtosis
def housing_prices(training_filename='training set.csv',test_filename='test.csv'):
    '''
    This is a machine learning experiment using the housing price dataset from kaggle.
    '''
    #safely open the csv file in python using pandas
    try:
        df = pd.read_csv(training_filename)
        test_df = pd.read_csv(test_filename)
        test_ids = test_df['Id']
    except FileNotFoundError:
        print('File not found')

    #transform the target values (y) to minimize skewing 
    df['logSalePrice'] = np.log1p(df['SalePrice'])

    #drop columns with lots of missing data values
    cols_to_drop = ['PoolQC','MiscFeature','Fence','MasVnrType']
    df.drop(columns=cols_to_drop,inplace=True,errors='ignore')
    test_df.drop(columns=cols_to_drop,inplace=True,errors='ignore')
    ## detect missing data values for each columns in descending order
    # missing = df.isnull().sum()
    # missing = missing[missing > 0].sort_values(ascending=False)
    # missing_percent = (missing / len(df)) * 100
    # missing_df = pd.DataFrame({'Missing Count':missing, 'Missing %':missing_percent})
    # print(missing_df)

    #drop saleprice from the features columns (x)
    x = df.drop(columns=['SalePrice', 'logSalePrice'])
    test_x = test_df.copy()
    #use saleprice as target value
    y = df['logSalePrice']
    # #split feature and target price into training and testing sets
    # x_train,x_test,y_train,y_test = train_test_split(
    #     x,y, test_size=0.3, random_state=42
    # )
    #separate the columns into numerical and categorical features
    num_features = x.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = x.select_dtypes(include='object').columns.tolist()
    #perform eda on numerical values, replacing null values with median values and normalising numerical features
    numerical_transformers = Pipeline(steps=[
        ('imputer',IterativeImputer(random_state=42)),
        ('scaler', StandardScaler()),
        ('poly',PolynomialFeatures(include_bias=False))]
    )
    #perform eda on categorcial values, replacing null values with most frequent values, also enconding them into numerical features
    categorical_transformers = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])
    #tells scikit-learn to apply each transformer to each feature
    preprocessor = ColumnTransformer(transformers=[
        ('num',numerical_transformers,num_features),
        ('cat',categorical_transformers,cat_features)
    ])
    #select ridge for model training,(linear regression + regularization)
    model = Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('model',Ridge())]
    )
    #select hyperparameters for learning and training
    grid_params = {
        'preprocessor__num__poly__degree' : [1,2],
        'model__alpha' : [0.01,0.1,1.0,10.0]
    }
    grid = GridSearchCV(model,grid_params,cv=5,scoring='neg_mean_squared_error')
    #train mmodel
    grid.fit(x,y)
    #apply trained model on testing set
    y_pred = grid.predict(x)
    print('Best polynomial degree : ',grid.best_params_['preprocessor__num__poly__degree'])
    print('Best regularization strength : ',grid.best_params_['model__alpha'])
    print('Best Cross Validation MSE : ',-grid.best_score_)

    print('R^2 score : ',r2_score(y,y_pred))
    print('Mean Squared Error : ',mean_squared_error(y,y_pred))
    # num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # cat_features = df.select_dtypes(include=['object']).columns.tolist()

    # print(f'Number features: {num_features[:5]}')
    # print(f'Categorical features: {cat_features[:5]}')
    test_pred_log = grid.predict(test_x)
    final_pred = np.expm1(test_pred_log)

    submission = pd.DataFrame(
        {
            'Id': test_ids,
            'Saleprice': final_pred
        }
    )
    submission.to_csv('submission.csv',index=False)
    plt.scatter(y,y_pred,alpha=0.6)
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # filename = input('Enter file name (default: training set.csv): ')
    housing_prices()