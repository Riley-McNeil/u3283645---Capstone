from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Ridge ,Lasso
from sklearn.preprocessing import StandardScaler ,PolynomialFeatures , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from category_encoders import BinaryEncoder
from sklearn.metrics import mean_squared_error
from Test_2 import df


def main(model, param_grid, X_train, X_test, y_train, y_test):

    print("predictive model using scikit-learn")
    print('--------------------------------------')
    # Hyper-parameter tuning, gets a grid search array, linear regresion,
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print("Grid search param: ", grid_search.best_params_)
    print("Mean of cross-validation: ", grid_search.best_score_)
    print('--------------------------------------')

    # Measuring accuracy and error
    print("Training accuracy:", round(model.score(X_train, y_train), 4) * 100)
    print("Testing accuracy:", round(model.score(X_test, y_test), 4) * 100)
    print("Mean squared error: ", mean_squared_error(y_test, model.predict(X_test)))

    return model

features = df.columns.drop(['Price'])
target = 'Price'

X = df[features]
y = df[target]

num_features = X.select_dtypes('number').columns
cat_features = X.select_dtypes('object').columns

X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


def pipe_line():
    num_pipeline = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        PolynomialFeatures(degree=2)
    )

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )

    preprocessor = make_column_transformer(
        (num_pipeline,num_features),
        (cat_pipeline,cat_features)
    )

    Linear_reg = make_pipeline(
        preprocessor,
        LinearRegression()
    )

    ridge_reg = make_pipeline(
        preprocessor,
        Ridge()
    )

    lasso_reg = make_pipeline(
        preprocessor,
        Lasso()
    )
    return Linear_reg, ridge_reg, lasso_reg, cat_pipeline, num_pipeline
# hyper param space search using estimator 'transformer'
def hyper_param(Linear_reg, ridge_reg, lasso_reg):
    param_grid = {
        'columntransformer__pipeline-1__polynomialfeatures__degree': [2, 3, 4]
    }
    linear_reg = main(Linear_reg, param_grid, X_train, X_test, y_train, y_test)

    param_grid_ridge = {
        'ridge__alpha': [1e-16, 1e-17, 1e-18],
        'columntransformer__pipeline-1__polynomialfeatures__degree': [2, 3, 4]
    }
    ridge_reg = main(ridge_reg, param_grid_ridge, X_train, X_test, y_train, y_test)

    param_grid_lasso = {
        "lasso__alpha": [1e-5, 1e-6, 1e-7],
        "columntransformer__pipeline-1__polynomialfeatures__degree": [2, 3, 4]
    }
    lasso_reg = main(lasso_reg, param_grid, X_train, X_test, y_train, y_test)
    return lasso_reg, ridge_reg, linear_reg

features = df.columns.drop(['Price'])
target = 'Price'

X = df[features]
y = df[target]

num_features = X.select_dtypes('number').columns
cat_features = X.select_dtypes('object').columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Linear_reg, ridge_reg, lasso_reg, cat_pipeline, num_pipeline = pipe_line()
lasso_reg, ridge_reg, linear_reg = hyper_param(Linear_reg, ridge_reg, lasso_reg)