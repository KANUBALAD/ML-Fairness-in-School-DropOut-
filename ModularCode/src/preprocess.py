import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(data):
    """
    Preprocesses the covariates data by standardizing numerical columns and one-hot encoding categorical columns.

    Parameters:
    data (pd.DataFrame): The feature dataframe.

    Returns:
    pd.DataFrame: The preprocessed feature dataframe.
    """
    # Select the numerical and categorical features
    numerical_features = data.select_dtypes(include=[float, int]).columns
    categorical_features = data.select_dtypes(include=[object]).columns
    
    # Create the numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    # Check if there are categorical features
    if len(categorical_features) > 0:
        # Create the categorical pipeline only if there are categorical features
        categorical_pipeline = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
        # Define the column transformer with both pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)])
    else:
        # If no categorical features, only apply the numerical pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features)])
    
    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(data)
    
    # Get feature names
    feature_names = list(numerical_features)
    
    # If there are categorical features, append their transformed names
    if len(categorical_features) > 0:
        feature_names += list(preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features))
    
    # Create a DataFrame with the transformed data and appropriate column names
    X_transformed_df = pd.DataFrame(X_transformed, index=data.index, columns=feature_names)
    
    return X_transformed_df
