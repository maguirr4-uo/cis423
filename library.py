
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

#This class maps values in a column, numeric or categorical.
class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  

#This class will rename one or more columns.
class RenamingTransformer(BaseEstimator, TransformerMixin):
  #your __init__ method below
  def __init__(self, renaming_dict:dict):  
    self.renaming_dict = renaming_dict
  #write the transform method without asserts. Again, maybe copy and paste from MappingTransformer and fix up.
  def fit(self, X, y = None):
    print("Warning: RenamingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'RenamingTransformer.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    X_.rename(columns=self.renaming_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=True):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first
  
  #fill in the rest below
  def fit(self, X, y = None):
    print("Warning: OHETransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'OHETransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'OHETransformer.transform unknown column {self.target_column}'
   # X_ = X.copy()
    X_ = pd.get_dummies(X,
                        prefix=self.target_column,
                        prefix_sep='_',    
                        columns=[self.target_column],
                        dummy_na=self.dummy_na,   
                        drop_first=self.drop_first
                        )
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  #fill in rest below
  def fit(self, X, y = None):
    print("Warning: DropColumnsTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'DropColumnsTransformer.transform expected Dataframe but got {type(X)} instead.'
    col_names = set(X.columns.values.tolist())
    drop_set = set(self.column_list)
    drop_set.update(col_names)
    invalid_list = list(drop_set.difference(col_names))
    assert not invalid_list, f'DropColumnsTransformer.transform unkown column {invalid_list}'
   
    if self.action == "drop": 
      X_ = X.drop(columns=self.column_list)
    elif self.action == "keep":
      X_ = X[self.column_list]
    
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
 

class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold):
    assert isinstance(threshold, float), f'PearsonTransformer expects float as an argument but got {type(threshold)} instead.'
    self.threshold = threshold

  #define methods below
  def fit(self, X, y = None):
    print("Warning: PearsonTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'PearsonTransformer.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    df_corr = X_.corr(method='pearson')
    masked_df = df_corr.abs() > self.threshold  # Mask df based on threshold
    upper_mask = np.triu(masked_df, 1)  # Use only the top triangle of the mask
    true_index_list = np.any(upper_mask, axis=0)  # A list of indices of which columns have a true value
    correlated_columns = [j for i, j in enumerate(masked_df) if np.any(true_index_list[i])] # A list of correlated columns
    new_df = transformed_df.drop(correlated_columns, axis=1)  # Dropping the correlated columns
    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  

class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):  
    self.target_column = target_column

  def fit(self, X, y = None):
    print("Warning: Sigma3Transformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'Sigma3Transformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'Sigma3Transformer.transform unknown column {self.target_column}'
    X_ = X.copy()
    low_b, up_b = self.compute_3sigma_bounds(X_, self.target_column)
    X_[self.target_column] = X_[self.target_column].clip(lower=low_b, upper=up_b)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  def compute_3sigma_bounds(self, df, column_name):
    assert isinstance(df, pd.core.frame.DataFrame), f'expected Dataframe but got {type(df)} instead.'
    assert column_name in df.columns.to_list(), f'unknown column {column_name}'
    assert all([isinstance(v, (int, float)) for v in df[column_name].to_list()])

    #compute mean of column - look for method
    m = df[column_name].mean()
    #compute std of column - look for method
    sigma = df[column_name].std()
    return m - 3*sigma, m + 3*sigma #(lower bound, upper bound)
  

class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer'], f'fence should be one of "inner, outer". got {fence} instead.'
    self.target_column = target_column
    self.fence = fence
    
  def fit(self, X, y = None):
    print("Warning: TukeyTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'TukeyTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'TukeyTransformer.transform unknown column {self.target_column}'
    X_ = X.copy()
    q1 = X_[self.target_column].quantile(0.25)
    q3 = X_[self.target_column].quantile(0.75)
    iqr = q3-q1

    if(self.fence == 'inner'):
      low = q1-1.5*iqr
      high = q3+1.5*iqr
    elif(self.fence == 'outer'):
      low = q1-3*iqr
      high = q3+3*iqr
    
    X_[self.target_column] = X_[self.target_column].clip(lower=low, upper=high)

    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    
  #fill in rest below
    pass

  def fit(self, X, y = None):
    print("Warning: MinMaxTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MinMaxTransformer.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    
    X_ = (X_ - X_.min())/(X_.max() - X_.min())

    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform", add_indicator=False):
    self.n_neighbors = n_neighbors
    self.weights=weights 
    self.add_indicator=add_indicator

  def fit(self, X, y = None):
    print("Warning: KNNTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'KNNTransformer.transform expected Dataframe but got {type(X)} instead.'
    
    imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights, add_indicator=self.add_indicator)  #do not add extra column for NaN
    imputed_data = imputer.fit_transform(X)
    np.count_nonzero(np.isnan(imputed_data))  #no NaNs left now

    X_ = pd.DataFrame(imputed_data, columns = X.columns)

    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
# This function finds a random value (random state) through variance splitting.
# Utilizes Sklearn's F1 Score
def find_random_state(df, labels, n=200):
  var = []  #collect test_error/train_error where error based on F1 score
  model = LogisticRegressionCV(random_state=1, max_iter=5000)

  for i in range(1, n):
      train_X, test_X, train_y, test_y = train_test_split(df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)  #predict against training set
      test_pred = model.predict(test_X)    #predict against test set
      train_error = f1_score(train_y, train_pred)  #how bad did we do with prediction on training data?
      test_error = f1_score(test_y, test_pred)     #how bad did we do with prediction on test data?
      error_ratio = test_error/train_error        #take the ratio
      var.append(error_ratio)

  rs_value = sum(var)/len(var)

  idx = np.array(abs(var - rs_value)).argmin()

  return idx
