
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
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
