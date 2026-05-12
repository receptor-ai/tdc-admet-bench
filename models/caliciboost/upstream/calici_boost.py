from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

def add_padel_descriptors(data_df: pd.DataFrame, name: str):

  # descriptor = data_df['Drug'].apply(get_descriptors)

  descriptor_df = pd.read_csv('PaDEL_Descriptors.csv')
  data_padel_df = pd.merge(data_df, descriptor_df, on=['Drug_ID', 'Drug'], how='left')
#   data_padel_df.to_csv(f'{name}_with_loaded_padel.csv', index=False)
  return data_padel_df


def clean_data(train_data_df: pd.DataFrame, test_data_df: pd.DataFrame):
  # remove the train data without padel descriptors (got error when generating)
  nan_mask = train_data_df.iloc[:, 3:].isnull().all(axis=1)
  train_data_df = train_data_df[~nan_mask]
  
  # remove the duplicate lines, which smiles is same and descriptor values are all same
  train_data_df['drug_lower'] = train_data_df['Drug'].str.lower()
  check_df = pd.concat([train_data_df['drug_lower'], train_data_df.iloc[:, 3:]], axis=1)
  duplicate_mask = check_df.duplicated(keep='first')
  np.where(duplicate_mask)
  train_clean_df = train_data_df[~duplicate_mask].drop(columns=['drug_lower'])
  
#   train_clean_df.to_csv('train_clean.csv', index=False)
#   test_data_df.to_csv('test_clean.csv', index=False)
  return train_clean_df, test_data_df


gb_params = {
  "n_estimators": 500,
  "max_depth": 4,
  "min_samples_split": 5,
  "learning_rate": 0.01,
  "loss": "squared_error",
  }


xg_parmas = {
  'subsample': 0.8, 
  'scale_pos_weight': 3, 
  'reg_lambda': 1.2, 
  'reg_alpha': 0.5, 
  'n_estimators': 200, 
  'min_child_weight': 1, 
  'max_depth': 9, 
  'max_delta_step': 3, 
  'learning_rate': 0.05, 
  'gamma': 0.2, 
  'colsample_bytree': 0.7
  }


feature_candidates = {
    1: [1200, 1201, 674, 521, 1253, 0, 166, 793, 140, 569, 440, 1396, 1207, 329, 322, 230, 1208, 1441, 929, 325, 251, 1144, 782, 1407, 649, 452, 463, 29, 1020, 506, 514, 565, 153, 1246, 425, 604, 588, 1465, 220, 1269, 178, 171, 837, 102, 566, 132, 1069, 214, 250, 553],
    2: [1200, 1201, 674, 1253, 521, 0, 166, 793, 569, 440, 782, 140, 230, 322, 1396, 1441, 325, 1208, 1207, 425, 29, 929, 463, 1020, 329, 649, 452, 514, 1144, 178, 1407, 837, 251, 1246, 1409, 506, 1270, 250, 153, 220, 132, 1075, 565, 1465, 102, 566, 553, 171, 588, 1269, 1025, 334, 214, 286, 1448, 493, 1639, 318, 340, 968, 1190, 312, 1151, 604, 632, 89, 1069, 313, 137, 532, 326, 663, 617, 152, 201],
    3: [1200, 674, 1201, 1253, 521, 140, 0, 166, 793, 569, 440, 230, 782, 322, 1396, 463, 1020, 1, 1207, 425, 1441, 1208, 929, 506, 1407, 29, 251, 649, 325, 514, 329, 1246, 178, 1144, 250, 565, 132, 1251, 452, 133, 102, 220, 553, 1409, 364, 152, 617, 214, 318, 153, 604, 915, 588, 1448, 89],
    4: [1200, 1253, 674, 1201, 521, 0, 166, 140, 440, 569, 230, 782, 1207, 793, 1020, 322, 29, 1208, 1441, 506, 325, 1396, 1270, 1407, 251, 514, 649, 329, 837, 1409, 153, 1246, 425, 604, 1144, 1269, 1, 553, 463, 220, 214, 132, 364, 1231, 452, 1465, 102, 929, 171, 566, 956, 389, 1069, 250, 318, 286, 1151, 493, 1448, 178, 945, 588, 340, 312, 565, 1639, 273, 336, 201, 476],
    5: [1200, 1253, 1201, 674, 521, 0, 166, 140, 440, 569, 230, 793, 1207, 782, 322, 1208, 425, 29, 1020, 329, 463, 1441, 506, 325, 1409, 1396, 649, 1407, 251, 1144, 514, 1246, 1, 929, 553, 132, 102, 171, 364, 318, 220, 1465, 566, 1277, 1231, 286, 250, 178, 493, 1025, 214, 1448, 588, 133, 201, 1639, 1190, 1270, 389, 1269, 326, 963, 532, 196, 1273, 945, 1192, 334, 137, 312, 604, 153, 632]
}


def featurize(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int):
  x_train = train_df.iloc[:, 3:].to_numpy()
  y_train = train_df.iloc[:, 2].to_numpy()

  x_test = test_df.iloc[:, 3:].to_numpy()
  y_test = test_df.iloc[:, 2].to_numpy()

  scaler = StandardScaler()
  scaler.fit(x_train)
  x_train_scaled = scaler.transform(x_train)
  x_test_scaled = scaler.transform(x_test)
  
  feature_idxs = feature_candidates[seed]

  return x_train_scaled[:, feature_idxs], y_train, x_test_scaled[:, feature_idxs], y_test