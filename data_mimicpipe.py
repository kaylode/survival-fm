from typing import *
from ehrmeds.utils.logger import LoggerObserver
from ehrmeds.utils.cohort import DISEASE_GROUP_TO_ICDS
import json
import numpy as np
import pandas as pd
import os.path as osp
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .utils import DataPreprocessor
from ehrmeds.utils.feature_types import FEATURE_TYPES_IDS
from ehrmeds.utils.selection import select_features

RARE_DISEASE_ICDS = [i for key, i in DISEASE_GROUP_TO_ICDS.items() if key in ['HCAI']]
RARE_DISEASE_ICDS = [item for sublist in RARE_DISEASE_ICDS for item in sublist]
RARE_DISEASE_ICDS.extend([i[:5] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS.extend([i[:4] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS.extend([i[:3] for i in RARE_DISEASE_ICDS])
RARE_DISEASE_ICDS = list(set(RARE_DISEASE_ICDS))

LOGGER = LoggerObserver.getLogger("main")


def get_mimic_dataset(
		data_csv:str, 
		train_label_path:str=None, test_label_path:str=None,
		feature_selection_path:str=None, selection_method:str=None,
		include_icd:bool=True
	):

	train_ids_df = pd.read_csv(train_label_path).drop('admission_id', axis=1, errors='ignore').drop_duplicates()
	train_ids_df = train_ids_df.groupby(['subject_id', 'stay_id', 'hadm_id']).agg({
		'label': 'max'
	}).reset_index() # remove duplicates
	test_ids_df = pd.read_csv(test_label_path).drop('admission_id', axis=1, errors='ignore').drop_duplicates()
	test_ids_df = test_ids_df.groupby(['subject_id', 'stay_id', 'hadm_id']).agg({
		'label': 'max'
	}).reset_index() # remove duplicates

	df = pd.read_csv(data_csv)
	df = df.drop_duplicates(subset=['subject_id', 'stay_id', 'hadm_id'], keep='first')
	df = df.drop(columns=['label'], errors='ignore', axis=1) # use label from split file

	train_df = pd.merge(
		df,
		train_ids_df,
		left_on=["subject_id", "stay_id", 'hadm_id'],
		right_on=["subject_id", "stay_id", 'hadm_id'],
		how="inner"
	)

	test_df = pd.merge(
		df,
		test_ids_df,
		left_on=["subject_id", "stay_id", 'hadm_id'],
		right_on=["subject_id", "stay_id", 'hadm_id'],
		how="inner"
	)

	train_ids_df = pd.DataFrame({
		"subject_id": train_df["subject_id"],
		"stay_id": train_df["stay_id"],
		"hadm_id": train_df["hadm_id"],
	})

	test_ids_df = pd.DataFrame({
		"subject_id": test_df["subject_id"],
		"stay_id": test_df["stay_id"],
		"hadm_id": test_df["hadm_id"],
	})

	X_train_df = train_df.drop(columns=['label', 'subject_id', 'hadm_id', 'stay_id', 'edstay_id', 'admission_id', 'icustay_id'], errors='ignore')
	y_train_df = train_df['label']
	X_test_df = test_df.drop(columns=['label', 'subject_id', 'hadm_id', 'stay_id', 'edstay_id', 'admission_id', 'icustay_id'], errors='ignore')
	y_test_df = test_df['label']

	if not include_icd:
		# Remove ICD columns
		icd_columns = [col for col in X_train_df.columns if col.startswith(('ICD', 'CPT'))]
		X_train_df = X_train_df.drop(columns=icd_columns, errors='ignore')
		X_test_df = X_test_df.drop(columns=icd_columns, errors='ignore')

	categorical_feature_indices = [idx for idx, i in enumerate(X_train_df.columns.tolist()) if X_train_df[i].dtype == 'object']
	feature_names = X_train_df.columns.tolist()
	icd10_indices = [i for i in range(len(feature_names)) if feature_names[i].startswith(('ICD', 'CPT'))]
	
	label_encoder = [LabelEncoder() for _ in categorical_feature_indices]
	for idx, le in zip(categorical_feature_indices, label_encoder):
		X_train_df.iloc[:, idx] = le.fit_transform(X_train_df.iloc[:, idx])
		X_test_df.iloc[:, idx] = le.transform(X_test_df.iloc[:, idx])

	# Create new column with ICD10/CPT codes for each row
	code_train = X_train_df.iloc[:, icd10_indices].fillna(0)
	code_test = X_test_df.iloc[:, icd10_indices].fillna(0)
	# Create list of lists of column names where value > 0 for each row
	raw_codes_train = []
	for idx, row in code_train.iterrows():
		active_codes = [feature_names[icd10_indices[i]] for i, val in enumerate(row) if val > 0]
		raw_codes_train.append(active_codes)
	
	raw_codes_test = []
	for idx, row in code_test.iterrows():
		active_codes = [feature_names[icd10_indices[i]] for i, val in enumerate(row) if val > 0]
		raw_codes_test.append(active_codes)

	# Fit your data on the scaler object
	# Different imputation for different columns
	icd10_columns = [feature_names[i] for i in icd10_indices]
	categorical_columns = [feature_names[i] for i in categorical_feature_indices]
	preprocessor = DataPreprocessor(
		impute_method={
			icd_col: "constant" for icd_col in icd10_columns
		},
		scaler = {
			col: "none" for col in categorical_columns # disable scaling for categorical columns
		},
		default_impute_method="mean",  # All other columns will use mean
		default_scaler="standard",  # Default for unspecified columns
		feature_names=feature_names
	)
	X_train_df = preprocessor.fit_transform(X_train_df)
	X_test_df = preprocessor.transform(X_test_df)

	X_train, y_train, feature_names, selected_indices = select_features(
		X_train_df, y_train_df, selection_method=selection_method, 
		feature_selection_path=feature_selection_path, 
		selected_indices=None
	)

	X_test, y_test, _, _ = select_features(
		X_test_df, y_test_df, selection_method=selection_method,
		feature_selection_path=feature_selection_path,
		selected_indices=selected_indices
	)
	class_distribution = y_train.value_counts(normalize=True).to_dict()
	categorical_feature_indices = [idx for idx, i in enumerate(X_train.columns.tolist()) if i in categorical_columns]
	categorical_cardinalities = [
		int(X_train.iloc[:, idx].nunique()) for idx in categorical_feature_indices
	]
	feature_type_ids = [FEATURE_TYPES_IDS.get(feat.split("//")[0] + "//", 0) for feat in feature_names]
	X_train = X_train.values.astype(np.float64)
	X_test = X_test.values.astype(np.float64)
	y_train = y_train.values
	y_test = y_test.values

	LOGGER.text(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}", level=LoggerObserver.INFO)
	LOGGER.text(f"Train label distribution: {pd.Series(y_train).value_counts(normalize=True)}", level=LoggerObserver.INFO)
	LOGGER.text(f"Test label distribution: {pd.Series(y_test).value_counts(normalize=True)}", level=LoggerObserver.INFO)

	return {
		'X_train': X_train, 'y_train': y_train,  
		'X_test': X_test, 'y_test': y_test, 
		"X_train_original": X_train_df,
		"X_test_original": X_test_df,
		'categorical_feature_indices': categorical_feature_indices,
		"categorical_cardinalities": categorical_cardinalities,
		'feature_names': feature_names,
		'train_ids': train_ids_df,
		'test_ids': test_ids_df,
		'class_distribution': class_distribution,
		'icd10_indices': icd10_indices,
		'raw_codes_train': raw_codes_train,
		'raw_codes_test': raw_codes_test,
		"is_classification": True,
		"num_classes": len(np.unique(y_train)),
		"feature_type_ids": feature_type_ids,
	}