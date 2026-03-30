from typing import *

import torch
from theseus import move_to
from ehrmeds import FeatureTypes, get_pretrained_icd_encoder

class PromptingPipeline():
	"""
	Process EHR data to prepare sequential ICD embeddings.
	"""

	def __init__(
			self,  
			data,
			mode:str='train',
			**kwargs
		):
		(
			X_train, y_train,
			X_test, y_test,
			feature_names,
			feature_type_ids,
			categorical_indices,
			categorical_cardinalities
		) = (
			data["X_train"], data['y_train'],
			data["X_test"], data['y_test'],
			data["feature_names"],
			data['feature_type_ids'],
			data['categorical_feature_indices'],
			data['categorical_cardinalities'],
		)
		self.task = 'cls' if data['is_classification'] else 'reg'
		self.mode = mode
		
		self.feature_names = feature_names
		self.feature_type_ids = feature_type_ids
		self.categorical_indices = categorical_indices
		self.categorical_cardinalities = categorical_cardinalities

		# Identify feature indices by type
		self.lab_indices = [
			i for i, feat_type in enumerate(self.feature_type_ids) 
			if feat_type == FeatureTypes.LAB.value
		]
		self.icd_indices = [
			i for i, feat_type in enumerate(self.feature_type_ids) 
			if feat_type == FeatureTypes.DIAGNOSIS.value
		]
		self.icd_names = [
			self.feature_names[i].replace('.', '').split('//')[-1] 
			for i in self.icd_indices
		]

		self.numerical_indices = [
			i for i in range(len(self.feature_type_ids))
			if i not in self.lab_indices + self.icd_indices + self.categorical_indices
		]

	def format_batch(
			self, X_batch, y_batch, task, 
			icd_indices, lab_indices, 
			numerical_indices, categorical_indices, 
			device=None
		):
		batch_dict = {}
		batch_dict['y'] = y_batch
		batch_dict['task'] = task

		# Structure data by modality
		batch_dict['x_num'] = X_batch[..., numerical_indices+lab_indices]
		batch_dict['x_cat'] = X_batch[..., categorical_indices].long()
	
		if len(icd_indices):
			batch_dict['x_icd'] = (X_batch[..., icd_indices] > 0).long()
		
		if device is not None:
			batch_dict = move_to(batch_dict, device)
			
		return batch_dict
	
	def format_batch_normal(self, x: torch.Tensor, y: torch.Tensor, tasks=None, device=None, **kwargs):
		batch_dict = {
			'x': x,
			'y': y,
		}
		if tasks is not None:
			batch_dict['task'] = tasks
		if device is not None:
			batch_dict = move_to(batch_dict, device)
		return batch_dict

		
	def __call__(self, Xs, ys, tasks, device=None, **kwargs):
		"""Optimized batch processing."""
		
		# batch_dict = self.format_batch(
		# 	Xs, ys, tasks,
		# 	self.icd_indices,
		# 	self.lab_indices,
		# 	self.numerical_indices,
		# 	self.categorical_indices,
		# 	device=device
		# )
		batch_dict = self.format_batch_normal(
			Xs, ys, tasks=tasks, device=device
		)

		return batch_dict