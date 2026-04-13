from __future__ import annotations

from dataclasses import asdict
from typing import cast
import torch
import torch.nn as nn
from tabpfn import model_loading
from tabpfn import TabPFNClassifier
from tabpfn.architectures import tabpfn_v2_5
from tabpfn.architectures.base.transformer import PerFeatureTransformer
from importlib.resources import files
from pathlib import Path

TABPFN_CHECKPOINT_PATH: Path = files("survpfn.models.models_diff") / "tabpfn-v2.5-classifier-v2.5_default.ckpt"

def load_model_workflow(device='cuda', only_inference=True, finetune=False):
	loaded_models, _, loaded_configs, _ = model_loading.load_model_criterion_config(
		model_path=TABPFN_CHECKPOINT_PATH,
		check_bar_distribution_criterion=False,
		cache_trainset_representation=False,
		which="classifier",
		version="v2.5",
		download_if_not_exists=True,
	)
	arch_base = cast(PerFeatureTransformer, loaded_models[0])
	config_base = loaded_configs[0]
	model = tabpfn_v2_5.get_architecture(
		tabpfn_v2_5.TabPFNV2p5Config(**asdict(config_base)),
		cache_trainset_representation=False,
	)
	model.load_state_dict(arch_base.state_dict(), strict=True)
	model.to(torch.float32)
	model.to(device)

	if only_inference:
		model.eval()
		# Freezing logic
		for param in model.parameters():
			param.requires_grad = False

	elif finetune:
		for name, param in model.named_parameters():
			if "output_projection" in name:
				param.requires_grad = True
			else:
				param.requires_grad = False
		


	class Wrapper(nn.Module):
		def __init__(self):
			super().__init__()
			self.model = model
			# Expose the embedding dimension
			if hasattr(model, "d_model"):
				self.ninp = model.d_model
			elif hasattr(config_base, "d_model"):
				self.ninp = config_base.d_model
			else:
				# Fallback for standard TabPFN v2.5
				self.ninp = 512
		def forward(self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor=None):
			x_train = x_train.to(torch.float32)  # N_sample, batch, Feats
			y_train = y_train.to(torch.float32)
			if x_test is not None:
				x_test = x_test.to(torch.float32)
				x = torch.cat([x_train, x_test], dim=0)
			else:
				x = x_train

			if len(x.shape) == 2:
				x = x.unsqueeze(1)

			x = x.to(device)
			y_train = y_train.to(device)
				

			output = model(x, y_train, only_return_standard_out=False)

			return {
				"logits": output["standard"],  # N_test_sample, batch, 10
				"train_embeddings": output["train_embeddings"],
				"test_embeddings": output["test_embeddings"],
			}
	

	return Wrapper()

def get_classifier(device="cuda"):
	return TabPFNClassifier(
		model_path=TABPFN_CHECKPOINT_PATH,
		n_estimators=1,
        device=device,
	)