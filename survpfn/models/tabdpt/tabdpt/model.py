import torch
import torch.nn as nn
from omegaconf import DictConfig

from .transformer_layer import TransformerEncoderLayer
from .models_utils import clip_outliers, normalize_data

from importlib.resources import files
from pathlib import Path
TABDPT_CHECKPOINT_PATH: Path = files("survpfn.models.models_diff") / "tabdpt1_1.pth"

class TabDPTModel(nn.Module):
    def __init__(
        self,
        dropout: float,
        n_out: int,
        nhead: int,
        nhid: int,
        ninp: int,
        nlayers: int,
        num_features: int,
        freeze_transformer:bool=False
    ):
        """TabDPTModel initialization.

        Args:
            dropout (float): Dropout rate.
            n_out (int): Number of output classes.
            nhead (int): Number of attention heads.
            nhid (int): Hidden dimension.
            ninp (int): Input dimension.
            nlayers (int): Number of transformer layers.
            num_features (int): Number of input features.
        """
        super().__init__()
        self.n_out = n_out  # number of output classes
        self.ninp = ninp  # embedding dimension
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=ninp,
                    num_heads=nhead,
                    ff_dim=nhid,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp)
        self.dropout = nn.Dropout(p=dropout)
        self.y_encoder = nn.Linear(1, ninp)
        self.head = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out + 1))

        if freeze_transformer:
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        eval_pos: int = None,
        return_log_act_norms: bool = False,
        return_embeddings: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass of the TabDPTModel.
        Args:
            x_src (torch.Tensor): Input features of shape (time, batch, hidden dimension).
            y_src (torch.Tensor): Target values of shape (T, B).
            return_log_act_norms (bool): Whether to return activation norms for logging.
        Returns:
            torch.Tensor: Predicted values of shape (T, B, n_out + 1).
        """

        if eval_pos is None:
            eval_pos = y.shape[0]

        y_src = y[:eval_pos]

        while len(y_src.shape) < len(x.shape):
            y_src = y_src.unsqueeze(-1)

        # preproces features by normalizing and clipping outliers
        x_src = clip_outliers(x, -1 if self.training else eval_pos, n_sigma=4)
        x_src = normalize_data(x_src, -1 if self.training else eval_pos)
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=4)
        x_src = torch.nan_to_num(x_src, nan=0)

        # feature encoding
        x_src = self.encoder(x_src)
        mean = (x_src**2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean)
        x_src = x_src / rms

        # target encoding
        y_src = self.y_encoder(y_src)
        train_x = x_src[:eval_pos] + y_src
        src = torch.cat([train_x, x_src[eval_pos:]], 0)

        log_act_norms = {}
        log_act_norms["y"] = torch.norm(y_src, dim=-1).mean()

        # transformer layers
        for l, layer in enumerate(self.transformer_encoder):
            if l in [0, 1, 3, 6, 9]:
                log_act_norms[f"layer_{l}"] = torch.norm(src, dim=-1).mean()
            src = layer(src, eval_pos)

        # final head
        pred = self.head(src)

        if return_log_act_norms:
            return pred[eval_pos:], log_act_norms
            
        if return_embeddings:
            return pred[eval_pos:], src

        return pred[eval_pos:]

    @classmethod
    def load(cls, model_state: dict, config: DictConfig) -> nn.Module:
        """Load a pre-trained TabDPTModel from a state dictionary.

        Args:
            cls: TODO
            model_state (dict): state dictionary containing the model parameters.
            config (DictConfig): configuration object containing model parameters.

        Returns:
            nn.Module: model instance with loaded parameters.
        """
        # TODO loading model inside its own class without self?
        assert config.model.max_num_classes > 2
        model = TabDPTModel(
            dropout=config.training.dropout,
            n_out=config.model.max_num_classes,
            nhead=config.model.nhead,
            nhid=config.model.emsize * config.model.nhid_factor,
            ninp=config.model.emsize,
            nlayers=config.model.nlayers,
            num_features=config.model.max_num_features,
            freeze_transformer=config.model.get("freeze_transformer", False)
        )

        module_prefix = "_orig_mod."
        model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        model.to(config.env.device)
        model.eval()
        # print("Loaded TabDPTModel from checkpoint.")
        return model
