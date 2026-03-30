from .adapters.lora import get_lora_adapted_function
import torch
from .model import TabDPTModel

def get_lora_tabdpt(
        model: TabDPTModel,
        lora_r = 8,
        lora_alpha = 16,
        lora_qkv = True,
        lora_mlp = True,
        lora_projection = True,
        lora_encoder = True,
        lora_y_encoder = True,
        lora_head = True,
    ):
    """Adds LoRA adaptation to specified linear layers in TabDPTModel.
    """

    for param in model.parameters():
        param.requires_grad = False

    assign_lora = get_lora_adapted_function(lora_r, lora_alpha)

    if lora_qkv:
        for layer in model.transformer_encoder:
            if lora_qkv:
                layer.kv_proj = assign_lora(layer.kv_proj)
                layer.q_proj = assign_lora(layer.q_proj)
            if lora_projection:
                layer.out_proj = assign_lora(layer.out_proj)
            if lora_mlp:
                layer.ff[0] = assign_lora(layer.ff[0])  # first Linear in ff
                layer.ff[2] = assign_lora(layer.ff[2])  # second Linear in ff
    if lora_encoder:
        model.encoder = assign_lora(model.encoder)
    if lora_y_encoder:
        model.y_encoder = assign_lora(model.y_encoder)
    if lora_head:
        for i, layer in enumerate(model.head):
            if isinstance(layer, torch.nn.Linear):
                model.head[i] = assign_lora(layer)

    return model

if __name__ == "__main__":

    # Example usage
    model = TabDPTModel(
        dropout=0.1,
        n_out=10,
        nhead=4,
        nhid=128,
        ninp=64,
        nlayers=2,
        num_features=20,
    )

    lora_model = get_lora_tabdpt(
        model,
        lora_r=4,
        lora_alpha=8,
        lora_qkv=True,
        lora_mlp=True,
        lora_projection=True,
        lora_encoder=False,
        lora_y_encoder=False,
        lora_head=True,
    )

    breakpoint()