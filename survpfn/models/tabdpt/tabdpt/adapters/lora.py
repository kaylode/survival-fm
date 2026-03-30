from functools import partial
import torch

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


def get_lora_adapted_function(lora_r, lora_alpha):
    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
    return assign_lora

def add_lora_to_linear_layer(
        model,
        lora_r = 8,
        lora_alpha = 16,
        lora_query = True,
        lora_key = False,
        lora_value = True,
        lora_projection = False,
        lora_mlp = False,
        lora_head = False,
    ):
    """Wraps a given nn.Linear layer with LoRA adaptation.
    """

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

    for layer in model.distilbert.transformer.layer:
        if lora_query:
            layer.attention.q_lin = assign_lora(layer.attention.q_lin)
        if lora_key:
            layer.attention.k_lin = assign_lora(layer.attention.k_lin)
        if lora_value:
            layer.attention.v_lin = assign_lora(layer.attention.v_lin)
        if lora_projection:
            layer.attention.out_lin = assign_lora(layer.attention.out_lin)
        if lora_mlp:
            layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
            layer.ffn.lin2 = assign_lora(layer.ffn.lin2)
    if lora_head:
        model.pre_classifier = assign_lora(model.pre_classifier)
        model.classifier = assign_lora(model.classifier)

    return model