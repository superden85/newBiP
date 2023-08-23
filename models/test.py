import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GetSubnetUnstructured


class Mini(nn.Module):

    def __init__(self, conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
        super(Mini, self).__init__()
        self.hidden_layer = linear_layer(in_features=2, out_features=1, bias=True)
        self.k = kwargs['k'] if 'k' in kwargs else None
        self.unstructured_pruning = kwargs['unstructured'] if 'unstructured' in kwargs else False

    def _forward_impl(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel
        else:
            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            vec.w = attr
                            pointer += attr.numel()
        
        x = self.hidden_layer(x)
        #choose sigmoid as activation function
        x = torch.sigmoid(x)
        return x
    
    def forward(self, x):
        return self._forward_impl(x)
        

def mini_model(conv_layer, linear_layer, init_type='kaiming_normal', **kwargs):
    assert init_type == "kaiming_normal", "only supporting kaiming_normal init"
    model = Mini(conv_layer, linear_layer, init_type, **kwargs)
    return model