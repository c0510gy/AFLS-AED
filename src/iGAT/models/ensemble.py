import torch
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            for model in self.models:
                outputs += F.softmax(model(x), dim=-1)
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](x)
            
class Ensemble_max(nn.Module):
    def __init__(self, models):
        super(Ensemble_max, self).__init__()
        self.models = models
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = F.softmax(self.models[0](x), dim=-1)
            for i in range(1, len(self.models)):
                outputs = torch.max(outputs, F.softmax(self.models[i](x), dim=-1))
            outputs = torch.clamp(outputs, min=1e-40)
            return torch.log(outputs)
        else:
            return self.models[0](x)