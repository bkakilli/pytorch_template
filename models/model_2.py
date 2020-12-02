import torch

class Model2(torch.nn.Module):

    def __init__(self):
        super(Model2, self).__init__()

        self.labelweights = None

    def forward(self, X_batch):
        raise NotImplementedError()

    def get_loss_depr(self, logits, target):
        return torch.nn.functional.cross_entropy(logits, target, weight=self.labelweights)
