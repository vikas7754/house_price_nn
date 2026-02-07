import torch.nn as nn

class HousePriceNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input features: 5
        self.net = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)
