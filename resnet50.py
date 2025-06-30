import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import sys

## RESNET50 NETWORK
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze backbone

# replacibng FC allows it to train
model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 2)
)

n_healthy = 221
n_unhealthy = 3179
total = num_healthy + num_unhealthy

weights = tensor([
    total / (2 * num_healthy),    # class 0 (healthy)
    total / (2 * num_unhealthy)   # class 1 (unhealthy)
], dtype=torch.float).to(device)

model = model.to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.fc.parameters())






# In[ ]:




