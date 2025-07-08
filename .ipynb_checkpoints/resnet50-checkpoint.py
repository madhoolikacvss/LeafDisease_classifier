import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import sys
import pandas as pd
df = pd.read_csv('./lables.csv')
from collections import Counter
## RESNET50 NETWORK
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet_model = models.resnet50(pretrained=True)
for param in resnet_model.parameters():
    param.requires_grad = False  # freeze backbone
    
#========UNCOMMNET THIS FOR BINARY CLASS MODEL==================
# replacibng FC allows it to train
# resnet_model.fc = nn.Sequential(
#     nn.Linear(2048, 128),
#     nn.ReLU(inplace=True),
#     nn.Linear(128, 2)
# )
# n_healthy = 221
# n_unhealthy = 3179
# total = n_healthy + n_unhealthy

# weights = torch.tensor([
#     total / (2 * n_healthy),    # class 0 (healthy)
#     total / (2 * n_unhealthy)   # class 1 (unhealthy)
# ], dtype=torch.float).to(device)
#---------------------------------------------------------------

#========UNCOMMENT THIS FOR MULTI-CLASS MODEL===================
resnet_model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 14)
)
class_counts = Counter(df['multi_cat'])
total = sum(class_counts.values())
num_classes = 14

valid_classes = [i for i in range(num_classes) if class_counts.get(i, 0) > 0]
total = sum([class_counts[i] for i in valid_classes])
num_valid_classes = len(valid_classes)
weights_list = [
    total / (num_valid_classes * class_counts[i])
    for i in valid_classes
]

weights = torch.zeros(num_classes)
for i, cls in enumerate(valid_classes):
    weights[cls] = weights_list[i]

weights = weights.to(device)


resnet_model = resnet_model.to(device)
criterion_res = nn.CrossEntropyLoss(weight=weights)
optimizer_res = optim.Adam(resnet_model.fc.parameters())






