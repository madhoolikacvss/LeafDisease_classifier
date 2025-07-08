import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import pandas as pd
df = pd.read_csv('./lables.csv')

from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mobilenet_model = models.mobilenet_v2(pretrained=True)
for param in mobilenet_model.features.parameters():
    param.requires_grad = False


#========UNCOMMNET THIS FOR BINARY CLASS MODEL==================
# mobilenet_model.classifier = nn.Sequential(
#     nn.Linear(mobilenet_model.last_channel, 128),
#     nn.ReLU(inplace=True),
#     nn.Linear(128, 2)  # Binary classification
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
mobilenet_model.classifier = nn.Sequential(
    nn.Linear(mobilenet_model.last_channel, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 14)  # Binary classification
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

criterion_mob = nn.CrossEntropyLoss(weight=weights)
optimizer_mob = optim.Adam(mobilenet_model.classifier.parameters())
