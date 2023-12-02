import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, num_features_list, **kwargs):
    super().__init__()
    self.num_features_list = num_features_list
    num_layers = len(num_features_list)
    self.batch_norm = kwargs.get('batch_norm', True)
    self.lambda_transform = kwargs.get('lambda_transform', True)

    layers = []
    
    for i in range(num_layers-1):
        layers.append(nn.Linear(num_features_list[i], num_features_list[i+1]))
        layers.append(nn.BatchNorm1d(num_features = num_features_list[i+1])) if self.batch_norm else None
        layers.append(torch.nn.LeakyReLU())

    layers.append(nn.Linear(num_features_list[-1], 3, bias = True)) # 3 traffics
    self.layers = nn.Sequential(*layers)
   

  def forward(self, x, mu):
    '''Forward pass'''
    mu = nn.functional.normalize(mu, p = 1) if self.lambda_transform else mu
    x = torch.cat((x, mu), dim = -1) # state-augmentation
    y = self.layers(x)
    y = torch.nn.functional.softmax(y, dim = -1)
    return y
  

class MLP_v2(MLP):
  def __init__(self, num_features_list, augment_input_layer = True, n = None):
    super(MLP_v2, self).__init__(num_features_list, augment_input_layer, n)
    
  def forward(self, x, mu):
    '''Forward pass'''

    if self.augment_input_layer:  
      x = torch.cat((x, mu), dim = -1)
    y = self.layers(x)
    if not self.augment_input_layer:
      y = torch.cat((y, mu), dim=-1)
    y = self.final_layer(y)
    y = torch.nn.functional.softmax(y, dim = -1)

    return y


# class dualMLP(MLP):
#   def __init__(self, num_features_list, z_max):
#     super().__init__(num_features_list)
#     self.z_max = z_max

#   def forward(self, x, transmitters_index):
#     y = self.layers(x)
#     if self.num_features_list[0] != 1 or self.num_features_list[0] != 2:
#       y = torch.reshape(y, (-1, 1))

#     if self.z_max == np.Inf:
#       mu_star = torch.relu(y[transmitters_index])
#     else:
#       mu_star = self.z_max * torch.sigmoid(y[transmitters_index])

#     return mu_star

