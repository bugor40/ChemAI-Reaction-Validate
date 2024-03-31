#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm_notebook
from IPython.display import clear_output


# In[2]:


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        self.fc2 = nn.Linear(latent_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# In[ ]:


def model_fit(data_tensor, validation_data_tensor, learning_rate, num_epochs, input_size, hidden_size, latent_size):
    
    model = VAE(input_size, hidden_size, latent_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    epoch_list = []
    loss_list = []
    loss_list_val = []

    for epoch in tqdm_notebook(range(num_epochs)):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data_tensor)
        BCE = F.mse_loss(recon_batch, data_tensor, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        loss.backward()
        optimizer.step()
        
        
        with torch.no_grad():
            val_recon_batch, val_mu, val_logvar = model(validation_data_tensor)
            val_BCE = F.mse_loss(val_recon_batch, validation_data_tensor, reduction='sum')
            val_KLD = -0.5 * torch.sum(1 + val_logvar - val_mu.pow(2) - val_logvar.exp())
            val_loss = val_BCE + val_KLD
            
        epoch_list.append(epoch+1)
        loss_list.append(loss)
        loss_list_val.append(val_loss)
        

        pd.DataFrame(np.array([epoch_list, loss_list]).T, columns = ['epoch', 'loss'])       .to_csv(f'epoch_{num_epochs}_hidden_size_{hidden_size}_latent_size_{latent_size}.tsv', index = 0, sep = '\t')

        clear_output(wait=True)
        plt.plot(epoch_list, loss_list, label = 'train')
        plt.plot(epoch_list, loss_list_val, label = 'val')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'epoch_{num_epochs}_hidden_size_{hidden_size}_latent_size_{latent_size}.png')
        plt.show()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, BCE: {BCE.item():.4f}, KLD: {KLD.item():.4f}')
        
    return model


# In[3]:


def model_predict(model, data_tensor, latent_size):
    with torch.no_grad():
        latent_representation = model.encode(data_tensor)[0].numpy()
        
    return pd.DataFrame(latent_representation)


# In[ ]:




