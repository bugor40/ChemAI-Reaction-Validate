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
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pytorch_metric_learning import losses



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
    
    
class ImprovedVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(ImprovedVAE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, input_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(latent_size)
        
    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_mean(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar



# In[ ]:


def model_fit(data_tensor, validation_data_tensor, target_train, target_val, cl = None, inter_score_train = None, inter_score_val = None, learning_rate = None, num_epochs = None, input_size = None, hidden_size = None, latent_size = None):
    
    if cl == 'VAE':
        model = VAE(input_size, hidden_size, latent_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if cl == 'ImprovedVAE':
        model = ImprovedVAE(input_size, hidden_size, latent_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    epoch_list = []
    auc_list = []
    auc_list_val = []

    for epoch in tqdm_notebook(range(num_epochs)):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data_tensor)
        BCE = F.mse_loss(recon_batch, data_tensor, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Обучение линейной регрессии и получение вероятностей для обучающей выборки
        lr = LogisticRegression()
        if inter_score_train is None:
            lr.fit(mu.detach().numpy(), target_train)
            pred_probs = lr.predict_proba(mu.detach().numpy())[:, -1]
            auc = roc_auc_score(target_train, pred_probs)
        else:
            train_np = np.hstack((mu.detach().numpy(), inter_score_train))
            scaler = StandardScaler()
            train_np = scaler.fit_transform(train_np)
            lr.fit(train_np, target_train)
            pred_probs = lr.predict_proba(train_np)[:, -1]
            auc = roc_auc_score(target_train, pred_probs)
        
        loss = (BCE+KLD) * torch.tensor(1-auc, requires_grad=True) # Используем обратное значение AUC как функцию потерь
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            val_recon_batch, val_mu, val_logvar = model(validation_data_tensor)
            
            # Получение вероятностей для валидационной выборки с помощью обученной линейной регрессии
            
            if inter_score_val is None:
                val_pred_probs = lr.predict_proba(val_mu.detach().numpy())[:, -1]
                val_auc = roc_auc_score(target_val, val_pred_probs)
            else:
                val_np = np.hstack((val_mu.detach().numpy(), inter_score_val))
                val_np = scaler.transform(val_np)
                val_pred_probs = lr.predict_proba(val_np)[:, -1]
                val_auc = roc_auc_score(target_val, val_pred_probs)
            
#             val_loss = -val_auc
        
        epoch_list.append(epoch+1)
        auc_list.append(auc)
        auc_list_val.append(val_auc)
        
        pd.DataFrame(np.array([epoch_list, auc_list, auc_list_val]).T,
                     columns=['epoch', 'auc', 'auc_val']) \
            .to_csv(f'epoch_{num_epochs}_hidden_size_{hidden_size}_latent_size_{latent_size}.tsv',
                    index=0, sep='\t')
        
        clear_output(wait=True)
        plt.plot(epoch_list, auc_list, label='train')
        plt.plot(epoch_list, auc_list_val, label='val')
#         plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'epoch_{num_epochs}_hidden_size_{hidden_size}_latent_size_{latent_size}.png')
        plt.show()

        print(f'Epoch [{epoch+1}/{num_epochs}], AUC: {auc:.4f}, Val AUC: {val_auc:.4f}')
        
    return model


# In[3]:


def model_predict(model, data_tensor, latent_size):
    with torch.no_grad():
        latent_representation = model.encode(data_tensor)[0].numpy()
        
    return pd.DataFrame(latent_representation)


# In[ ]:




