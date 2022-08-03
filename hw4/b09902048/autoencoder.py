from data_transformer import DataTransformer
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

class AutoencoderModel(nn.Module):
    """Add more of your code here if you want to"""
    def __init__(self, latent_space_dim, architecture_callback):
        super().__init__()
        architecture_callback(self, latent_space_dim)

class Autoencoder(DataTransformer):
    """Add more of your code here if you want to"""
    
    def __init__(self, args, architecture_callback):
        self.args = args
        self.model = AutoencoderModel(args.latent_space_dim, architecture_callback).to(args.device)

    def fit(self, X):
        model = self.model
        epochs = self.args.num_epochs
        dataloader = DataLoader(X, batch_size=self.args.batch_size,shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        best_loss = np.inf
       
        avg_mse = []
        for i in range(100):
            total_loss = list()
            for data in dataloader:
                x = model.encoder(data)
                x = model.decoder(x)
                loss = criterion(x, data)
                total_loss.append(loss.item())
                #print(f'{i}', loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #x = x.detach()

            mean = np.mean(total_loss)
            avg_mse.append(mean)
            if mean < best_loss:
                best_loss = mean
                self.model = model
            #plt.plot(avg_mse)
            #plt.title("Average squared error")
            #plt.savefig('autoencoder.png')
        return
        raise NotImplementedError
    
    def transform(self, X):
        x = self.model.encoder(X)
        return x.detach()
        raise NotImplementedError
    
    def reconstruct(self, X_transformed):
        return self.model.decoder(X_transformed).detach()
        raise NotImplementedError

class DenoisingAutoencoder(Autoencoder):
    """Add more of your code here if you want to"""
    def __init__(self, args, architecture_callback):
        super().__init__(args, architecture_callback)
    def fit_transform(self, X):
        noise_factor = self.args.noise_factor
        X_noise = np.random.normal(X, noise_factor)
        X_noise = np.clip(X_noise, 0., 1.)
        X_noise = torch.tensor(X_noise).float()

        model = self.model
        epochs = self.args.num_epochs
        dataloader = DataLoader(X, batch_size=self.args.batch_size,shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        best_loss = np.inf
       
        avg_mse = []
        for i in range(100):
            total_loss = list()
            for data in dataloader:
                x_noise = np.random.normal(data, noise_factor)
                x_noise = np.clip(x_noise, 0., 1.)
                x_noise = torch.tensor(x_noise).float()
                x = model.encoder(x_noise)
                x = model.decoder(x)
                loss = criterion(x, data)
                total_loss.append(loss.item())
                #print(f'{i}', loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #x = x.detach()

            mean = np.mean(total_loss)
            avg_mse.append(mean)
            if mean < best_loss:
                best_loss = mean
                self.model = model
            #plt.plot(avg_mse)
            #plt.title("Average squared error")
            #plt.savefig('denoising autoencoder.png')

        return self.model.encoder(X_noise).detach()
        return super().fit_transform(x_noise)
    def reconstruct(self, X_transformed):
        return super().reconstruct(X_transformed)