import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvg1D(nn.Module):
    def __init__(self, dim):
        super(GlobalAvg1D, self).__init__()
        
        self.dim = dim
    
    def __call__(self, tensor):
        if len(tensor.shape) != 4:
            raise Exception('tensor must be rank of 4')
            
        return tensor.mean(self.dim, keepdim=True)

class GlobalAvg2D(nn.Module):
    def __call__(self, tensor):
        if len(tensor.shape) != 4:
            raise Exception('tensor must be rank of 4')
            
        return tensor.mean([2, 3], keepdim=True)
    
def straight_through_estimator(logits):
    argmax = torch.eq(logits, logits.max(-1, keepdim=True).values).to(logits.dtype)
    return (argmax - logits).detach() + logits

def gumbel_softmax(logits, temperature=1.0, eps=1e-20):
    u = torch.rand(logits.size(), dtype=logits.dtype, device=logits.device)
    g = -torch.log(-torch.log(u + eps) + eps)
    return F.softmax((logits + g) / temperature, dim=-1)

class Autoencoder(nn.Module):
    def __init__(self, h_dim=512, sigmoid=False, depths=(64, 64, 128, 128, 256, 256, 512)):
        super(Autoencoder, self).__init__()
        
        self.h_dim = h_dim
        self.sigmoid = sigmoid
        self.depths = depths
        
        # (1, 128, 251)
        self.encode = nn.Sequential(
            nn.Conv2d(1, depths[0], (3, 4), padding=(1, 4), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[0]),
            
            nn.Conv2d(depths[0], depths[1], 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[1]),
            
            nn.Conv2d(depths[1], depths[2], 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[2]),
            
            nn.Conv2d(depths[2], depths[3], 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[3]),
            
            nn.Conv2d(depths[3], depths[4], 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[4]),
            
            nn.Conv2d(depths[4], depths[5], 3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[5]),
            
            nn.Conv2d(depths[5], depths[6], (2, 4), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[6]),
            
            nn.Conv2d(depths[6], h_dim, 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(h_dim)
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(h_dim, depths[6], 1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(depths[6]),
            
            nn.ConvTranspose2d(depths[6], depths[5], (4, 6), padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[5]),
            
            nn.ConvTranspose2d(depths[5], depths[4], 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[4]),
            
            nn.ConvTranspose2d(depths[4], depths[3], 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[3]),
            
            nn.ConvTranspose2d(depths[3], depths[2], 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[2]),
            
            nn.ConvTranspose2d(depths[2], depths[1], 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[1]),
            
            nn.ConvTranspose2d(depths[1], depths[0], 4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(depths[0]),
            
            nn.ConvTranspose2d(depths[0], 1, (4, 5), padding=(1, 4), stride=2)
        )
        
    def forward(self, x):
        h = self.encode(x)
        
        if self.sigmoid:
            return torch.sigmoid(self.decode(h))
        else:
            return self.decode(h)

class VAE(Autoencoder):
    def __init__(self, h_dim=512, sigmoid=False, depths=(64, 64, 128, 128, 256, 256, 512)):
        super(VAE, self).__init__(h_dim=h_dim, sigmoid=sigmoid, depths=depths)
        
        self.fc_mu = nn.Linear(h_dim, h_dim)
        self.fc_log_var = nn.Linear(h_dim, h_dim)
        
    def _reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def _latent_layer(self, h, sample=True, temperature=1.0):
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        
        if sample:
            z = self._reparameterize(mu, log_var)
        else:
            z = mu
        return z, mu, log_var
        
    def forward(self, x, sample=True, temperature=1.0):
        h = self.encode(x)
        h = h.view(h.size(0), -1)
        
        z, mu, log_var = self._latent_layer(h, sample=sample, temperature=temperature)
        z = z[..., None, None]
        
        outputs = self.decode(z)
        
        if self.sigmoid:
            return torch.sigmoid(outputs), mu, log_var
        else:
            return outputs, mu, log_var

class CVAE(VAE):
    def __init__(self, n_classes, h_dim=512, sigmoid=False, depths=(64, 64, 128, 128, 256, 256, 512)):
        super(CVAE, self).__init__(h_dim=h_dim, sigmoid=sigmoid, depths=depths)
        
        self.n_classes = n_classes

        self.fc_cond = nn.Linear(h_dim, n_classes)
        self.fc_merge = nn.Linear(h_dim + n_classes, h_dim)
        
    def _latent_layer(self, h, sample=True, temperature=1.0):
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        c_logits = self.fc_cond(h)
        
        if sample:
            z = self._reparameterize(mu, log_var)
            c_dist = gumbel_softmax(c_logits, temperature=temperature)
        else:
            z = mu
            c_dist = F.softmax(c_logits, dim=-1)            

        c = straight_through_estimator(c_dist)
        
        # merge
        y = F.relu(self.fc_merge(torch.cat([z, c], dim=1)))
        return z, y, c_logits, mu, log_var

    def forward(self, x, sample=True, temperature=1.0):
        h = self.encode(x)
        h = h.view(h.size(0), -1)
        
        z, y, c_logits, mu, log_var = self._latent_layer(h, sample=sample, temperature=temperature)
        y = y[..., None, None]
        
        outputs = self.decode(y)
        
        if self.sigmoid:
            return torch.sigmoid(outputs), c_logits, mu, log_var
        else:
            return outputs, c_logits, mu, log_var