import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

class extract_tensor(nn.Module):
    def forward(self, x):
        if isinstance(x, tuple):
            tensor, _ = x
        else:
            tensor = x
        if tensor.dim() == 2:
            return tensor
        return tensor.mean(dim=1)

class Decoder(nn.Module):
    def __init__(self, seq_len, no_features, output_size):
        super().__init__()
        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size=no_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.fc2 = nn.Linear(3 * self.hidden_size, 5 * self.hidden_size)
        self.fc3 = nn.Linear(5 * self.hidden_size, 3 * self.hidden_size)
        self.fc4 = nn.Linear(3 * self.hidden_size, output_size)
        
    def forward(self, x, y):
        x = torch.cat((x, y.reshape(-1, 1)), dim=1)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.LSTM1(x)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        out = self.fc4(x)
        return out

class DySurvCR(nn.Module):
    def __init__(self, in_features, encoded_features, out_features, num_causes, seq_len=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_causes = num_causes
        
        self.lstm1 = nn.LSTM(in_features, in_features, batch_first=True)
        self.extract = extract_tensor()
        self.fc11 = nn.Linear(in_features, 3 * in_features)
        self.fc12 = nn.Linear(3 * in_features, 5 * in_features)
        self.fc13 = nn.Linear(5 * in_features, 3 * in_features)
        self.fc14 = nn.Linear(3 * in_features, encoded_features)
        self.fc24 = nn.Linear(3 * in_features, encoded_features)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.output_all = True

        self.surv_net = nn.Sequential(
            nn.Linear(encoded_features, 3 * in_features), nn.ReLU(), 
            nn.Linear(3 * in_features, 5 * in_features), nn.ReLU(), 
            nn.Linear(5 * in_features, 3 * in_features), nn.ReLU(), 
            nn.Linear(3 * in_features, out_features * num_causes),
        )
        
        self.decoder2 = Decoder(seq_len, encoded_features + 1, in_features)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def encoder(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x, _ = self.lstm1(x)
        x = self.extract(x)

        x = self.relu(self.fc11(x)) 
        x = self.relu(self.fc12(x))
        x = self.relu(self.fc13(x))
        mu_z = self.fc14(x)
        logvar_z = self.fc24(x)

        return mu_z, logvar_z
    
    def forward(self, x):
        mu, logvar = self.encoder(x.float())
        z = self.reparameterize(mu, logvar)
        phi = self.surv_net(z)
        
        # Reshape phi to (batch, num_causes, num_time_bins)
        phi = phi.view(-1, self.num_causes, self.out_features)

        if self.output_all:
            y_dummy = torch.zeros(x.shape[0], 1, device=x.device)
            decoded = self.decoder2(z, y_dummy)
            return decoded, phi, mu, logvar
        
        return phi

    def predict(self, x):
        mu, _ = self.encoder(x.float())
        phi = self.surv_net(mu)
        return phi.view(-1, self.num_causes, self.out_features)
