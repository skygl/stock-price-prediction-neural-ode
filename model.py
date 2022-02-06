import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.seq = nn.Sequential(nn.GRUCell(hidden_dim, hidden_dim))

    def forward(self, t, x):
        out = self.seq(x)
        return out


class ODEBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEBlock, self).__init__()
        self.hidden_dim = hidden_dim

        self.ode_func = ODEFunc(self.hidden_dim)

    def forward(self, z0, t):
        return odeint(self.ode_func, z0, t)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)

    def forward(self, x):
        # x: [bs, n, input_dim]
        # out: [bs, n, hidden_dim]
        # last_h: [bs, hidden_dim]
        out, _ = self.rnn(x)
        last_h = out[:, -1]
        return last_h


class PredictModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PredictModel, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)

        self.ode_block = ODEBlock(hidden_dim)

        self.decoder = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # x: [bs, n, input_dim]
        # h: [bs, hidden_dim]
        h = self.encoder(x)

        # mu, logvar: [bs, hidden_dim]
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)

        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, n, m):
        # x: [bs, n, input_dim]
        # mu, logvar: [bs, hidden_dim]
        # z0: [bs, hidden_dim]
        mu, logvar = self.encode(x)
        z0 = self.reparameterize(mu, logvar)

        # t: [n+m,]
        t = torch.linspace(0, n + m - 1, n + m)

        # ode_out: [n+m, bs, hidden_dim]
        # ode_out: [bs, n+m, hidden_dim]
        ode_out = self.ode_block(z0, t)
        ode_out = ode_out.permute(1, 0, 2)

        # out: [bs, n+m, input_dim]
        out = self.decoder(ode_out)

        return out, mu, logvar
