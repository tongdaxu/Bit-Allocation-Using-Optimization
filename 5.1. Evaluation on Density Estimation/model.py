import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from compressai.ops import LowerBound

SEED=3470

class BinarizeMnist(object):
    def __init__(self):
        pass
    def __call__(self,image):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        return torch.bernoulli(image)

class ScaleMnist(object):
    def __init__(self):
        pass
    def __call__(self,image):
        return 255 * image

class VAE_SingleLayer(nn.Module):
    def __init__(self, h_dim: int = 200, z_dim: int = 50, im_shape: tuple = (1,28,28)):
        super(VAE_SingleLayer, self).__init__()
        self.c, self.h, self.w  = im_shape
        self.encoder = nn.Sequential(
            nn.Linear(self.c * self.h *self.w, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
        )
        self.encoder_mu = nn.Linear(h_dim, z_dim)
        self.encoder_sigma = nn.Sequential(
            nn.Linear(h_dim, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 2 * self.c * self.h *self.w)
        )
        self.eps=1e-4

    def _encode(self, x):
        x = x.reshape(-1, self.c * self.h * self.w)
        h_enc = self.encoder(x)
        return self.encoder_mu(h_enc), torch.exp(self.encoder_sigma(h_enc))

    def _decode(self, z):
        h_dec = self.decoder(z)
        return h_dec.reshape(-1, self.c, self.h, self.w, 2)

    def forward(self, x, mode="sgvb2"):
        z_mu, z_sigma = self._encode(x)
        dist_q_z_con_x = torch.distributions.normal.Normal(z_mu, z_sigma+self.eps)
        z_hat = dist_q_z_con_x.rsample()
        log_q_z_con_x = dist_q_z_con_x.log_prob(z_hat)
        dist_p_z = torch.distributions.normal.Normal(0,1)
        log_p_z = dist_p_z.log_prob(z_hat)
        x_logits = self._decode(z_hat)
        dist_x_con_z = torch.distributions.categorical.Categorical(logits=x_logits)
        log_p_x_con_z = dist_x_con_z.log_prob(x.long())
        if mode == "sgvb1":
            elbo = torch.sum(log_p_x_con_z, dim=(1,2,3)) + torch.sum(log_p_z, dim=1) - torch.sum(log_q_z_con_x, dim=1)
        elif mode == "sgvb2":
            elbo = torch.sum(log_p_x_con_z, dim=(1,2,3)) + 0.5 * torch.sum((1 + 2 * torch.log(z_sigma) - z_mu ** 2 - z_sigma ** 2), dim=1)
        else:
            raise NotImplementedError
        return elbo

    def nll_iwae(self, x, k):
        b, _, _, _ = x.shape
        elbos = torch.zeros([b,k])
        for i in range(k):
            elbos[:, i] = self.forward(x, mode="sgvb1")
        weights = F.softmax(elbos, dim=1)
        iwelbo = torch.sum(elbos * weights, dim=1)
        return iwelbo


class VAE_TwoLayer_Alt(nn.Module):
    def __init__(self, h_dim: int = 200, z1_dim: int = 100, im_shape: tuple = (1,28,28)):
        super(VAE_TwoLayer_Alt, self).__init__()
        self.c, self.h, self.w  = im_shape
        self.encoder_1 = nn.Sequential(
            nn.Linear(self.c * self.h *self.w, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
        )
        self.encoder_mu_1 = nn.Linear(h_dim, z1_dim)
        self.encoder_sigma_1 = nn.Linear(h_dim, z1_dim)
        self.decoder_1 = nn.Sequential(
            nn.Linear(z1_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, self.c * self.h *self.w)
        )
        self.eps=1e-5
        self.eps2=1e-5
        self.mcs=5000
        self.mom=0.999
        # self.mom=0.0
        self.lb=LowerBound(self.eps)

    def setseed(self, seed):
        if seed!=-1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

    def _encode_1(self, x):
        x = x.reshape(-1, self.c * self.h * self.w)
        h1_enc = self.encoder_1(x)
        return self.encoder_mu_1(h1_enc), torch.exp(self.encoder_sigma_1(h1_enc))

    def _decode_1(self, z1_hat):
        h1_dec = self.decoder_1(z1_hat)
        return h1_dec.reshape(-1, self.c, self.h, self.w)

    def forward(self, x, mode="sgvb2", return_param=False, seed=-1):
        dist_p_z = torch.distributions.normal.Normal(0,1)
        z1_mu, z1_sigma = self._encode_1(x)
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, z1_sigma)
        z1_hat = dist_q_z1_con_x.rsample()
        log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
        x1_mu = self._decode_1(z1_hat)
        res = x - x1_mu
        z2_mu, z2_sigma = self._encode_1(x - res)
        dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, z2_sigma)
        z2_hat = dist_q_z2_con_z1.rsample()
        log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)
        x2_mu = self._decode_1(z2_hat)
        dist_x_con_z1z2 = torch.distributions.normal.Normal(x1_mu + x2_mu, 1.0)
        log_p_x_conz1z2 = dist_x_con_z1z2.log_prob(x)
        log_p_z1 = dist_p_z.log_prob(z1_hat)
        log_p_z2 = dist_p_z.log_prob(z2_hat)
        if mode == "sgvb1":
            elbo = torch.sum(log_p_x_conz1z2, dim=(1,2,3)) +\
                   torch.sum(log_p_z1, dim=1) + torch.sum(log_p_z2, dim=1) -\
                   torch.sum(log_q_z1_con_x, dim=1) -\
                   torch.sum(log_q_z2_con_z1, dim=1)
        elif mode == "sgvb2":
            elbo = torch.sum(log_p_x_conz1z2, dim=(1,2,3)) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - z1_mu ** 2 - z1_sigma ** 2), dim=1)
        else:
            raise NotImplementedError

        if return_param:
            return elbo, z1_mu, z1_sigma, z2_mu, z2_sigma
        else:
            return elbo

class VAE_TwoLayer(nn.Module):
    def __init__(self, h_dim: int = 200, z1_dim: int = 100, z2_dim: int = 50, im_shape: tuple = (1,28,28)):
        super(VAE_TwoLayer, self).__init__()
        self.c, self.h, self.w  = im_shape
        self.encoder_1 = nn.Sequential(
            nn.Linear(self.c * self.h *self.w, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
        )
        self.encoder_mu_1 = nn.Linear(h_dim, z1_dim)
        self.encoder_sigma_1 = nn.Linear(h_dim, z1_dim)
        self.encoder_2 = nn.Sequential(
            nn.Linear(z1_dim, h_dim // 2),
            nn.Tanh(),
            nn.Linear(h_dim // 2, h_dim // 2),
            nn.Tanh(),
        )
        self.encoder_mu_2 = nn.Linear(h_dim // 2, z2_dim)
        self.encoder_sigma_2 = nn.Linear(h_dim // 2, z2_dim)
        self.decoder_2 = nn.Sequential(
            nn.Linear(z2_dim, h_dim // 2),
            nn.Tanh(),
            nn.Linear(h_dim // 2, h_dim // 2),
            nn.Tanh(),
        )
        self.decoder_mu_2 = nn.Linear(h_dim // 2, z1_dim)
        self.decoder_sigma_2 = nn.Linear(h_dim // 2, z1_dim)
        self.decoder_1 = nn.Sequential(
            nn.Linear(z1_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 2 * self.c * self.h *self.w)
        )
        self.eps=1e-5
        self.eps2=1e-5
        self.mcs=5000
        self.mom=0.999
        # self.mom=0.0
        self.lb=LowerBound(self.eps)

    def setseed(self, seed):
        if seed!=-1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

    def _encode_1(self, x):
        x = x.reshape(-1, self.c * self.h * self.w)
        h1_enc = self.encoder_1(x)
        return self.encoder_mu_1(h1_enc), torch.exp(self.encoder_sigma_1(h1_enc))

    def _encode_2(self, z1_hat):
        h2_enc = self.encoder_2(z1_hat)
        return self.encoder_mu_2(h2_enc), torch.exp(self.encoder_sigma_2(h2_enc))

    def _decode_2(self, z2_hat):
        h2_dec = self.decoder_2(z2_hat)
        return self.decoder_mu_2(h2_dec), torch.exp(self.decoder_sigma_2(h2_dec))

    def _decode_1(self, z1_hat):
        h1_dec = self.decoder_1(z1_hat)
        return h1_dec.reshape(-1, self.c, self.h, self.w, 2)

    def forward(self, x, mode="sgvb2", return_param=False, seed=-1):
        z1_mu, z1_sigma = self._encode_1(x)
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, z1_sigma)
        self.setseed(seed)
        z1_hat = dist_q_z1_con_x.rsample()
        log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
        z2_mu, z2_sigma = self._encode_2(z1_hat)
        dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, z2_sigma)
        self.setseed(seed)
        z2_hat = dist_q_z2_con_z1.rsample()
        log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)
        dist_p_z2 = torch.distributions.normal.Normal(0,1)
        log_p_z2 = dist_p_z2.log_prob(z2_hat)
        p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
        dist_p_z1_con_z2 = torch.distributions.normal.Normal(p_z1_mu, p_z1_sigma)
        log_p_z2_con_z1 = dist_p_z1_con_z2.log_prob(z1_hat)
        x_logits = self._decode_1(z1_hat)
        dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
        log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
        if mode == "sgvb1":
            elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                   torch.sum(log_p_z2_con_z1, dim=1) + torch.sum(log_p_z2, dim=1) -\
                   torch.sum(log_q_z1_con_x, dim=1) -\
                   torch.sum(log_q_z2_con_z1, dim=1)
        elif mode == "sgvb2":
            elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - 2 * torch.log(p_z1_sigma) - (z1_sigma**2 + (z1_mu - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
        else:
            raise NotImplementedError

        if return_param:
            return elbo, z1_mu, z1_sigma, z2_mu, z2_sigma
        else:
            return elbo

    def elbo_from_param(self, x, z1_mu, z1_sigma, z2_mu, z2_sigma, mode="sgvb2"):
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, z1_sigma)
        z1_hat = dist_q_z1_con_x.rsample()
        log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
        dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, z2_sigma)
        z2_hat = dist_q_z2_con_z1.rsample()
        log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)
        dist_p_z2 = torch.distributions.normal.Normal(0,1)
        log_p_z2 = dist_p_z2.log_prob(z2_hat)
        p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
        dist_p_z1_con_z2 = torch.distributions.normal.Normal(p_z1_mu, p_z1_sigma)
        log_p_z2_con_z1 = dist_p_z1_con_z2.log_prob(z1_hat)
        x_logits = self._decode_1(z1_hat)
        dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
        log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
        if mode == "sgvb1":
            elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                   torch.sum(log_p_z2_con_z1, dim=1) + torch.sum(log_p_z2, dim=1) -\
                   torch.sum(log_q_z1_con_x, dim=1) -\
                   torch.sum(log_q_z2_con_z1, dim=1)
        elif mode == "sgvb2":
            elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
                   0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - 2 * torch.log(p_z1_sigma) - (z1_sigma**2 + (z1_mu - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
        else:
            raise NotImplementedError
        return elbo

    def nll_iwae(self, x, k):
        b, _, _, _ = x.shape
        elbos = torch.zeros([b,k])
        for i in range(k):
            elbos[:, i] = self.forward(x, mode="sgvb1")
        weights = F.softmax(elbos, dim=1)
        iwelbo = torch.sum(elbos * weights, dim=1)
        return iwelbo

    def nll_iwae_from_param(self, x, z1_mu, z1_sigma, z2_mu, z2_sigma, k):
        b, _, _, _ = x.shape
        assert(b == 1)
        rb = 500
        assert(k % rb == 0)
        x = torch.repeat_interleave(x, rb, dim=0)
        z1_mu = torch.repeat_interleave(z1_mu, rb, dim=0)
        z1_sigma = torch.repeat_interleave(z1_sigma, rb, dim=0)
        z2_mu = torch.repeat_interleave(z2_mu, rb, dim=0)
        z2_sigma = torch.repeat_interleave(z2_sigma, rb, dim=0)
        elbos = torch.zeros([b,k])
        for i in range(k // rb):
            elbos[:, i*rb:(i+1)*rb] = self.elbo_from_param(x, z1_mu, z1_sigma, z2_mu, z2_sigma, mode="sgvb1")
        weights = F.softmax(elbos, dim=1)
        iwelbo = torch.sum(elbos * weights, dim=1)
        # return iwelbo
        return torch.mean(elbos, dim=1)

    def savi_naive(self, x, iter, lr, mode="sgvb2", seed=-1):
        b, c, h, w = x.shape
        assert (b==1)
        for param in self.parameters():
            param.requires_grad_(False)
        with torch.no_grad():
            elbo_favi, z1_mu, z1_sigma, z2_mu, z2_sigma = self.forward(x, mode="sgvb2", return_param=True, seed=seed)
        params = [param.detach().clone().requires_grad_(True) for param in [z1_mu, z1_sigma, z2_mu, z2_sigma]]
        vs = [0 for param in params]
        for i in range(iter):
            dist_q_z1_con_x = torch.distributions.normal.Normal(params[0], self.lb(params[1]))
            self.setseed(seed + i)
            z1_hat = dist_q_z1_con_x.rsample()
            log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
            dist_q_z2_con_z1 = torch.distributions.normal.Normal(params[2], self.lb(params[3]))
            self.setseed(seed + i)
            z2_hat = dist_q_z2_con_z1.rsample()
            log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)     
            dist_p_z2 = torch.distributions.normal.Normal(0,1)
            log_p_z2 = dist_p_z2.log_prob(z2_hat)
            p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
            dist_p_z1_con_z2 = torch.distributions.normal.Normal(p_z1_mu, p_z1_sigma)
            log_p_z2_con_z1 = dist_p_z1_con_z2.log_prob(z1_hat)
            x_logits = self._decode_1(z1_hat)
            dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
            log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
            if mode == "sgvb1":
                elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                    torch.sum(log_p_z2_con_z1, dim=1) + torch.sum(log_p_z2, dim=1) -\
                    torch.sum(log_q_z1_con_x, dim=1) -\
                    torch.sum(log_q_z2_con_z1, dim=1)
            elif mode == "sgvb2":
                elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                    0.5 * torch.sum((1 + 2 * torch.log(params[3]) - params[2] ** 2 - params[3] ** 2), dim=1) +\
                    0.5 * torch.sum((1 + 2 * torch.log(params[1]) - 2 * torch.log(p_z1_sigma) - (params[1]**2 + (params[0] - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
            else:
                raise NotImplementedError
            loss = - torch.mean(elbo / (c * h * w), dim=0)
            for param in params:
                if param.grad is None:
                    continue
                param.grad.zero_()
            loss.backward()
            for ip in range(len(params)):
                with torch.no_grad():
                    vs[ip] = self.mom * vs[ip] - params[ip].grad
                    params[ip] = params[ip] + lr * vs[ip]
                params[ip].requires_grad_(True)            
        return elbo_favi, elbo

    def savi_approx(self, x, iter, lr, mode="sgvb2", seed=-1):
        b, c, h, w = x.shape
        assert (b==1)
        for param in self.parameters():
            param.requires_grad_(False)
        with torch.no_grad():
            elbo_favi, z1_mu, z1_sigma, z2_mu, z2_sigma = self.forward(x, mode="sgvb2", return_param=True, seed=seed)
        params_1 = torch.cat([z1_mu.detach().clone(), z1_sigma.detach().clone()],dim=0).requires_grad_(True)
        vs_1 = 0
        for i in range(iter):
            z1_mu, z1_sigma = params_1.chunk(2, dim=0)
            dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, self.lb(z1_sigma))
            self.setseed(seed + i)
            z1_hat = dist_q_z1_con_x.rsample()
            log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
            z2_mu, z2_sigma = self._encode_2(z1_hat)
            dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, self.lb(z2_sigma))
            self.setseed(seed + i)
            z2_hat = dist_q_z2_con_z1.rsample()
            log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)
            dist_p_z2 = torch.distributions.normal.Normal(0,1)
            log_p_z2 = dist_p_z2.log_prob(z2_hat)
            p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
            dist_p_z1_con_z2 = torch.distributions.normal.Normal(p_z1_mu, p_z1_sigma)
            log_p_z2_con_z1 = dist_p_z1_con_z2.log_prob(z1_hat)
            x_logits = self._decode_1(z1_hat)
            dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
            log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
            if mode == "sgvb1":
                elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                    torch.sum(log_p_z2_con_z1, dim=1) + torch.sum(log_p_z2, dim=1) -\
                    torch.sum(log_q_z1_con_x, dim=1) -\
                    torch.sum(log_q_z2_con_z1, dim=1)
            elif mode == "sgvb2":
                elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                    0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
                    0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - 2 * torch.log(p_z1_sigma) - (z1_sigma**2 + (z1_mu - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
            else:
                raise NotImplementedError
            loss = - torch.mean(elbo / (c * h * w), dim=0)
            if params_1.grad is not None:
                params_1.grad.zero_()
            loss.backward()
            with torch.no_grad():
                vs_1 = self.mom * vs_1 - params_1.grad
                params_1 = params_1 + lr * vs_1
            params_1.requires_grad_(True)     
        params_1.requires_grad_(False)       
        z1_mu, z1_sigma = params_1.chunk(2, dim=0)
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, self.lb(z1_sigma))
        self.setseed(seed + iter - 1)
        z1_hat = dist_q_z1_con_x.rsample()
        log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
        params_2 = torch.cat([z2_mu.detach().clone(), z2_sigma.detach().clone()], dim=0).requires_grad_(True)
        vs_2 = 0
        for i in range(iter):
            z2_mu, z2_sigma = params_2.chunk(2, dim=0)            
            dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, self.lb(z2_sigma))
            self.setseed(seed + i)
            z2_hat = dist_q_z2_con_z1.rsample()
            log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)
            dist_p_z2 = torch.distributions.normal.Normal(0,1)
            log_p_z2 = dist_p_z2.log_prob(z2_hat)
            p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
            dist_p_z1_con_z2 = torch.distributions.normal.Normal(p_z1_mu, p_z1_sigma)
            log_p_z2_con_z1 = dist_p_z1_con_z2.log_prob(z1_hat)
            x_logits = self._decode_1(z1_hat)
            dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
            log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
            if mode == "sgvb1":
                elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                    torch.sum(log_p_z2_con_z1, dim=1) + torch.sum(log_p_z2, dim=1) -\
                    torch.sum(log_q_z1_con_x, dim=1) -\
                    torch.sum(log_q_z2_con_z1, dim=1)
            elif mode == "sgvb2":
                elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                    0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
                    0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - 2 * torch.log(p_z1_sigma) - (z1_sigma**2 + (z1_mu - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
            else:
                raise NotImplementedError
            loss = - torch.mean(elbo / (c * h * w), dim=0)
            if params_2.grad is not None:
                params_2.grad.zero_()
            loss.backward()
            with torch.no_grad():
                vs_2 = self.mom * vs_2 - params_2.grad
                params_2 = params_2 + lr * vs_2
            params_2.requires_grad_(True)     
        return elbo_favi, elbo

    def get_grad_params_1(self, x, params_1, params_2, r=0, v=0, seed=-1):
        b, c, h, w = x.shape
        z1_mu, z1_sigma = params_1.chunk(2, dim=0)
        if r==0:
            z2_mu, z2_sigma = params_2.chunk(2, dim=0)
        else:
            params_2_hat = params_2 + v * r
            z2_mu, z2_sigma = params_2_hat.chunk(2, dim=0)
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, self.lb(z1_sigma))
        self.setseed(seed)
        z1_hat = dist_q_z1_con_x.rsample()
        dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, self.lb(z2_sigma))
        self.setseed(seed)
        z2_hat = dist_q_z2_con_z1.rsample()
        ### get p(z1|z2) p(x|z1)
        p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
        x_logits = self._decode_1(z1_hat)
        dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
        log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
        ### eval elbo
        elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
            0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
            0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - 2 * torch.log(p_z1_sigma) - (z1_sigma**2 + (z1_mu - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
        loss = - torch.mean(elbo / (c * h * w), dim=0)
        if params_1.grad is not None:
            params_1.grad.zero_()
        if params_2.grad is not None:
            params_2.grad.zero_()
        loss.backward()
        with torch.no_grad():
            grad = params_1.grad
        return grad

    def get_grad_params_2(self, x, params_1, params_2, r=0, v=0, seed=-1):
        b, c, h, w = x.shape
        z1_mu, z1_sigma = params_1.chunk(2, dim=0)
        if r==0:
            z2_mu, z2_sigma = params_2.chunk(2, dim=0)
        else:
            params_2_hat = params_2 + v * r
            z2_mu, z2_sigma = params_2_hat.chunk(2, dim=0)
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, self.lb(z1_sigma))
        self.setseed(seed)
        z1_hat = dist_q_z1_con_x.rsample()
        dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, self.lb(z2_sigma))
        self.setseed(seed)
        z2_hat = dist_q_z2_con_z1.rsample()
        ### get p(z1|z2) p(x|z1)
        p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
        x_logits = self._decode_1(z1_hat)
        dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
        log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
        ### eval elbo
        elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
            0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
            0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - 2 * torch.log(p_z1_sigma) - (z1_sigma**2 + (z1_mu - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
        loss = - torch.mean(elbo / (c * h * w), dim=0)
        if params_1.grad is not None:
            params_1.grad.zero_()
        if params_2.grad is not None:
            params_2.grad.zero_()
        loss.backward()
        with torch.no_grad():
            grad = params_2.grad
        return grad

    def hessian_bb(self, x, params_1_i, params_2_j, b_j_plus_1, r, seed=-1):
        params_1_i = params_1_i.detach().clone()
        params_2_j = params_2_j.detach().clone().requires_grad_(True)
        # hessian bb = (1/r)*(rhs - lhs)
        grad_lhs = self.get_grad_params_2(x, params_1_i, params_2_j, r=0, v=0, seed=seed)
        grad_rhs = self.get_grad_params_2(x, params_1_i, params_2_j, r=r, v=b_j_plus_1, seed=seed)
        hessian_bb = (grad_rhs - grad_lhs)/r
        return hessian_bb


    def hessian_ab(self, x, params_1_i, params_2_j, b_j_plus_1, r, seed=-1):
        params_1_i = params_1_i.detach().clone().requires_grad_(True)
        params_2_j = params_2_j.detach().clone()
        # hessian bb = (1/r)*(rhs - lhs)
        grad_lhs = self.get_grad_params_1(x, params_1_i, params_2_j, r=0, v=0, seed=seed)
        grad_rhs = self.get_grad_params_1(x, params_1_i, params_2_j, r=r, v=b_j_plus_1, seed=seed)
        hessian_ab = (grad_rhs - grad_lhs)/r
        return hessian_ab


    def savi_accurate(self, x, iter, lr, mode="sgvb2", seed=-1):
        b, c, h, w = x.shape
        assert (b==1)
        for param in self.parameters():
            param.requires_grad_(False)
        with torch.no_grad():
            elbo_favi, z1_mu, z1_sigma, _, _ = self.forward(x, mode="sgvb2", return_param=True, seed=seed)
        params_1 = torch.cat([z1_mu.detach().clone(), z1_sigma.detach().clone()],dim=0).requires_grad_(True)
        vs_1 = 0
        for i in range(iter):
            z1_mu, z1_sigma = params_1.chunk(2, dim=0)
            dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, self.lb(z1_sigma))
            self.setseed(seed + i)
            z1_hat = dist_q_z1_con_x.rsample()
            log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
            z2_mu, z2_sigma = self._encode_2(z1_hat)
            params_2 = torch.cat([z2_mu, z2_sigma], dim=0)
            # get the subpart of the computational graph, splitting it from here!
            params_2_j = params_2.detach().clone().requires_grad_(True)
            params_2_hist = []
            params_2_hist.append(params_2_j.detach().clone().requires_grad_(False))
            for j in range(iter+1):
                z2_mu_j, z2_sigma_j = params_2_j.chunk(2, dim=0)        
                dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu_j, self.lb(z2_sigma_j))
                self.setseed(seed + j)
                z2_hat = dist_q_z2_con_z1.rsample()
                log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)
                dist_p_z2 = torch.distributions.normal.Normal(0,1)
                log_p_z2 = dist_p_z2.log_prob(z2_hat)
                p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
                dist_p_z1_con_z2 = torch.distributions.normal.Normal(p_z1_mu, p_z1_sigma)
                log_p_z2_con_z1 = dist_p_z1_con_z2.log_prob(z1_hat)
                x_logits = self._decode_1(z1_hat)
                dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
                log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
                if mode == "sgvb1":
                    elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                        torch.sum(log_p_z2_con_z1, dim=1) + torch.sum(log_p_z2, dim=1) -\
                        torch.sum(log_q_z1_con_x, dim=1) -\
                        torch.sum(log_q_z2_con_z1, dim=1)
                elif mode == "sgvb2":
                    elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                        0.5 * torch.sum((1 + 2 * torch.log(z2_sigma_j) - z2_mu_j ** 2 - z2_sigma_j ** 2), dim=1) +\
                        0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - 2 * torch.log(p_z1_sigma) - (z1_sigma**2 + (z1_mu - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
                else:
                    raise NotImplementedError
                loss = - torch.mean(elbo / (c * h * w), dim=0)
                if params_1.grad is not None:
                    params_1.grad.zero_()
                if params_2_j.grad is not None:
                    params_2_j.grad.zero_()
                loss.backward(retain_graph=True)
                if j == iter:
                    d_a = params_1.grad
                    d_b = params_2_j.grad
                    if d_b is None:
                        assert 0
                else:
                    if params_2_j.grad is None:
                        assert 0
                    with torch.no_grad():
                        params_2_j = params_2_j - lr * params_2_j.grad
                    params_2_j.requires_grad_(True)
                    params_2_hist.append(params_2_j.detach().clone())          
            for j in reversed(range(iter)):
                d_a = d_a + lr * self.hessian_ab(x, params_1, params_2_hist[j], d_b, self.eps2, seed + j)
                d_b = d_b + lr * self.hessian_bb(x, params_1, params_2_hist[j], d_b, self.eps2, seed + j)
            # after a while ...
            if params_1.grad is not None:
                params_1.grad.zero_()
            params_2.backward(d_b)
            d_a += params_1.grad
            params_1.grad = d_a
            with torch.no_grad():
                vs_1 = self.mom * vs_1 - params_1.grad
                params_1 = params_1 + lr * vs_1
            params_1.requires_grad_(True)
        # after those updates, we fix params_1 now, and update params_2 for a final round
        params_1.requires_grad_(False)
        z1_mu, z1_sigma = params_1.chunk(2, dim=0)
        dist_q_z1_con_x = torch.distributions.normal.Normal(z1_mu, self.lb(z1_sigma))
        self.setseed(seed + iter - 1)
        z1_hat = dist_q_z1_con_x.rsample()
        log_q_z1_con_x = dist_q_z1_con_x.log_prob(z1_hat)
        z2_mu, z2_sigma = self._encode_2(z1_hat)
        params_2 = torch.cat([z2_mu.detach().clone(), z2_sigma.detach().clone()], dim=0).requires_grad_(True)
        vs_2 = 0
        for i in range(iter):
            z2_mu, z2_sigma = params_2.chunk(2, dim=0)
            dist_q_z2_con_z1 = torch.distributions.normal.Normal(z2_mu, self.lb(z2_sigma))
            self.setseed(seed + i)
            z2_hat = dist_q_z2_con_z1.rsample()
            log_q_z2_con_z1 = dist_q_z2_con_z1.log_prob(z2_hat)
            dist_p_z2 = torch.distributions.normal.Normal(0,1)
            log_p_z2 = dist_p_z2.log_prob(z2_hat)
            p_z1_mu, p_z1_sigma = self._decode_2(z2_hat)
            dist_p_z1_con_z2 = torch.distributions.normal.Normal(p_z1_mu, p_z1_sigma)
            log_p_z2_con_z1 = dist_p_z1_con_z2.log_prob(z1_hat)
            x_logits = self._decode_1(z1_hat)
            dist_x_con_z1 = torch.distributions.categorical.Categorical(logits=x_logits)
            log_p_x_con_z1 = dist_x_con_z1.log_prob(x.long())
            if mode == "sgvb1":
                elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                    torch.sum(log_p_z2_con_z1, dim=1) + torch.sum(log_p_z2, dim=1) -\
                    torch.sum(log_q_z1_con_x, dim=1) -\
                    torch.sum(log_q_z2_con_z1, dim=1)
            elif mode == "sgvb2":
                elbo = torch.sum(log_p_x_con_z1, dim=(1,2,3)) +\
                    0.5 * torch.sum((1 + 2 * torch.log(z2_sigma) - z2_mu ** 2 - z2_sigma ** 2), dim=1) +\
                    0.5 * torch.sum((1 + 2 * torch.log(z1_sigma) - 2 * torch.log(p_z1_sigma) - (z1_sigma**2 + (z1_mu - p_z1_mu)**2) / (p_z1_sigma**2)), dim=1)
            else:
                raise NotImplementedError
            loss = - torch.mean(elbo / (c * h * w), dim=0)
            if params_2.grad is not None:
                params_2.grad.zero_()
            loss.backward()
            with torch.no_grad():
                vs_2 = self.mom * vs_2 - params_2.grad
                params_2 = params_2 + lr * vs_2
            params_2.requires_grad_(True)
        return elbo_favi, elbo

    def eval_savi(self, x, iter, lr, mode="sgvb2", seed=-1):
        elbo_favi, elbo_naive = self.savi_naive(x,iter,lr,mode,seed)
        _, elbo_appro = self.savi_approx(x,iter,lr,mode,seed)
        _, elbo_accur = self.savi_accurate(x,iter,lr,mode,seed)
        print("[savi] elbo favi: {0:.10f} --- naive: {1:.10f} --- approx: {2:.10f} --- accur: {3:.10f}".format(elbo_favi.item(), elbo_naive.item(), elbo_appro.item(), elbo_accur.item()))
        # assert(0)
        return elbo_favi, elbo_naive, elbo_appro, elbo_accur