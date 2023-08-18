import torch
from torch import nn, autograd
import matplotlib.pyplot as plt
from model_utils import reset_weights
from utils import plot_windowed

# Wasserstein GAN obtained from: https://medium.com/dejunhuang/implementing-gan-and-wgan-in-pytorch-551099afde3c

class lstm_generator(nn.Module):

    def __init__(self, cond_size, noise_size, out_dim, hidden_dim=256, n_layers=1, dropout=0.0, device = torch.device('cpu')):
        super().__init__()
        self.n_layers = n_layers if n_layers > 1 else 2
        self.hidden_dim = hidden_dim
        self.cond_size = cond_size
        self.noise_size = noise_size
        self.out_dim = out_dim
        self.in_dim = cond_size + noise_size
        self.device = device

        self.encoder = nn.LSTM(input_size=self.in_dim, hidden_size=hidden_dim, num_layers=self.n_layers, batch_first=True, dropout=dropout).to(device)
        self.linear_out = nn.Linear(self.hidden_dim, out_dim).to(device)


    def initHidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=self.device), torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=self.device)


    def forward(self, input):

        batch_size, seq_len = input.size(0), input.size(1)
        h, c = self.initHidden(batch_size)

        recurrent_features = torch.empty(batch_size, seq_len, self.out_dim, device=self.device) # batch_size x  seq_len x proj_size
        hidden_features = torch.empty(self.n_layers, batch_size, seq_len, self.hidden_dim, device=self.device)
        cell_features = torch.empty(self.n_layers, batch_size, seq_len, self.hidden_dim, device=self.device)

        
        for t in range(seq_len):
            output, (h, c) = self.encoder(input[:,t,:].unsqueeze(1), (h, c))
            recurrent_features[:,t,:] = self.linear_out(output.squeeze(1))
            hidden_features[:,:,t,:] = h # Use all layers for hidden_features
            cell_features[:,:,t,:] = c

        return recurrent_features, hidden_features, cell_features


class lstm_discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1, device = torch.device('cpu')):
        super(lstm_discriminator, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True).to(device)
        self.linear = nn.Linear(hidden_size, 1).to(device)


    def forward(self, input):

        batch_size, seq_len = input.size(0), input.size(1)
        recurrent_features, _ = self.lstm(input) # If h_0, c_0 are not given they are set to zero!
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_size))
        outputs = outputs.view(batch_size, seq_len, 1)

        return outputs


class WRCGAN(nn.Module):
    
    def __init__(self, input_size, cond_in_size, noise_size, hidden_size, num_hidden_layers = 1, dropout=0.0, L1_coeff = 1.0, device = torch.device('cpu')):
        super(WRCGAN, self).__init__()

        self.G = lstm_generator(cond_size = cond_in_size, noise_size = noise_size, out_dim = input_size
                                , hidden_dim = hidden_size, n_layers=num_hidden_layers, dropout = dropout, device = device)
        self.D = lstm_discriminator(input_size+cond_in_size, hidden_size, n_layers=num_hidden_layers, device = device)
        self.device = device


    def reset_weights(self, verbose):
        reset_weights(self.G, verbose)
        reset_weights(self.D, verbose)

    
    def gradient_penalty_old(self, xr, xf):
        """
        :param xr: [b, 2]
        :param xf: [b, 2]
        :return:
        """
        # [b, 1]
        batch_size = xr.shape[0]
        t = torch.rand(xr.shape[0], xr.shape[1], 1, device=self.device)
        # [b, 1] => [b, 2]  broadcasting so t is the same for x1 and x2
        t = t.expand_as(xr)
        # interpolation
        mid = t * xr + (1 - t) * xf
        # set it to require grad info
        mid.requires_grad_()
        pred = self.D(mid)
        grads = autograd.grad(outputs=pred, inputs=mid,
                            grad_outputs=torch.ones_like(pred),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_norm = grads.norm(2, dim=1)

        gp = torch.pow(gradient_norm - 1, 2).mean()

        return gp



    # From: https://medium.com/the-ai-team/gans-to-wasserstein-gans-74028d3b0aae
    def get_gradient(self, xr, xf, t):

        mid = t*xr + (1-t)*xf

        if self.device.type == 'cuda':
            with torch.backends.cudnn.flags(enabled=False):
                pred = self.D(mid)
        else:
            pred = self.D(mid)

        """
        Computes and returns the sum of gradients of outputs with respect to the inputs.
        See also: https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
        """

        grads = torch.autograd.grad(outputs = pred,
                                        inputs = mid,
                                        grad_outputs = torch.ones_like(pred), # Equivalent (but faster) to grad_outputs = None. You can see these outputs as providing dL/dout (where L is your loss) so that the autograd can compute dL/dw (where w are the parameters for which you want the gradients) as dL/dw = dL/dout * dout/dw.
                                        create_graph = True, #  If True, graph of the derivative will be constructed, allowing to compute higher order derivative products. 
                                        retain_graph = True)[0]
        return grads

    # From: https://medium.com/the-ai-team/gans-to-wasserstein-gans-74028d3b0aae as ('get_gradient()' above)
    def gradient_penalty_new(self, grads):
        grads = grads.contiguous()
        grads = grads.view(len(grads), -1)
        gradient_norm = grads.norm(2, dim=1)
        gp = torch.nn.MSELoss()(gradient_norm, torch.ones_like(gradient_norm))
        return gp

    def gradient_epsilon_and_penalty(self, real_batch, fake_batch):
        epsilon = torch.rand(len(real_batch),1,1, device = self.device, requires_grad=True)
        gradient = self.get_gradient(real_batch, fake_batch.detach(), epsilon)
        gp = self.gradient_penalty_new(gradient)
        return gp



def cgan_cond(cond_history):
    cond = cond_history
    return cond



def noise_like_sequence(shape, device):

    noise = torch.randn(shape, device=device) # batch_size x input_window x n_features
    return noise


def train_iteration_WRCGAN(gan, X_history, y_history, gan_gen_optimizer, gan_disc_optimizer):

    seq_dim = 1

    #######################
    # Train Discriminator #
    #######################

    gan_disc_optimizer.zero_grad()
    
    # Create condition
    cond = cgan_cond(X_history)

    # Create real batch: [Condition | Real data]
    real_batch = torch.cat((y_history, cond), dim=2)

    # Create fake-generating batch: [Condition | noise]
    noise = noise_like_sequence(y_history.shape, y_history.device)
    gen_input_batch = torch.cat((noise, cond), dim=2) # batch_size x input_window x (fs_input_size + noise_input_size(=fs_input_size) )  

    # Create fake batch from fake-generating batch:
    if gan.device.type == 'cuda':
        with torch.backends.cudnn.flags(enabled=False):
            fake_y_history, _, _ = gan.G(gen_input_batch)
    else:
        fake_y_history, _, _ = gan.G(gen_input_batch)

    fake_batch = torch.cat((fake_y_history, cond), dim=2)

    # Calculate Discriminator Loss and Total Loss # 
    gp = gan.gradient_epsilon_and_penalty(real_batch, fake_batch)

    # Adversarial Loss using fake batch #
    if gan.device.type == 'cuda':
        with torch.backends.cudnn.flags(enabled=False):
            prob_fake = gan.D(fake_batch)
    else:
        prob_fake = gan.D(fake_batch)

    D_fake_loss = prob_fake.mean()

    # Adversarial Loss using real batch #
    prob_real = gan.D(real_batch)

    D_real_loss = -prob_real.mean()
    # END Adversarial Loss using real batch #

    D_loss = D_fake_loss - D_real_loss # https://github.com/u7javed/Conditional-WGAN-GP/blob/master/train.py
    wass_loss = D_fake_loss + D_real_loss + 0.2 * gp

    wass_loss.backward() # https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350/15
    gan_disc_optimizer.step()

    ###################
    # Train Generator #
    ###################

    gan_gen_optimizer.zero_grad()

    # Recalculate - we already did a back-propagation on D_loss so we have to do a forward again: ...
    # ...see also: https://stackoverflow.com/questions/69123542/trying-to-backward-through-the-graph-a-second-time-with-gans-model
    fake_y_history, gen_hidden, gen_cell = gan.G(gen_input_batch)
    fake_batch = torch.cat((fake_y_history, cond), dim=2)
    prob_fake = gan.D(fake_batch)
    
    G_loss = -prob_fake.mean()

    return G_loss, D_loss, gen_hidden, gen_cell, fake_y_history
