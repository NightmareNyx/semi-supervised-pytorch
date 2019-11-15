import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np

from layers import GaussianSample, GaussianMerge, GumbelSoftmax
from inference import log_gaussian, log_standard_gaussian
from utils import dynamic_partition, dynamic_stitch, batch_normalization

from inference import loglik


class Perceptron(nn.Module):
    def __init__(self, dims, activation_fn=F.relu, output_activation=None):
        super(Perceptron, self).__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.output_activation = output_activation

        self.layers = nn.ModuleList(list(map(lambda d: nn.Linear(*d), list(zip(dims, dims[1:])))))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.output_activation is not None:
                x = self.output_activation(x)
            else:
                x = self.activation_fn(x)

        return x


class Encoder(nn.Module):
    def __init__(self, dims, sample_layer=GaussianSample):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(Encoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        neurons = [x_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.sample(x)


class Decoder(nn.Module):
    def __init__(self, dims):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims

        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)

        self.reconstruction = nn.Linear(h_dim[-1], x_dim)

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output_activation(self.reconstruction(x))


class HIDecoder(nn.Module):
    def __init__(self, dims, types_list):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(HIDecoder, self).__init__()

        [z_dim, h_dim, gamma_dim, x_dim] = dims

        self.types_list = types_list
        self.gamma_dim_partition = gamma_dim * np.ones(len(types_list), dtype=int)
        self.gamma_dim_output = np.sum(self.gamma_dim_partition)
        self.hidden = None
        gamma_input_dim = z_dim
        if h_dim is not None or h_dim != [] or h_dim != 0:
            neurons = [z_dim, *h_dim]
            linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
            self.hidden = nn.ModuleList(linear_layers)
            gamma_input_dim = h_dim[-1]
        # deterministic homogeneous gamma layer
        self.gamma_layer = nn.Linear(gamma_input_dim, self.gamma_dim_output)
        self.obs_layer = self.get_obs_layers(gamma_dim)
        # self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        # self.output_activation = nn.Sigmoid()

    def get_obs_layers(self, gamma_dim):
        # Different layer models for each type of variable
        obs_layers = []
        for type in self.types_list:
            type_dim = type['dim']
            if type['type'] == 'real' or type['type'] == 'pos':
                obs_layers.append(nn.ModuleList([nn.Linear(gamma_dim, type_dim),  # mean
                                                nn.Linear(gamma_dim, type_dim)]))  # sigma
            elif type['type'] == 'count':
                obs_layers.append(nn.ModuleList([nn.Linear(gamma_dim, type_dim)]))  # lambda
            elif type['type'] == 'cat':
                obs_layers.append(nn.ModuleList[nn.Linear(gamma_dim, type_dim - 1)])  # log pi
            elif type['type'] == 'ord':
                obs_layers.append(nn.ModuleList[nn.Linear(gamma_dim, type_dim - 1),  # theta
                                                nn.Linear(gamma_dim, 1)])           # mean, single value
        return nn.ModuleList(obs_layers)

    def forward(self, z, batch_x, miss_list, norm_params):
        if self.hidden is not None:
            for layer in self.hidden:
                z = F.relu(layer(z))
        # Deterministic homogeneous representation gamma = g(z)
        gamma = self.gamma_layer(z)
        gamma_grouped = self.gamma_partition(gamma)
        theta = self.theta_estimation_from_gamma(gamma_grouped, miss_list)

        log_p_x, log_p_x_missing, samples_x, params_x = self.loglik_and_reconstruction(theta, batch_x,
                                                                                       miss_list, norm_params)
        return log_p_x, log_p_x_missing, samples_x, params_x

    def loglik_and_reconstruction(self, theta, batch_data, miss_list, normalization_params):
        log_p_x = []
        log_p_x_missing = []
        samples_x = []
        params_x = []

        # independent gamma_d -> Compute log(p(xd|gamma_d))
        for i, d in enumerate(batch_data.T):
            d = d.unsqueeze(1)
            # Select the likelihood for the types of variables
            loglik_function = getattr(loglik, 'loglik_' + self.types_list[i]['type'])
            out = loglik_function([d, miss_list[:, i]], self.types_list[i], theta[i], normalization_params[i])

            log_p_x.append(out['log_p_x'].unsqueeze(1))
            log_p_x_missing.append(out['log_p_x_missing'].unsqueeze(1))  # Test-loglik element
            samples_x.append(out['samples'])
            params_x.append(out['params'])
        # return log_p_x, log_p_x_missing, samples_x, params_x
        return torch.cat(log_p_x, dim=1), torch.cat(log_p_x_missing, dim=1), torch.cat(samples_x, dim=1), params_x

    def gamma_partition(self, gamma):
        grouped_samples_gamma = []
        # First element must be 0 and the length of the partition vector must be len(types_dict)+1
        if len(self.gamma_dim_partition) != len(self.types_list):
            raise Exception("The length of the partition vector must match the number of variables in the data + 1")
        # Insert a 0 at the beginning of the cumsum vector
        partition_vector_cumsum = np.insert(np.cumsum(self.gamma_dim_partition), 0, 0)
        for i in range(len(self.types_list)):
            grouped_samples_gamma.append(gamma[:, partition_vector_cumsum[i]:partition_vector_cumsum[i + 1]])
        return grouped_samples_gamma

    def theta_estimation_from_gamma(self, gamma, miss_list):
        theta = []
        # independent yd -> Compute p(xd|yd)
        for i, d in enumerate(gamma):  # gamma is a list
            # Partition the data in missing data (0) and observed data (1)
            missing_y, observed_y = dynamic_partition(d, miss_list[:, i], 2)
            condition_indices = dynamic_partition(torch.arange(d.size()[0]), miss_list[:, i], 2)
            # Different layer models for each type of variable
            params = self.observed_data_layer(observed_y, missing_y, condition_indices, i)
            theta.append(params)
        return theta

    def observed_data_layer(self, observed_data, missing_data, condition_indices, i):
        outputs = []
        for obs_layer_param in self.obs_layer[i]:
            obs_output = obs_layer_param(observed_data)
            with torch.no_grad():
                miss_output = obs_layer_param(missing_data)
            # Join back the data
            output = dynamic_stitch(condition_indices, [miss_output, obs_output])
            if len(self.obs_layer[i]) == 1:
                return output
            outputs.append(output)
        return outputs


class VariationalAutoencoder(nn.Module):
    def __init__(self, dims):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VariationalAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.flow = None

        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, p_param=None):
        """
        Computes the KL-divergence of
        some element z.

        KL(q||p) = -∫ q(z) log [ p(z) / q(z) ]
                 = -E[log p(z) - log q(z)]

        :param z: sample from q-distribution
        :param q_param: (mu, log_var) of the q-distribution
        :param p_param: (mu, log_var) of the p-distribution
        :return: KL(q||p)
        """
        (mu, log_var) = q_param

        if self.flow is not None:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)

        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz

        return kl

    def add_flow(self, flow):
        self.flow = flow

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z, z_mu, z_log_var = self.encoder(x)

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        x_mu = self.decoder(z)

        return x_mu

    def sample(self, z):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)


class GumbelAutoencoder(nn.Module):
    def __init__(self, dims, n_samples=100):
        super(GumbelAutoencoder, self).__init__()

        [x_dim, z_dim, h_dim] = dims
        self.z_dim = z_dim
        self.n_samples = n_samples

        self.encoder = Perceptron([x_dim, *h_dim])
        self.sampler = GumbelSoftmax(h_dim[-1], z_dim, n_samples)
        self.decoder = Perceptron([z_dim, *reversed(h_dim), x_dim], output_activation=F.sigmoid)

        self.kl_divergence = 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, qz):
        k = Variable(torch.FloatTensor([self.z_dim]), requires_grad=False)
        kl = qz * (torch.log(qz + 1e-8) - torch.log(1.0 / k))
        kl = kl.view(-1, self.n_samples, self.z_dim)
        return torch.sum(torch.sum(kl, dim=1), dim=1)

    def forward(self, x, y=None, tau=1):
        x = self.encoder(x)

        sample, qz = self.sampler(x, tau)
        self.kl_divergence = self._kld(qz)

        x_mu = self.decoder(sample)

        return x_mu

    def sample(self, z):
        return self.decoder(z)


class LadderEncoder(nn.Module):
    def __init__(self, dims):
        """
        The ladder encoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions [input_dim, [hidden_dims], [latent_dims]].
        """
        super(LadderEncoder, self).__init__()
        [x_dim, h_dim, self.z_dim] = dims
        self.in_features = x_dim
        self.out_features = h_dim

        self.linear = nn.Linear(x_dim, h_dim)
        self.batchnorm = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, self.z_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.leaky_relu(self.batchnorm(x), 0.1)
        return x, self.sample(x)


class LadderDecoder(nn.Module):
    def __init__(self, dims):
        """
        The ladder dencoder differs from the standard encoder
        by using batch-normalization and LReLU activation.
        Additionally, it also returns the transformation x.

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(LadderDecoder, self).__init__()

        [self.z_dim, h_dim, x_dim] = dims

        self.linear1 = nn.Linear(x_dim, h_dim)
        self.batchnorm1 = nn.BatchNorm1d(h_dim)
        self.merge = GaussianMerge(h_dim, self.z_dim)

        self.linear2 = nn.Linear(x_dim, h_dim)
        self.batchnorm2 = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, self.z_dim)

    def forward(self, x, l_mu=None, l_log_var=None):
        if l_mu is not None:
            # Sample from this encoder layer and merge
            z = self.linear1(x)
            z = F.leaky_relu(self.batchnorm1(z), 0.1)
   
        # Sample from the decoder and send forward
        z = self.linear2(x)
        z = F.leaky_relu(self.batchnorm2(z), 0.1)
        z, p_mu, p_log_var = self.sample(z)

        if l_mu is None:
            return z

        return z, (q_z, (q_mu, q_log_var), (p_mu, p_log_var))


class LadderVariationalAutoencoder(VariationalAutoencoder):
    def __init__(self, dims):
        """
        Ladder Variational Autoencoder as described by
        [Sønderby 2016]. Adds several stochastic
        layers to improve the log-likelihood estimate.

        :param dims: x, z and hidden dimensions of the networks
        """
        [x_dim, z_dim, h_dim] = dims
        super(LadderVariationalAutoencoder, self).__init__([x_dim, z_dim[0], h_dim])

        neurons = [x_dim, *h_dim]
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]
        decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder([z_dim[0], h_dim, x_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        for encoder in self.encoder:
            x, (z, mu, log_var) = encoder(x)
            latents.append((mu, log_var))

        latents = list(reversed(latents))

        self.kl_divergence = 0
        for i, decoder in enumerate([-1, *self.decoder]):
            # If at top, encoder == decoder,
            # use prior for KL.
            l_mu, l_log_var = latents[i]
            if i == 0:
                self.kl_divergence += self._kld(z, (l_mu, l_log_var))

            # Perform downward merge of information.
            else:
                z, kl = decoder(z, l_mu, l_log_var)
                self.kl_divergence += self._kld(*kl)

        x_mu = self.reconstruction(z)
        return x_mu

    def sample(self, z):
        for decoder in self.decoder:
            z = decoder(z)
        return self.reconstruction(z)
