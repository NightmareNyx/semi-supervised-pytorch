import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import numpy as np

from models.vae import VariationalAutoencoder
from models.vae import Encoder, Decoder, HIDecoder, LadderEncoder, LadderDecoder
from utils import batch_normalization


class Classifier(nn.Module):
    def __init__(self, dims):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x


class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, dims):
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = Classifier([x_dim, h_dim[0], self.y_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        return x_mu

    def classify(self, x):
        logits = self.classifier(x)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x


class StackedDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims, features):
        """
        M1+M2 model as described in [Kingma 2014].

        Initialise a new stacked generative model
        :param dims: dimensions of x, y, z and hidden layers
        :param features: a pretrained M1 model of class `VariationalAutoencoder`
            trained on the same dataset.
        """
        [x_dim, y_dim, z_dim, h_dim] = dims
        super(StackedDeepGenerativeModel, self).__init__([features.z_dim, y_dim, z_dim, h_dim])

        # Be sure to reconstruct with the same dimensions
        in_features = self.decoder.reconstruction.in_features
        self.decoder.reconstruction = nn.Linear(in_features, x_dim)

        # Make vae feature model untrainable by freezing parameters
        self.features = features
        self.features.train(False)

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Sample a new latent x from the M1 model
        x_sample, _, _ = self.features.encoder(x)

        # Use the sample as new input to M2
        return super(StackedDeepGenerativeModel, self).forward(x_sample, y)

    def classify(self, x):
        _, x, _ = self.features.encoder(x)
        logits = self.classifier(x)
        return logits


class AuxiliaryDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        Auxiliary Deep Generative Models [Maaløe 2016]
        code replication. The ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions.

        :param dims: dimensions of x, y, z, a and hidden layers.
        """
        [x_dim, y_dim, z_dim, a_dim, h_dim] = dims
        super(AuxiliaryDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim, h_dim])

        self.aux_encoder = Encoder([x_dim, h_dim, a_dim])  # q(a|x)
        self.aux_decoder = Encoder([x_dim + z_dim + y_dim, list(reversed(h_dim)), a_dim])  # p(a|x,y,z)

        self.classifier = Classifier([x_dim + a_dim, h_dim[0], y_dim])  # q(y|a,x)

        self.encoder = Encoder([a_dim + y_dim + x_dim, h_dim, z_dim])  # q(z|a,y,x)
        self.decoder = Decoder([y_dim + z_dim, list(reversed(h_dim)), x_dim])  # p(x|y,z)

    def classify(self, x):
        # Auxiliary inference q(a|x)
        a, a_mu, a_log_var = self.aux_encoder(x)

        # Classification q(y|a,x)
        logits = self.classifier(torch.cat([x, a], dim=1))
        return logits

    def forward(self, x, y, miss_list=None, norm_params=None):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :param norm_params: normalization parameters, not useful here
        :return: reconstruction
        """
        # Auxiliary inference q(a|x)
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x)

        # Latent inference q(z|a,y,x)
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y, q_a], dim=1))

        # Generative p(x|z,y)
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        # Generative p(a|z,y,x)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([x, y, z], dim=1))

        a_kl = self._kld(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        z_kl = self._kld(z, (z_mu, z_log_var))

        self.kl_divergence = a_kl + z_kl

        return x_mu


class HIAuxiliaryDeepGenerativeModel(AuxiliaryDeepGenerativeModel):
    def __init__(self, dims, types_list):
        """
        HI-Auxiliary Deep Generative Model. The HI-ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions, and also handles heterogeneous and missing data
        by creating an intermediate homogeneous representation of them through a
        deterministic layer "gamma" before generating heterogeneous x in the decoder.

        :param dims: dimensions of x, y, z, a, gamma and hidden layers.
        """
        [x_dim, y_dim, z_dim, a_dim, gamma_dim, h_dim] = dims
        super(HIAuxiliaryDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim, a_dim, h_dim])
        # everything else the same # TODO ???
        self.decoder = HIDecoder([y_dim + z_dim, h_dim, gamma_dim, x_dim], types_list)  # p(x|z,g(z))
        self.types_list = types_list
        self.miss_list = None
        self.x_norm = None
        self.samples_z = []
        self.samples_qa = []
        self.samples_pa = []

    def forward(self, x, y, miss_list=None, norm_params=None):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :param miss_list: miss list
        :param norm_params: normalization parameters
        :return: reconstruction
        """
        self.miss_list = miss_list
        # Batch normalization of the data
        x_norm, norm_params = batch_normalization(x, self.types_list, miss_list)
        self.x_norm = x_norm  # to classify later

        # Auxiliary inference q(a|x)
        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x_norm)
        self.samples_qa.append([q_a, y])

        # Latent inference q(z|a,y,x)
        z, z_mu, z_log_var = self.encoder(torch.cat([x_norm, y, q_a], dim=1))
        self.samples_z.append([z, y])

        # Generative p(x|g(z),y)
        # the data x are also given to later compute the reconstruction loss / log likelihood,
        # alongside parameters and samples
        # during the handling of the different likelihoods
        # It may be confusing as a design choice, but it's convenient
        log_p_x, log_p_x_missing, samples_x, params_x = self.decoder(torch.cat([z, y], dim=1), x, miss_list,
                                                                     norm_params)

        # Generative p(a|z,y,x)
        p_a, p_a_mu, p_a_log_var = self.aux_decoder(torch.cat([x_norm, y, z], dim=1))
        self.samples_pa.append([p_a, y])

        # KL(q(a|x) || p(a|z,y,x))
        a_kl = self._kld(q_a, (q_a_mu, q_a_log_var), (p_a_mu, p_a_log_var))
        # KL(q(z|a,y,x) || p(z))
        z_kl = self._kld(z=z, q_param=(z_mu, z_log_var), p_param=None)  # using z prior

        self.kl_divergence = (a_kl + z_kl).unsqueeze(1)

        return log_p_x, log_p_x_missing, samples_x, params_x

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        # TODO
        y = y.float()
        log_p_x, log_p_x_missing, samples_x, params_x = self.decoder(torch.cat([z, y], dim=1), x, miss_list,
                                                                     norm_params)
        return samples_x

    def classify(self, x=None):
        if x is None:
            x = self.x_norm
        # Auxiliary inference q(a|x)
        a, a_mu, a_log_var = self.aux_encoder(x)

        # Classification q(y|a,x)
        logits = self.classifier(torch.cat([x, a], dim=1))
        return logits


class LadderDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        Ladder version of the Deep Generative Model.
        Uses a hierarchical representation that is
        trained end-to-end to give very nice disentangled
        representations.

        :param dims: dimensions of x, y, z layers and h layers
            note that len(z) == len(h).
        """
        [x_dim, y_dim, z_dim, h_dim] = dims
        super(LadderDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim[0], h_dim])

        neurons = [x_dim, *h_dim]
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]

        e = encoder_layers[-1]
        encoder_layers[-1] = LadderEncoder([e.in_features + y_dim, e.out_features, e.z_dim])

        decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]

        self.classifier = Classifier([x_dim, h_dim[0], y_dim])

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder([z_dim[0] + y_dim, h_dim, x_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        for i, encoder in enumerate(self.encoder):
            if i == len(self.encoder) - 1:
                x, (z, mu, log_var) = encoder(torch.cat([x, y], dim=1))
            else:
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

            # Perform downword merge of information.
            else:
                z, kl = decoder(z, l_mu, l_log_var)
                self.kl_divergence += self._kld(*kl)

        x_mu = self.reconstruction(torch.cat([z, y], dim=1))
        return x_mu

    def sample(self, z, y):
        for i, decoder in enumerate(self.decoder):
            print(decoder)
            z = decoder(z)
        return self.reconstruction(torch.cat([z, y], dim=1))
