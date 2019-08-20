#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Imports
import torch

cuda = torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../semi-supervised")

# # Ladder Deep Generative Model
# 
# The final model we present is the ladder deep generative model which combines the idea of the ladder variational auto encoder with the deep generative model. It therefore learns a hierarchical relationship of latent variables and therefore has many of the properties of standard ladder VAEs such as strong generative performance and good disentanglement. The model below shows a three layer LadderDGM, but the number of stochastic layer can be arbitrary.
# 
# <img src="../images/ladderdgm.png" width="400px"/>
# 
# This model serves as an example of what you can do with the tools provided this library to create new and novel models.

# In[5]:


from models import LadderDeepGenerativeModel

y_dim = 10
z_dim = [64, 32, 16]
h_dim = [256, 128, 64]

model = LadderDeepGenerativeModel([784, y_dim, z_dim, h_dim])
model

# ## Training
# 
# We train the model similarly to the Ladder VAE, but add label information to the lower bound. The final bound is
# similar to that presented in the "Deep Generative Model" notebook. From a programming point of view, the model is
# trained in exactly the same way as the `DeepGenerativeModel` and the `AuxiliaryDeepGenerativeModel`.
# 
# This simultanously highlights the power of the library, as it is model agnostic as long as the given model subclasses the `VariationalAutoencoder` class.

# In[6]:


from datautils import get_mnist

# Only use 10 labelled examples per class
# The rest of the data is unlabelled.
labelled, unlabelled, validation = get_mnist(location="./", batch_size=64, labels_per_class=10)
alpha = 0.1 * (len(unlabelled) + len(labelled)) / len(labelled)


def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

# In[8]:


from itertools import cycle
from inference import SVI, DeterministicWarmup

# We will need to use warm-up in order to achieve good performance.
# Over 200 calls to SVI we change the autoencoder from
# deterministic to stochastic.
beta = DeterministicWarmup(n=200)

if cuda: model = model.cuda()
elbo = SVI(model, likelihood=binary_cross_entropy, beta=beta)

# The library is conventially packed with the `SVI` method that does all of the work of calculating the lower bound
# for both labelled and unlabelled data depending on whether the label is given. It also manages to perform the
# enumeration of all the labels.
# 
# Remember that the labels have to be in a *one-hot encoded* format in order to work with SVI.

# In[ ]:


from torch.autograd import Variable

for epoch in range(1):
    model.train()
    total_loss, accuracy = (0, 0)
    for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
        # Wrap in variables
        x, y, u = Variable(x), Variable(y), Variable(u)

        if cuda:
            # They need to be on the same device and be synchronized.
            x, y = x.cuda(device=0), y.cuda(device=0)
            u = u.cuda(device=0)

        L = -elbo(x, y)
        U = -elbo(u)

        # Add auxiliary classification loss q(y|x)
        logits = model.classify(x)

        # Regular cross entropy
        classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

        J_alpha = L - alpha * classication_loss + U

        J_alpha.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += J_alpha.data.item()
        accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

    if epoch % 1 == 0:
        model.eval()
        m = len(unlabelled)
        print("Epoch: {}".format(epoch))
        print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

        total_loss, accuracy = (0, 0)
        for x, y in validation:
            x, y = Variable(x), Variable(y)

            if cuda:
                x, y = x.cuda(device=0), y.cuda(device=0)

            L = -elbo(x, y)
            U = -elbo(x)

            logits = model.classify(x)
            classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

            J_alpha = L + alpha * classication_loss + U

            total_loss += J_alpha.data.item()

            _, pred_idx = torch.max(logits, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

        m = len(validation)
        print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

# ## Conditional generation
# 
# When the model is done training you can generate samples conditionally given some normal distributed noise $z$ and a label $y$.
# 
# *The model below has only trained for 10 iterations, so the perfomance is not representative*.

# In[22]:


from utils import onehot

model.eval()

z = Variable(torch.randn(16, 32))

# Generate a batch of 5s
y = Variable(onehot(10)(5).repeat(16, 1))

x_mu = model.sample(z, y)

# In[13]:


f, axarr = plt.subplots(1, 16, figsize=(18, 12))

samples = x_mu.data.view(-1, 28, 28).numpy()

for i, ax in enumerate(axarr.flat):
    ax.imshow(samples[i])
    ax.axis("off")

# In[ ]:
