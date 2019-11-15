#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import torch
cuda = torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import sys
sys.path.append("../../semi-supervised")


# In[2]:


from utils import batch_normalization, dynamic_partition, dynamic_stitch


# # HIAuxiliary Deep Generative Model
# 
# The Auxiliary Deep Generative Model [[Maaløe, 2016]](https://arxiv.org/abs/1602.05473) posits a model that with an auxiliary latent variable $a$ that infers the variables $z$ and $y$. This helps in terms of semi-supervised learning by delegating causality to their respective variables. This model was state-of-the-art in semi-supervised until 2017, and is still very powerful with an MNIST accuracy of *99.4%* using just 10 labelled examples per class.
# 
# <img src="../images/adgm.png" width="400px"/>
# 

# ## Training
# 
# The lower bound we derived in the notebook for the **deep generative model** is similar to the one for the ADGM. Here, we also need to integrate over a continuous auxiliary variable $a$.
# 
# For labelled data, the lower bound is given by.
# \begin{align}
# \log p(x,y) &= \log \int \int p(x, y, a, z) \ dz \ da\\
# &\geq \mathbb{E}_{q(a,z|x,y)} \bigg [\log \frac{p(x,y,a,z)}{q(a,z|x,y)} \bigg ] = - \mathcal{L}(x,y)
# \end{align}
# 
# Again when no label information is available we sum out all of the labels.
# 
# \begin{align}
# \log p(x) &= \log \int \sum_{y} \int p(x, y, a, z) \ dz \ da\\
# &\geq \mathbb{E}_{q(a,y,z|x)} \bigg [\log \frac{p(x,y,a,z)}{q(a,y,z |x)} \bigg ] = - \mathcal{U}(x)
# \end{align}
# 
# Where we decompose the q-distribution into its constituent parts. $q(a, y, z|x) = q(z|a,y,x)q(y|a,x)q(a|x)$, which is also what can be seen in the figure.
# 
# The distribution over $a$ is similar to $z$ in the sense that it is also a diagonal Gaussian distribution. However by introducing the auxiliary variable we allow for $z$ to become arbitrarily complex - something we can also see when using normalizing flows.

# In[3]:


# from datautils import get_mnist

# # Only use 10 labelled examples per class
# # The rest of the data is unlabelled.
# labelled, unlabelled, validation = get_mnist(location="./", batch_size=64, labels_per_class=10)
# alpha = 0.1 * (len(unlabelled) + len(labelled)) / len(labelled)


# In[4]:


import read_functions as rf


# In[5]:


data_file = 'Wine/data.csv'
types_file = 'Wine/data_types.csv'
miss_file = 'Wine/Missing10-50_1.csv'
true_miss_file = None
batch_size = 128


# In[6]:


train_data, types_dict, miss_mask, true_miss_mask, n_samples = rf.read_data(data_file, types_file,
                                                                                        miss_file,
                                                                                        true_miss_file)
# Randomize the data in the mini-batches
random_perm = np.random.permutation(range(np.shape(train_data)[0]))
train_data_aux = train_data[random_perm, :]
miss_mask_aux = miss_mask[random_perm, :]
true_miss_mask_aux = true_miss_mask[random_perm, :]

# Check batch size
if batch_size > n_samples:
    batch_size = n_samples
    
for t in types_dict:
    t['dim'] = int(t['dim'])

# Compute the real miss_mask
miss_mask = np.multiply(miss_mask, true_miss_mask)

train_data = torch.Tensor(train_data)
miss_mask = torch.Tensor(miss_mask)

labelled = train_data[miss_mask[:,-1] == 0, :-2], train_data[miss_mask[:,-1] == 0, -2:]
unlabelled = train_data[miss_mask[:,-1] == 1, :-2], train_data[miss_mask[:,-1] == 1, -2:]
miss_mask = miss_mask[:,:-1]
type_label = types_dict[-1]
types_dict = types_dict[:-1]


# In[7]:


alpha = 0.1 * (len(unlabelled) + len(labelled)) / len(labelled)


# In[8]:


x, y = labelled
u, v = unlabelled


# In[9]:


# Get an integer number of batches
n_batches = int(np.floor(max(x.size()[0], u.size()[0]) / batch_size))
n_batches


# In[10]:


if x.size()[0] < u.size()[0]:
    xx = torch.zeros_like(u)
    yy = torch.zeros_like(v)
    xx[:x.size()[0], :] = x
    xx[x.size()[0]:, :] = x[:u.size()[0] - x.size()[0], :]
    x = xx
    yy[:y.size()[0], :] = y
    yy[y.size()[0]:, :] = y[:v.size()[0] - y.size()[0], :]
    y = yy


# In[11]:


u.size(), x.size(), y.size(), v.size(), len(types_dict)


# In[12]:


from models import HIAuxiliaryDeepGenerativeModel

y_dim = y.size()[1]
z_dim = 2
a_dim = 2
gamma_dim = 5
h_dim = [5]
input_dim = x.size()[1]

model = HIAuxiliaryDeepGenerativeModel([input_dim, y_dim, z_dim, a_dim, gamma_dim, h_dim], types_dict)
model


# In[13]:


def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))


# In[14]:


from itertools import cycle
from inference import HISVI, DeterministicWarmup

# We will need to use warm-up in order to achieve good performance.
# Over 200 calls to SVI we change the autoencoder from
# deterministic to stochastic.
beta = DeterministicWarmup(n=200)


if cuda: model = model.cuda()
elbo = HISVI(model, beta=beta)


# In[21]:


for epoch in range(10):
    total_loss, accuracy = (0, 0)
    for i in range(n_batches):
        # Create inputs for the feed_dict
        data_tensor, labels, batch_miss_mask = rf.next_batch(x, y, types_dict, miss_mask,
                                                         batch_size, index_batch=i)
        unl_data_tensor, hid_labels, unl_batch_miss_mask = rf.next_batch(u, v, types_dict, miss_mask,
                                                         batch_size, index_batch=i)

        # Delete not known data (input zeros)
        data_observed = data_tensor * batch_miss_mask
        unl_data_observed = unl_data_tensor * unl_batch_miss_mask

        # easier names
        x_batch = data_observed
        u_batch = unl_data_observed
        y_batch = labels
        v_batch = hid_labels

        ######
        if cuda:
            # They need to be on the same device and be synchronized.
            x_batch, y_batch = x_batch.cuda(device=0), y_batch.cuda(device=0)
            u_batch, v_batch = u_batch.cuda(device=0), v_batch.cuda(device=0)

        L = -elbo(x_batch, y_batch, batch_miss_mask)

        # Add auxiliary classification loss q(y|x,a)
        logits = model.classify(x_batch)

        U = -elbo(u_batch, y=None, miss_list=unl_batch_miss_mask)

        # Add auxiliary classification loss q(y|x,a)
        logits = model.classify(x_batch)

        # Regular cross entropy
        classication_loss = torch.sum(y_batch * torch.log(logits + 1e-8), dim=1).mean()

        J_alpha = L - alpha * classication_loss + U
        
        print(J_alpha)

        J_alpha.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += J_alpha.item()
        accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y_batch, 1)[1].data).float())
        ######
        
         # TODO
#     if epoch % 1 == 0:
#         model.eval()
#         m = len(unlabelled)
#         print("Epoch: {}".format(epoch))
#         print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

#         total_loss, accuracy = (0, 0)
#         for x, y in validation:
#             x, y = Variable(x), Variable(y)

#             if cuda:
#                 x, y = x.cuda(device=0), y.cuda(device=0)

#             L = -elbo(x, y)
#             U = -elbo(x)

#             logits = model.classify(x)
#             classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

#             J_alpha = L + alpha * classication_loss + U

#             total_loss += J_alpha.data[0]

#             _, pred_idx = torch.max(logits, 1)
#             _, lab_idx = torch.max(y, 1)
#             accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

#         m = len(validation)
#         print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))


# The library is conventially packed with the `SVI` method that does all of the work of calculating the lower bound for both labelled and unlabelled data depending on whether the label is given. It also manages to perform the enumeration of all the labels.
# 
# Remember that the labels have to be in a *one-hot encoded* format in order to work with SVI.

# ## Conditional generation
# 
# When the model is done training you can generate samples conditionally given some normal distributed noise $z$ and a label $y$.
# 
# *The model below has only trained for 10 iterations, so the perfomance is not representative*.

# In[ ]:


from utils import onehot
model.eval()

z = Variable(torch.randn(16, 32))

# Generate a batch of 5s
y = Variable(onehot(10)(5).repeat(16, 1))

x_mu = model.sample(z, y)


# In[ ]:


f, axarr = plt.subplots(1, 16, figsize=(18, 12))

samples = x_mu.data.view(-1, 28, 28).numpy()

for i, ax in enumerate(axarr.flat):
    ax.imshow(samples[i])
    ax.axis("off")

