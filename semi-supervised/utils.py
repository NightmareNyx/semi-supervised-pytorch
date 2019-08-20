import torch
from torch.autograd import Variable


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """

    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())


def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """

    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y

    return encode


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max


def dynamic_partition(data, partitions, num_partitions):
    res = []
    for i in range(num_partitions):
        res += [data[(partitions == i).nonzero().squeeze(1)]]
    return res


def dynamic_partition_(inputs, labels):
    '''
    Splits tensor along first axis according to partition label

    Input: - inputs: tensor(shape=(N,...), dtype=T)
           - labels: tensor(shape=(N), dtype=?) with M<=N unique values
             the k-th unique value occurs exactly N_k times
    Returns: outputs: [tensor(shape=(N_1,...), dtype=T), ..., tensor(shape=(N_M,...), dtype=T)]
    '''
    classes = torch.unbind(torch.unique(labels))
    inputs_partition = [inputs[labels == cl] for cl in classes]
    return inputs_partition


def dynamic_stitch(indices, data):
    n = sum(idx.numel() for idx in indices)
    res = [None] * n
    for i, data_ in enumerate(data):
        idx = indices[i].view(-1)
        d = data_.view(idx.numel(), -1)
        k = 0
        for idx_ in idx: res[idx_] = d[k]; k += 1
    return res


def dynamic_stitch_(inputs, conditional_indices):
    maxs = []
    for ci in conditional_indices:
        maxs.append(torch.max(ci))
    size = int(max(maxs)) + 1
    stitched = torch.Tensor(size)
    for i, idx in enumerate(conditional_indices):
        stitched[idx] = inputs[i]
    return stitched


def one_hot(seq_batch, depth):
    # seq_batch.size() should be [seq,batch] or [batch,]
    # return size() would be [seq,batch,depth] or [batch,depth]
    out = torch.zeros(seq_batch.size() + torch.Size([depth]))
    dim = len(seq_batch.size())
    index = seq_batch.view(seq_batch.size() + torch.Size([1]))
    return out.scatter_(dim, index, 1)


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    mask = ~(torch.ones((len(lengths), maxlen)).cumsum(dim=1).t() > lengths).t()
    mask.type(dtype)
    return mask


def log_poisson_loss(targets, log_input, compute_full_loss=False):
    """Computes log Poisson loss given `log_input`.
    Gives the log-likelihood loss between the prediction and the target under the
    assumption that the target has a Poisson distribution.
    Caveat: By default, this is not the exact loss, but the loss minus a
    constant term [log(z!)]. That has no effect for optimization, but
    does not play well with relative loss comparisons. To compute an
    approximation of the log factorial term, specify
    compute_full_loss=True to enable Stirling's Approximation.
    For brevity, let `c = log(x) = log_input`, `z = targets`.  The log Poisson
    loss is
        -log(exp(-x) * (x^z) / z!)
      = -log(exp(-x) * (x^z)) + log(z!)
      ~ -log(exp(-x)) - log(x^z) [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
          [ Note the second term is the Stirling's Approximation for log(z!).
            It is invariant to x and does not affect optimization, though
            important for correct relative loss comparisons. It is only
            computed when compute_full_loss == True. ]
      = x - z * log(x) [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
      = exp(c) - z * c [+ z * log(z) - z + 0.5 * log(2 * pi * z)]
    Args:
    targets: A `Tensor` of the same type and shape as `log_input`.
    log_input: A `Tensor` of type `float32` or `float64`.
    compute_full_loss: whether to compute the full loss. If false, a constant
      term is dropped in favor of more efficient optimization.
    name: A name for the operation (optional).
    Returns:
    A `Tensor` of the same shape as `log_input` with the componentwise
    logistic losses.
    Raises:
    ValueError: If `log_input` and `targets` do not have the same shape.
    """
    try:
        assert(targets.size() == log_input.size())
    except ValueError:
        raise ValueError(
            "log_input and targets must have the same shape (%s vs %s)" %
            (log_input.size(), targets.size()))

    result = torch.exp(log_input) - log_input * targets
    if compute_full_loss:
        # need to create constant tensors here so that their dtypes can be matched
        # to that of the targets.
        point_five = 0.5  # constant_op.constant(0.5, dtype=targets.dtype)
        two_pi = 2 * math.pi  # constant_op.constant(2 * math.pi, dtype=targets.dtype)

        stirling_approx = (targets * torch.log(targets)) - targets + (
                point_five * torch.log(two_pi * targets))
        zeros = torch.zeros_like(targets, dtype=targets.dtype)
        ones = torch.ones_like(targets, dtype=targets.dtype)
        cond = (targets >= zeros) & (targets <= ones)  # math_ops.logical_and(targets >= zeros, targets <= ones)
        result += torch.where(cond, zeros, stirling_approx)
    return result
