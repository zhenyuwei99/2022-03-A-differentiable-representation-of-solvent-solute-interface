# Basic conceptions

## underlying data
Appears in:
- Torch.storage
- Tensor.view()

Basic idea is every tensor is a specific view of the origin contiguous, one-dimensional array of a single data type.
> A `torch.Storage` is a contiguous, one-dimensional array of a single data type.

So one Torch.storage can have different view of Tensor, with different dimensions.

## Embedding

Embedding technique exist for replacing the low efficient one-hot embedding method for large system.

An embedding example

For a sentence:

> Deep learning is very deep

We can align an unique index for each word, turning sentence into:

> 1、2、3、4、1

Then we give how many latent parameters are used to determine a vector used to represent each index, giving an embedding matrix:

![center](./image/embedding_matrix.png)


Then a large one-hot encode vector is replaced by a embedding matrix, increasing efficiency dramatically

# nn package

## nn.Module
[doc](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

`torch.nn.Module`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes:

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

## nn.ModuleList
[doc](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList)

`torch.nn.ModuleList(modules=None)`

Holds submodules in a list.

`ModuleList` can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all `Module` methods.

Parameters
- `modules` (iterable, optional) – an iterable of modules to add

Examples
```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
```

## nn.Linear
[doc](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)

`torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)`

Applies a linear transformation to the incoming data:

$$
y = xA^T + b
$$

Parameters

- `in_features` – size of each input sample

- `out_features` – size of each output sample

- `bias` – If set to False, the layer will not learn an additive bias. Default: `True`

Shape:

- Input: $(*, H_{in})$
- Output: $(*, H_{out})$

## nn.LayerNorm
[doc](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm)

`torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)`

Applies Layer Normalization over a mini-batch of inputs:

$$
y = \frac{x - \mathrm{E}[x]}{\mathrm{Var}[x] + \varepsilon} * \gamma + \beta
$$

- The mean and standard-deviation are calculated over the last D dimensions, where D is the dimension of `normalized_shape`
- $\gamma$ and $\beta$ are learnable affine transform parameters of `normalized_shape` if `elementwise_affine` is True.

Parameters:
- `normalized_shape` (int or list or torch.Size)
    input shape from an expected input of size
    $$
    [* \times \mathrm{normalized\_shape}[0] \times \mathrm{normalized\_shape}[1] \times \cdots \times \mathrm{normalized\_shape}[-1]]
    $$
    If a single integer is used, it is treated as a singleton list, and this module will normalize over the last dimension which is expected to be of that specific size.
- `eps` – a value added to the denominator for numerical stability. Default: 1e-5
- `elementwise_affine` – a boolean value that when set to `True`, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases). Default: True.

## nn.Dropout
[doc](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout)

`torch.nn.Dropout(p=0.5, inplace=False)`

During training, randomly zeroes some of the elements of the input tensor with probability `p` using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.

> This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the paper Improving neural networks by preventing co-adaptation of feature detectors .

Furthermore, the outputs are scaled by a factor of
$$
\frac{1}{1-p}
$$
during training. This means that during evaluation the module simply computes an identity function.

Parameters
- `p` – probability of an element to be zeroed. Default: `0.5`
- `inplace` – If set to True, will do this operation in-place. Default: `False`

## nn.Embedding
[doc](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding)

`torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None)`

A simple lookup table that stores embeddings of a fixed dictionary and size.

This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.

Parameters
- `num_embeddings` (int) – size of the dictionary of embeddings
- `embedding_dim` (int) – the size of each embedding vector
- `padding_idx` (int, optional) – If specified, the entries at `padding_idx` do not contribute to the gradient; therefore, the embedding vector at `padding_idx` is not updated during training, i.e. it remains as a fixed “pad”. For a newly constructed Embedding, the embedding vector at `padding_idx` will default to all zeros, but can be updated to another value to be used as the padding vector.
- `max_norm` (float, optional) – If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.
- `norm_type` (float, optional) – The p of the p-norm to compute for the max_norm option. Default `2`.
- `scale_grad_by_freq` (boolean, optional) – If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default `False`.
- `sparse` (bool, optional) – If `True`, gradient w.r.t. `weight` matrix will be a sparse tensor. See Notes for more details regarding sparse gradients.



```python
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(10, 3)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
>>> embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])


>>> # example with padding_idx
>>> embedding = nn.Embedding(10, 3, padding_idx=0)
>>> input = torch.LongTensor([[0,2,0,5]])
>>> embedding(input)
tensor([[[ 0.0000,  0.0000,  0.0000],
         [ 0.1535, -2.0309,  0.9315],
         [ 0.0000,  0.0000,  0.0000],
         [-0.1655,  0.9897,  0.0635]]])

>>> # example of changing `pad` vector
>>> padding_idx = 0
>>> embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
>>> embedding.weight
Parameter containing:
tensor([[ 0.0000,  0.0000,  0.0000],
        [-0.7895, -0.7089, -0.0364],
        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
>>> with torch.no_grad():
...     embedding.weight[padding_idx] = torch.ones(3)
>>> embedding.weight
Parameter containing:
tensor([[ 1.0000,  1.0000,  1.0000],
        [-0.7895, -0.7089, -0.0364],
        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
```

# Tensor's methods

## Tensor.masked_fill_
[doc](https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_)

`Tensor.masked_fill_(mask, value)`

Fills elements of self tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.

Parameters
- `mask` (BoolTensor) – the boolean mask
- `value` (float) – the value to fill in with

## Tensor.view
[doc](https://pytorch.org/docs/stable/tensor_view.html)

PyTorch allows a tensor to be a View of an existing tensor. View tensor shares the same underlying data with its base tensor. Supporting View avoids explicit data copy, thus allows us to do fast and memory efficient reshaping, slicing and element-wise operations.

```python
>>> t = torch.rand(4, 4)
>>> b = t.view(2, 8)
>>> t.storage().data_ptr() == b.storage().data_ptr()  # `t` and `b` share the same underlying data.
True
# Modifying view tensor changes base tensor as well.
>>> b[0][0] = 3.14
>>> t[0][0]
tensor(3.14)
```

Typically a PyTorch op returns a new tensor as output, e.g. `add()`. But in case of view ops, outputs are views of input tensors to avoid unnecessary data copy.

**No data movement occurs when creating a view, view tensor just changes the way it interprets the same data.**

## Tensor.unsqueeze

[doc](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch.unsqueeze)

`torch.unsqueeze(input, dim) → Tensor`

Returns a new tensor with a dimension of size one inserted at the specified position.

Parameters
- `input` (Tensor) – the input tensor.
- `dim` (int) – the index at which to insert the singleton dimension

```python
>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```

## Tensor.transpose

[doc](https://pytorch.org/docs/stable/generated/torch.transpose.html#torch.transpose)

`torch.transpose(input, dim0, dim1) → Tensor`

Returns a tensor that is a transposed version of `input`. The given dimensions `dim0` and `dim1` are swapped.

Parameters
- `input` (Tensor) – the input tensor.
- `dim0` (int) – the first dimension to be transposed
- `dim1` (int) – the second dimension to be transposed

```python
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 1.0028, -0.9893,  0.5809],
        [-0.1669,  0.7299,  0.4942]])
>>> torch.transpose(x, 0, 1)
tensor([[ 1.0028, -0.1669],
        [-0.9893,  0.7299],
        [ 0.5809,  0.4942]])
```

## Tensor.mT

[doc](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.mT)

Returns a view of this tensor with the last two dimensions transposed.

`x.mT` is equivalent to `x.transpose(-2, -1)`.

# Save and load model
Based on `state_dict`

## Save
```python
torch.save(model.state_dict(), PATH)
```

## Load
```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
```

Remember the to specify the network mode:
```python
model.eval()
model.train()
```