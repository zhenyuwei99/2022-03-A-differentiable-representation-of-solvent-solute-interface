# To Do list:
- [ ] Finding best performance network
    - Larger `num_heads` and smaller `dim_model`
- [ ] Implementing in calculation of binding free energy
    - [ ] The surface area approximation
    - [ ] The differentiable relative permittivity
- [x] Model serialization and save; Complete at 2022-04-01
- [x] Validating using multi-coordinates as the input of decoder; Complete at 2022-04-01
- [x] Implementation of weighted sampler; Complete at 2022-04-01
- [x] Regularization of transformer; Complete at 2022-04-01
- [x] Code for testing performance; Complete at 2022-04-02
- [x] Code for visualizing performance; Complete at 2022-04-02
- [x] Adding dropout layer; Complete at 2022-04-03
- [x] Validating the smallest applicable model size; Complete at 2022-04-05
    - [x] `dim_model`; Complete at 2022-04-05
    - [x] `num_heads`; Complete at 2022-04-05
    - [x] `num_layers`; Complete at 2022-04-05

# 2022-03-31
## Progress
- Finishing network configuration
- Starting the first trail on training on CPU

# 2022-04-01
## Solved questions
- Implementing weighted sampler using `collect_fn`
- Implementing regularization in optimizer initialization
- Validating the number of samples of decoder input does not influence the final result or the independence of each sample of decoder input.
- Implementing `save_model` and `load_model` function

## Progress
- Starting GPU training on the HPC

# 2022-04-02
## Solved questions
- Implementing code for testing the performance
- Implementing visualization code

# 2022-04-03
## Solved questions
- Adding dropout layer to `MultiHeadAttention` class:
    - Adding dropout before add and norm
    - Adding dropout after norm
    - **Prove to have bad performance**

# 2022-04-04
## Progress
- Dropout in embedding layer is useful **False**
- Random sample size is useful **False**
- Over-fitting is not a big issue
- Learning rate can change from 1e-3 to 1e-6

# 2022-04-05
## Progress
- Implementing training on cloud server
- **Batch size is crucial**
- Small model (**330703** parameters in total):
    ```python
    dim_model = 32
    dim_ffn = 256
    dim_k = dim_v = 32
    num_layers = 6
    num_heads = 4
    ```
    already have performance that good enough
- Learning rate modification is useful

## Solved questions
- Less layer model (**176335** parameters in total):
    ```python
    dim_model = 32
    dim_ffn = 256
    dim_k = dim_v = 32
    num_layers = 3
    num_heads = 4
    ```
    also has a good performance:
    - 98% accuracy for 0.5 deviation
    - 95% accuracy for 0.1 deviation
- Less head model (**151759** parameters in total):
    ```python
    dim_model = 32
    dim_ffn = 256
    dim_k = dim_v = 32
    num_layers = 3
    num_heads = 2
    ```
    **bad performance**
    - 90% accuracy for 0.5 deviation
    - 75% accuracy for 0.1 deviation
- Smaller dim model (**86207** parameters in total):
    ```python
    dim_model = 16
    dim_ffn = 256
    dim_k = dim_v = 32
    num_layers = 3
    num_heads = 4
    ```
    **bad performance**, decreasing slower than num head
    - 94% accuracy for 0.5 deviation
    - 85% accuracy for 0.1 deviation