# To Do list:
- [ ] Code for testing performance
- [ ] Adding dropout layer
- [x] Model serialization and save; Complete at 2022-04-01
- [x] Validating using multi-coordinates as the input of decoder; Complete at 2022-04-01
- [x] Implementation of weighted sampler; Complete at 2022-04-01
- [x] Regularization of transformer; Complete at 2022-04-01

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