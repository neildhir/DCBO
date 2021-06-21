# Dynamic Causal Continuous and Categorical Bayesian Optimisation (DCoCaBO)

## Notes

- [disrete to discrete] Discrete variable transitions will happen through learned transition matrices (as used in HMMs) (if unknown, otherwise use the deterministic matrix)

- [discrete to continuous] The discrete variable will index a particular domain of the continuous part

## Major refactorings to do ('TODO')

- Parallelise sampling functions so that they are not sequential, that is far too slow

- Implement KDE for proper handling of exogeneous noise variables, as done in CEO

- Consider start using Julia/Fortran for heavy matrix operations e.g. for acquisition function evaluation

- For all classes, write optimal parameters with **kwargs instead, it is currently far too messy
