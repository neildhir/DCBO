# Dynamic Causal Bayesian Optimisation

## Major refactorings to do ('TODO')

- Write tutorials

- Check license

- Write proper main README file

- Parallelise sampling functions so that they are not sequential, that is far too slow

- Implement KDE for proper handling of exogeneous noise variables, as done in CEO

- Consider start using Julia/Fortran for heavy matrix operations e.g. for acquisition function evaluation

- For all classes, write optimal parameters with **kwargs instead, it is currently far too messy
