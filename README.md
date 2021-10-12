# Dynamic Causal Bayesian Optimisation

This is an official Python implementation of '[Dynamic Causal Bayesian Optimization](https://nips.cc/)' presented at Neurips 2021.

## Authors

- [Virginia Aglietti](https://uk.linkedin.com/in/virginia-aglietti-a80321a4)
- [Neil Dhir](https://neildhir.github.io/)
- [Javier Gonzalez](https://javiergonzalezh.github.io/)
- [Theodoros Damoulas](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/damoulas/)

## TODO

- Write tutorials

- Check license

- Write proper main README file

- Parallelise sampling functions so that they are not sequential, that is far too slow

- Implement KDE for proper handling of exogeneous noise variables, as done in CEO

- Consider start using Julia/Fortran for heavy matrix operations e.g. for acquisition function evaluation

- For all classes, write optimal parameters with **kwargs instead, it is currently far too messy

## Installation

Test

### Requirements

- python
- numpy
- scipy
- networkx

## Demo and tutorials

Test

### Data

The data used for the real experiments in section four of the paper can be found here:

- hi
- hi

## Citation

Please cite the NeurIPS paper if you use DCBO in your work:

```[bibtex]
@inproceedings{DCBO,
 author = {Aglietti, Virginia and Dhir, Neil and Gonz\'{a}lez, Javier and Damoulas, Theodoros},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Dynamic Causal Bayesian Optimization},
 volume = {35},
 year = {2021}
}
```

## License

**placeholder**

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.