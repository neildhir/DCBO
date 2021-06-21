import networkx as nx
from itertools import combinations, chain
from collections import Iterable


class DoCalculus:
    def __init__(self, user_DAG):

        self.dag = user_DAG  # NX graph provided by user
        self.observed_variables = frozenset(self.dag.to_undirected())
        self.set_nodes = frozenset()
        assert nx.is_directed_acyclic_graph(self.dag)
        for set_node in self.set_nodes:
            # set nodes cannot have parents
            assert not nx.ancestors(self.dag, set_node)
        self.graph = self.dag.to_undirected()

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.observed_variables)))
        return "{classname}({vars})".format(classname=self.__class__.__name__, vars=variables)

    def get_distribution(self):
        """
        Returns a string representing the factorized distribution implied by
        the CGM.
        """
        products = []
        for node in nx.topological_sort(self.dag):
            if node in self.set_nodes:
                continue

            parents = list(self.dag.predecessors(node))
            if not parents:
                p = "P({})".format(node)
            else:
                parents = ["do({})".format(n) if n in self.set_nodes else str(n) for n in parents]
                p = "P({}|{})".format(node, ",".join(parents))
            products.append(p)
        return "".join(products)

    def _check_d_separation(self, path, zs=None):
        """
        Check if a path is d-separated by set of variables zs.
        """
        zs = self._variable_or_iterable_to_set(zs)

        if len(path) < 3:
            return False

        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            structure = self._classify_three_structure(a, b, c)

            if structure in ("chain", "fork") and b in zs:
                return True

            if structure == "collider":
                descendants = nx.descendants(self.dag, b) | {b}
                if not descendants & set(zs):
                    return True

        return False

    def _classify_three_structure(self, a, b, c):
        """
        Classify three structure as a chain, fork or collider.
        """
        if self.dag.has_edge(a, b) and self.dag.has_edge(b, c):
            return "chain"

        if self.dag.has_edge(c, b) and self.dag.has_edge(b, a):
            return "chain"

        if self.dag.has_edge(a, b) and self.dag.has_edge(c, b):
            return "collider"

        if self.dag.has_edge(b, a) and self.dag.has_edge(b, c):
            return "fork"

        raise ValueError("Unsure how to classify ({},{},{})".format(a, b, c))

    def is_d_separated(self, x, y, zs=None):
        """
        Is x d-separated from y, conditioned on zs?
        """
        zs = self._variable_or_iterable_to_set(zs)
        assert x in self.observed_variables
        assert y in self.observed_variables
        assert all([z in self.observed_variables for z in zs])

        paths = nx.all_simple_paths(self.graph, x, y)
        return all(self._check_d_separation(path, zs) for path in paths)

    def get_all_independence_relationships(self):
        """
        Returns a list of all pairwise conditional independence relationships
        implied by the graph structure.
        """
        conditional_independences = []
        for x, y in combinations(self.observed_variables, 2):
            remaining_variables = set(self.observed_variables) - {x, y}
            for cardinality in range(len(remaining_variables) + 1):
                for z in combinations(remaining_variables, cardinality):
                    if self.is_d_separated(x, y, frozenset(z)):
                        conditional_independences.append((x, y, set(z)))

        return conditional_independences

    def get_all_backdoor_paths(self, x, y):
        """
        Get all backdoor paths between x and y
        """
        return [
            path
            for path in nx.all_simple_paths(self.graph, x, y)
            if len(path) > 2 and path[1] in self.dag.predecessors(x)
        ]

    def is_valid_backdoor_adjustment_set(self, x, y, z):
        """
        Test whether z is a valid backdoor adjustment set for
        estimating the causal impact of x on y via the backdoor
        adjustment formula:
        P(y|do(x)) = \sum_{z}P(y|x,z)P(z)
        Arguments
        ---------
        x: str or set
            Intervention Variable(s)
        y: str
            Target Variable
        z: str or set[str]
            Adjustment variables
        Returns
        -------
        is_valid_adjustment_set: bool
        """
        z = self._variable_or_iterable_to_set(z)

        assert y in self.observed_variables
        assert y not in z
        if isinstance(x, str):
            assert x in self.observed_variables
            assert x not in z
            if any([zz in nx.descendants(self.dag, x) for zz in z]):
                return False

            backdoor_paths = self.get_all_backdoor_paths(x, y)

        else:
            #  Multiple interventions
            assert isinstance(x, set)
            assert x <= self.observed_variables
            assert any([xx not in z for xx in x])
            backdoor_paths = []
            for x_mem in x:
                if any([zz in nx.descendants(self.dag, x_mem) for zz in z]):
                    return False
                backdoor_paths.append(self.get_all_backdoor_paths(x_mem, y))

        unblocked_backdoor_paths = [path for path in backdoor_paths if not self._check_d_separation(path, z)]

        if unblocked_backdoor_paths:
            return False

        return True

    def get_all_backdoor_adjustment_sets(self, x, y):
        """
        Get all sets of variables which are valid adjustment sets for
        estimating the causal impact of x on y via the back door
        adjustment formula:

        P(y|do(x)) = \sum_{z}P(y|x,z)P(z)

        Note that the empty set can be a valid adjustment set for some CGMs,
        in this case frozenset(frozenset(), ...) is returned. This is different
        from the case where there are no valid adjustment sets where the
        empty set is returned.
        Arguments
        ---------
        x: str or set
            Intervention Variable(s)
        y: str
            Target Variable
        Returns
        -------
        condition set: frozenset[frozenset[variables]]
        """
        assert y in self.observed_variables
        if isinstance(x, str):
            assert x in self.observed_variables
            possible_adjustment_variables = set(self.observed_variables) - {x} - {y} - set(nx.descendants(self.dag, x))
        else:
            #  Multiple interventions
            # assert isinstance(x, set)
            if ~isinstance(x, set):
                x = set(x)
            assert x <= self.observed_variables
            desc = []
            for x_mem in x:
                desc.extend(nx.descendants(self.dag, x_mem))
            possible_adjustment_variables = set(self.observed_variables) - x - {y} - set(desc)

        valid_adjustment_sets = frozenset(
            [
                frozenset(s)
                for s in self._powerset(possible_adjustment_variables)
                if self.is_valid_backdoor_adjustment_set(x, y, s)
            ]
        )

        if not valid_adjustment_sets:
            print("The empty set is _the_ valid adjustment set.")
            return valid_adjustment_sets
        else:
            return valid_adjustment_sets

    def is_valid_frontdoor_adjustment_set(self, x, y, z):
        """
        Test whether z is a valid frontdoor adjustment set for
        estimating the causal impact of x on y via the frontdoor
        adjustment formula:
        P(y|do(x)) = \sum_{z}P(z|x)\sum_{x'}P(y|x',z)P(x')
        Arguments
        ---------
        x: str
            Intervention Variable
        y: str
            Target Variable
        z: set
            Adjustment variables
        Returns
        -------
        is_valid_adjustment_set: bool
        """
        z = self._variable_or_iterable_to_set(z)

        # 1. does z block all directed paths from x to y?
        unblocked_directed_paths = [
            path for path in nx.all_simple_paths(self.dag, x, y) if not any(zz in path for zz in z)
        ]

        if unblocked_directed_paths:
            return False

        # 2. no unblocked backdoor paths between x and z
        unblocked_backdoor_paths_x_z = [
            path
            for zz in z
            for path in self.get_all_backdoor_paths(x, zz)
            if not self._check_d_separation(path, z - {zz})
        ]

        if unblocked_backdoor_paths_x_z:
            return False

        # 3. x is a valid backdoor adjustment set for z
        if not all(self.is_valid_backdoor_adjustment_set(zz, y, x) for zz in z):
            return False

        return True

    def get_all_frontdoor_adjustment_sets(self, x, y):
        """
        Get all sets of variables which are valid frontdoor adjustment sets for
        estimating the causal impact of x on y via the frontdoor adjustment
        formula:
        P(y|do(x)) = \sum_{z}P(z|x)\sum_{x'}P(y|x',z)P(x')
        Note that the empty set can be a valid adjustment set for some CGMs,
        in this case frozenset(frozenset(), ...) is returned. This is different
        from the case where there are no valid adjustment sets where the
        empty set is returned.
        Arguments
        ---------
        x: str
            Intervention Variable
        y: str
            Target Variable
        Returns
        -------
        condition set: frozenset[frozenset[variables]]
        """
        assert x in self.observed_variables
        assert y in self.observed_variables

        possible_adjustment_variables = set(self.observed_variables) - {x} - {y}

        valid_adjustment_sets = frozenset(
            [
                frozenset(s)
                for s in self._powerset(possible_adjustment_variables)
                if self.is_valid_frontdoor_adjustment_set(x, y, s)
            ]
        )

        return valid_adjustment_sets

    def _variable_or_iterable_to_set(self, x):
        """
        Convert variable or iterable x to a frozenset.
        If x is None, returns the empty set.
        Arguments
        ---------
        x: None, str or Iterable[str]
        Returns
        -------
        x: frozenset[str]
        """
        if x is None:
            return frozenset([])

        if isinstance(x, str):
            return frozenset([x])

        if not isinstance(x, Iterable) or not all(isinstance(xx, str) for xx in x):
            raise ValueError("{} is expected to be either a string or an iterable of strings".format(x))

        return frozenset(x)

    def _powerset(self, iterable):
        """
        https://docs.python.org/3/library/itertools.html#recipes
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
