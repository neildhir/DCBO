def assign_intervention_level(
    exploration_set: tuple,
    intervention_level,
    intervention_blanket: dict,
    target_temporal_index: int,  # exploration set
) -> None:
    """
    This function uses the assigned blanket to create an intervention blanket,
    making use of previously found optimal values for the manipulative and target variables(s).
    """

    assert isinstance(target_temporal_index, int)
    assert target_temporal_index > 0
    assert intervention_level is not None

    if len(exploration_set) == 1:
        # Intervention only happening on _one_ variable
        # Check to make sure we are not assigning a value to a node which already has one.
        assert intervention_blanket[exploration_set[0]][target_temporal_index] is None, (
            intervention_blanket,
            exploration_set,
            intervention_level,
            target_temporal_index,
        )
        intervention_blanket[exploration_set[0]][target_temporal_index] = float(intervention_level)

    else:
        # Intervention happening on _multiple_ variables
        for variable, lvl in zip(exploration_set, intervention_level):
            # Assign the intervention
            assert intervention_blanket[variable][target_temporal_index] is None, (
                intervention_blanket[variable][target_temporal_index],
                target_temporal_index,
            )
            intervention_blanket[variable][target_temporal_index] = float(lvl)


def assign_initial_intervention_level(
    exploration_set: tuple, intervention_level, intervention_blanket: dict, target_temporal_index: int,
) -> None:
    """
    This is the intervention assignment when we are at time 0 of the graph (for DCBO).
    At this point we do not yet have to consider the influence of the past optimal target value.
    Consequently we use this function for BO and CBO which are neither influences by the past.

    Parameters
    ----------
    intervention_blanket : [type], optional
        The intervention blanket which is commensurate with the topology of the graph
    target_temporal_index : str, optional
        The temporal index of the current target variable under consideration
    """

    assert isinstance(target_temporal_index, int)

    if len(exploration_set) == 1:
        # Assign the intervention
        intervention_blanket[exploration_set[0]][target_temporal_index] = float(intervention_level)

    else:
        # Intervention happening on _multiple_ variables
        for variable, lvl in zip(exploration_set, intervention_level):
            # Assign the intervention
            intervention_blanket[variable][target_temporal_index] = float(lvl)
