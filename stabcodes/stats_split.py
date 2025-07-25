################################################################################
# This is a shortened version of the file split.py rendered available here:    #
# https://github.com/tqec/tqec/blob/main/src/tqec/simulation/split.py          #
# The original version of this file was copied from commit:                    #
# https://github.com/tqec/tqec/commit/21423afa8cd6790b458c6ab11d7fdb47d062d8ef #
# that was licensed under Apache 2.0 License on 2025/07/25.                    #
# A copy of this license can be found here:  http://www.apache.org/licenses/   #
################################################################################


"""Split the statistics for multiple observables.

The use of this method was suggested here: https://quantumcomputing.stackexchange.com/a/43939/22557.
"""

import collections
from collections.abc import Mapping

import sinter


def split_stats_for_observables(
    stats: list[sinter.TaskStats],
    num_observables: int,
) -> list[list[sinter.TaskStats]]:
    """Split the statistics for each individual observable.

    This function should only be used when specifying ``count_observable_error_combos=True`` to
    ``sinter`` functions.

    Args:
        stats: The statistics for different observable error combinations.

    Returns:
        A list of statistics for each individual observable.

    """
    from sinter._data import ExistingData  # type: ignore

    # Combine the stats for each task
    data = ExistingData()
    for s in stats:
        data.add_sample(s)
    combined_stats = list(data.data.values())

    # For each task, split the stats by observable
    stats_by_observables: list[list[sinter.TaskStats]] = [[] for _ in range(num_observables)]
    for task_stats in combined_stats:
        split_counts = split_counts_for_observables(task_stats.custom_counts, num_observables)
        for obs_idx, count in enumerate(split_counts):
            stats_by_observables[obs_idx].append(
                task_stats.with_edits(
                    errors=count,
                    custom_counts=collections.Counter(
                        {
                            k: v
                            for k, v in task_stats.custom_counts.items()
                            if not k.startswith("obs_mistake_mask=")
                        }
                    ),
                )
            )
    return stats_by_observables


def split_counts_for_observables(counts: Mapping[str, int], num_observables: int) -> list[int]:
    """Split the error counts for each individual observable.

    This function should only be used when specifying ``count_observable_error_combos=True`` to
    ``sinter`` functions.

    Args:
        counts: The error counts for different observable error combinations.
        num_observables: The number of observables.

    Returns:
        A list of error counts for each individual observable.

    """
    split_counts: list[int] = [0] * num_observables
    for key, count in counts.items():
        if not key.startswith("obs_mistake_mask="):
            continue
        comb = key.split("=")[1]
        assert num_observables == len(comb)
        for i in range(num_observables):
            if comb[i] == "E":
                split_counts[i] += count
    return split_counts
