import sinter
import matplotlib.pyplot as plt
import datetime
import uuid
import math
from typing import Iterable, Optional, Union

from stabcodes.stats_split import split_stats_for_observables


def unique_name(date: bool = True):
    """
    Returns a unique name with a uuid component and optionally today's date.
    
    Parameters
    ----------
    date: bool
        Whether the date should be included at the beginning of the unique name.
        
    Returns
    -------
    str:
        A statistically unique name.

    """
    return ((str(datetime.date.today()) + "_") if date else "") + str(uuid.uuid4())


def dump_to_csv(code_stats: Iterable[sinter.TaskStats],
                namefile: str,
                clean_after: Optional[str] = None):
    """
    Write a collection of :class:`sinter.TaskStats` to a csv file.
    
    Parameters
    ----------
    code_stats: Iterable[sinter.TaskStats]
        Collections of statistics to save to file.
    namefile: str
        Name of the file to write on.
    clean_after: str, optional
        Cutoff string from which to mangle decoder name. 
        Helps ensuring stats decoded with different instantiated decoder to be group together properly.

    """
    with open(namefile + ".csv", "w") as f:
        f.write(sinter._data._csv_out.CSV_HEADER + "\n") # type: ignore
        for stats in code_stats:
            if clean_after is not None and clean_after in stats.decoder:
                object.__setattr__(stats, "decoder", stats.decoder[:stats.decoder.index(clean_after)])
            f.write(stats.to_csv_line() + "\n")


def plot_error_rate(namefile: str,
                    title: str = "Title",
                    xlabel: str = "Xlabel",
                    ylabel: str = "Ylabel",
                    xlim: Optional[tuple[float, float]] = None,
                    ylim: Optional[tuple[float, float]] = None,
                    split: bool = False,
                    filt: Optional[Iterable[int]] = None):
    """
    Plots the statistics gathered in the file `namefile`.
    
    Parameters
    ----------
    namefile: str
        Name of the file containing the collected statistics.
    title: str
        Title of the figure.
    xlabel: str
        Label of the X axis.
    ylabel: str
        Label of the Y axis.
    xlim: tuple[float, float], optional
        Range of the X axis to display on all the subfigures. If left to None, Matplotlib will decide for each individual figures.
    ylim: tuple[float, float], optional
        Range of the Y axis to display on all the subfigures. If left to None, Matplotlib will decide for each individual figures.
    split: bool
        Whether to attempt to split the observables into several subfigures. Data must have been collected with the `count_observable_error_combos=True` option.
    filt: Iterable[int], optional
        Choose a subset of observables to display. Requires `split=True`.
    """

    stats = sinter.stats_from_csv_files(namefile + ".csv")

    if split:
        num_observables = len(next(iter(stats[0].custom_counts)).split("=")[1])
        stats = split_stats_for_observables(stats, num_observables)
    else:
        stats = [stats]
        num_observables = 1

    if filt is not None:
        if not split:
            raise ValueError("Only split observables can be filtered. Please use split=True option.")
        num_observables = len(filt)
        stats = [stats[f] for f in filt]

    nrows = math.floor(math.sqrt(num_observables))
    ncols = math.ceil(num_observables / math.floor(math.sqrt(num_observables)))

    fig, axes = plt.subplots(nrows, ncols, squeeze=False)

    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(wspace=0.3)

    for i in range(nrows):
        for j in range(ncols):

            ax = axes[i][j]

            sinter.plot_error_rate(
                ax=ax,
                stats=stats[i * ncols + j],
                x_func=lambda stats: stats.json_metadata['noise'],
                group_func=lambda stats: f"d={str(stats.json_metadata['d'])} ({stats.decoder})",
            )

            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.tick_params(axis='both', which='minor', labelsize=20)
            ax.loglog()
            ax.set_xlabel(xlabel, fontdict={'weight': 'normal', 'size': 30})
            ax.set_ylabel(ylabel, fontdict={'weight': 'normal', 'size': 30})
            ax.grid(which='major')
            ax.grid(which='minor')
            ax.legend()
            if split:
                ax.set_title(f"Observable {filt[i * ncols + j] if filt is not None else i * ncols + j}")

    fig.suptitle(title, fontsize=36)
    fig.savefig(namefile + ".png")
