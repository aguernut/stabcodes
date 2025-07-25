import sinter
import matplotlib.pyplot as plt
import datetime
import uuid
import math

from stabcodes.stats_split import split_stats_for_observables


def unique_name(date=True):
    return ((str(datetime.date.today()) + "_") if date else "") + str(uuid.uuid4())


def dump_to_csv(code_stats, namefile, clean_after=None):
    with open(namefile + ".csv", "w") as f:
        f.write(sinter._data._csv_out.CSV_HEADER + "\n")
        for stats in code_stats:
            if clean_after is not None:
                object.__setattr__(stats, "decoder", stats.decoder[:stats.decoder.index(clean_after)])
            f.write(stats.to_csv_line() + "\n")


def plot_error_rate(namefile, title="Title", xlabel="Xlabel", ylabel="Ylabel",
                    xlim=(0.9e-6, 5.1e-1), ylim=(1e-4, 1e-0), split=False, filt=None):

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

            ax.set_ylim(*ylim)
            ax.set_xlim(*xlim)
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
