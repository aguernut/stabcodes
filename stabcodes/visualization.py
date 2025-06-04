import sinter
import matplotlib.pyplot as plt
import datetime
import uuid


def unique_name(date=True):
    return (datetime.date.today() + "_") if date else "" + str(uuid.uuid4())


def dump_to_csv(code_stats, namefile):
    with open(namefile + ".csv", "w") as f:
        f.write(sinter._data._csv_out.CSV_HEADER + "\n")
        for stats in code_stats:
            f.write(stats.to_csv_line() + "\n")


def plot_error_rate(namefile, title="Title", xlabel="Xlabel", ylabel="Ylabel",
                    xlim=(0.9e-6, 5.1e-1), ylim=(1e-4, 1e-0), picture_output=True, j=None):
    stats = sinter.stats_from_csv_files(namefile + ".csv")

    fig = plt.figure()

    ax = fig.add_subplot()
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stats: stats.json_metadata['noise'],
        group_func=lambda stats: f"d={str(stats.json_metadata['d'])} ({stats.decoder})",
        # picture_output=picture_output,
        # j=j
    )

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.loglog()
    ax.set_title(title, fontdict={'weight': 'normal', 'size': 36})
    ax.set_xlabel(xlabel, fontdict={'weight': 'normal', 'size': 30})
    ax.set_ylabel(ylabel, fontdict={'weight': 'normal', 'size': 30})
    ax.grid(which='major')
    ax.grid(which='minor')
    ax.legend(fontsize=20)
    fig.set_dpi(120)
    fig.savefig(namefile + ".png")
