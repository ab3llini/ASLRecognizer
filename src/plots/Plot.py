import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.colors as cols


"""Plots a two lines graph.
@:param x is a list of the values on the x-axis for both lines.
@:param y1 is a list with the corresponding y-values for one line
@:param y2 is a list with the corresponding y-values for the second line
@:param l1 is the label associated to y1
@:param l2 is the label associated to y2
@:param title is the title of the graph"""
def line2(x, y1, y2, l1, l2, title, xlab, ylab):
    sb.lineplot(x=x, y=y1, color="red", legend='full', label=l1)
    ax = sb.lineplot(x=x, y=y2, color="blue", legend='full', label=l2)
    ax.set(xlabel=xlab, ylabel=ylab)
    plt.title(title)
    plt.show()


"""Plots a 1-line graph.
@:param x is a list of the values on the x-axis for both lines.
@:param y1 is a list with the corresponding y-values
@:param l1 is the label associated to y1
@:param title is the title of the graph"""
def line(x, y1, l1, title):
    ax = sb.lineplot(x=x, y=y1, color="blue", legend='full', label=l1)
    ax.set(xlabel="#training samples", ylabel="accuracy")
    plt.title(title)
    plt.show()


def meshgrid(x, y, z, cmap, ticks):
    plt.pcolor(x, y, z, cmap=cmap, alpha=0.2)
    plt.colorbar(ticks=ticks)
    plt.clim(min(ticks), max(ticks))


def scatter(x, y, classes, colors, annotate=False):
    ax = sb.scatterplot(x, y, marker="o", hue=classes, legend=False, palette=colors, edgecolor="black")
    plt.rcParams["lines.markeredgewidth"] = 4
    if annotate:
        for i, txt in enumerate(classes):
            ax.annotate(txt, (x[i], y[i]))


def show():
    plt.show()


def colormap():
    return cols.ListedColormap(['yellow', 'blue', 'red']), [1, 2, 3]



