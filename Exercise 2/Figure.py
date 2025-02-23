import matplotlib.pyplot as plt

from os import makedirs
from typing import Tuple
from helpers import getOutputDir


class Figure:

    def __init__(self,
                 rows: int,
                 cols: int,
                 title: str,
                 filename: str,
                 figsize=(30, 20)):
        self.rows = rows
        self.cols = cols
        self.title = title
        self.filename = filename
        self.fig, self.axes = plt.subplots(nrows=self.rows,
                                           ncols=self.cols,
                                           figsize=figsize)
        self.fig.suptitle(self.title, fontsize=48)

    def _getAxis(self, row: int, col: int):
        if self.rows == 1 and self.cols == 1:
            return self.axes
        elif self.rows == 1 and self.cols > 1:
            return self.axes[col]
        elif self.cols == 1 and self.rows > 1:
            return self.axes[row]
        else:
            return self.axes[row, col]

    def addPlot(self,
                row: int,
                col: int,
                x,
                y,
                title: str | None = None,
                xLabel: str | None = None,
                yLabel: str | None = None,
                xLim: Tuple | None = None,
                yLim: Tuple | None = None,
                xticks: list[int] | None = None,
                yticks: list[int] | None = None,
                grid: bool = False,
                dotted: bool = False,
                emphasized: bool = False):
        ax = self._getAxis(row, col)
        ax.xaxis.set_tick_params(labelsize=24, width=4)
        ax.yaxis.set_tick_params(labelsize=24, width=4)
        ax.plot(x, y, "--" if dotted else "-", linewidth=3 if emphasized else 2)
        if title is not None:
            ax.set_title(title, fontsize=36)
        if xLabel is not None:
            ax.set_xlabel(xLabel, fontsize=24)
        if yLabel is not None:
            ax.set_ylabel(yLabel, fontsize=24)
        if xLim is not None:
            ax.set_xlim(xLim)
        if yLim is not None:
            ax.set_ylim(yLim)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if grid:
            ax.grid()

    def addSinglePlot(self,
                      x,
                      y,
                      title: str | None = None,
                      xLabel: str | None = None,
                      yLabel: str | None = None,
                      xLim: Tuple | None = None,
                      yLim: Tuple | None = None,
                      xticks: list[int] | None = None,
                      yticks: list[int] | None = None,
                      grid: bool = False):
        self.addPlot(0, 0, x, y, title, xLabel, yLabel, xLim, yLim, xticks,
                     yticks, grid)

    def addLegend(self, row: int, col: int, labels: list[str]):
        ax = self._getAxis(row, col)
        ax.legend(labels, prop={"size": 24})

    def savePlot(self):
        makedirs(f"{getOutputDir(2)}/figures", exist_ok=True)
        self.fig.savefig(f"{getOutputDir(2)}/figures/{self.filename}")
