"""
VISUALIZATION FUNCTIONS
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class CircularGraph:
    """
        Interactive circular graph, inspired by https://www.mathworks.com/matlabcentral/fileexchange/48576-circulargraph
    """

    def __init__(self, adjacency_matrix, labels, label_colors=None, node_colors=None, connection_colors=None,
                 fig_size=10, label_padding=10, visible=True, show_arrows=True, highlighted_labels=None,
                 max_linewidth=3, hide_buttons=False):
        """

        Parameters
        ----------
        adjacency_matrix : numpy n x n array
            Adjacency matrix to plot
        labels: list
            List of labels for each node
        label_colors : list
            List of colors for each text label
        node_colors : list
            List of colors for each node
        connection_colors: list
            Listo of color for each node's connections
        fig_size: tuple
            Figure size in (x,y) format
        label_padding: int
            Empty space pad size for labels
        visible: bool
            Show / Hide connections when plotting for the first time. Is ignored id "highlighted_labels" is not None
        show_arrows: bool
            Plot arrows instead of lines (not fully implemented yet)
        highlighted_labels: list
            List of node labels whose connections will be visible when plotting for the first time
        max_linewidth: float
            Maximum linewidth to which the adjacency matrix is normalized.
        hide_buttons:  bool
        If True, Show all and Hide all buttons will not be displayed in circular plot.

        """

        self.adjacency_matrix = adjacency_matrix
        self.labels = labels
        self.fig_size = fig_size
        self.nodes = []
        self.label_padding = label_padding
        self.visible = visible
        self.show_arrows = show_arrows
        self.highlighted_labels = highlighted_labels
        self.max_linewidth = max_linewidth

        if label_colors is None:
            self.label_colors = [(0, 0, 0)] * len(labels)
        else:
            self.label_colors = label_colors
        if node_colors is None:
            self.node_colors = [(0, 0, 0)] * len(labels)
        else:
            self.node_colors = node_colors
        if connection_colors is None:
            self.connection_colors = [(0, 0, 0)] * len(labels)
        else:
            self.connection_colors = connection_colors

        # self.t = - np.pi / 2 + np.linspace(np.pi , -np.pi, len(self.adjacency_matrix) + 1)
        # the line above should be ok, however, a jitter is needed to avoid drawing arcs through the outside of the circle
        jitter = 0.001 / (len(self.adjacency_matrix) + 1)
        self.t = - np.pi / 2 + np.linspace(np.pi,
                                           -np.pi + jitter,
                                           len(self.adjacency_matrix) + 1)

        # draw figure
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches([self.fig_size, self.fig_size])

        # draw nodes
        self.draw_nodes()
        for label in self.labels:
            self.draw_connections(label)

        # highlight some labels if necessary
        if self.highlighted_labels is not None:
            self.highlight_labels(self.highlighted_labels)

        self.fig.canvas.mpl_connect('pick_event', self.onpick)

        # draw buttons
        if hide_buttons is False:
            ax_showall = plt.axes([0.80, 0.08, 0.1, 0.02])
            ax_hideall = plt.axes([0.80, 0.05, 0.1, 0.02])
            self.bshowall = Button(ax_showall, 'Show All')
            self.bshowall.on_clicked(self.showall)
            self.bhideall = Button(ax_hideall, 'Hide All')
            self.bhideall.on_clicked(self.hideall)

    class Node:
        def __init__(self, x, y, label, label_colors, node_colors, marker, label_padding, connections,
                     connection_colors, visible):
            self.x = x
            self.y = y
            self.label = label
            self.label_colors = label_colors
            self.node_colors = node_colors
            self.marker = marker
            self.label_padding = label_padding
            self.connections = connections
            self.connection_colors = connection_colors
            self.visible = visible
            self.label_offset = 1.1
            self.arcs = []
            self.arrows = []
            self.marker_handle = None

            # draw node now
            self.draw_node()

        def draw_node(self):
            x = self.x
            y = self.y
            if self.visible:
                initial_edgecolor = 'black'
            else:
                initial_edgecolor = 'white'
            self.marker_handle = plt.scatter(x, y, c=self.node_colors, marker=self.marker, picker=1,
                                             edgecolors=initial_edgecolor, zorder=100000)
            t = np.arctan2(y, x)
            if np.abs(t) > (np.pi / 2):
                plt.text(x=x * self.label_offset, y=y * self.label_offset, s=self.label.ljust(self.label_padding),
                         rotation=(180 * (t / np.pi + 1)),
                         horizontalalignment='center', verticalalignment='center',
                         color=self.label_colors)
            else:
                plt.text(x=x * self.label_offset, y=y * self.label_offset, s=self.label.rjust(self.label_padding),
                         rotation=(t * 180 / np.pi),
                         horizontalalignment='center', verticalalignment='center',
                         color=self.label_colors)
            plt.gcf().axes[0].axis('off')

    def draw_nodes(self):
        for n in np.arange(0, len(self.adjacency_matrix)):
            new_node = self.Node(x=np.cos(self.t[n]), y=np.sin(self.t[n]), label=self.labels[n], label_padding=self.label_padding,
                                 marker='o',
                                 label_colors=self.label_colors[n], node_colors=self.node_colors[n],
                                 connection_colors=self.connection_colors[n], connections=self.adjacency_matrix[n, :],
                                 visible=self.visible)
            self.nodes.append(new_node)

    def draw_connections(self, node_label):
        node_number = self.labels.index(node_label)
        node_connection_color = self.connection_colors[node_number]
        locs = np.array(np.where(self.adjacency_matrix > 0))  # non zero connections
        locs = locs[:, np.where(locs[0] == node_number)[0]]  # keep connections of current node

        for i in np.arange(0, locs.shape[1]):
            if len(np.unique(locs[:, i])) > 1:
                u = np.array([np.cos(self.t[locs[:, i][0]]), np.sin(self.t[locs[:, i][0]])])
                v = np.array([np.cos(self.t[locs[:, i][1]]), np.sin(self.t[locs[:, i][1]])])
                if (np.abs(locs[:, i][0] - locs[:, i][1]) - len(self.labels) / 2) == 0:
                    # points are diametric, draw a straight line
                    new_arc = plt.plot([u[0], v[0]], [u[1], v[1]], color=node_connection_color, visible=self.visible,
                                       linewidth=self.nodes[node_number].connections[
                                                     locs[:, i][1]] * self.max_linewidth)
                    self.nodes[node_number].arcs.append(new_arc[0])
                else:
                    # points are nor diametric, draw an arc
                    x0 = -(u[1] - v[1]) / (u[0] * v[1] - u[1] * v[0])
                    y0 = (u[0] - v[0]) / (u[0] * v[1] - u[1] * v[0])
                    r = np.sqrt(x0 ** 2 + y0 ** 2 - 1)
                    thetalim1 = np.arctan2(u[1] - y0, u[0] - x0)
                    thetalim2 = np.arctan2(v[1] - y0, v[0] - x0)
                    if u[0] >= 0 and v[0] >= 0:
                        # ensure the arc is within the unit disk
                        theta = np.concatenate(
                            [np.linspace(max(thetalim1, thetalim2), np.pi, num=50),
                             np.linspace(-np.pi, min(thetalim1, thetalim2), num=50),
                             ]
                        )
                    else:
                        theta = np.linspace(thetalim1, thetalim2, num=100)
                    x = r * np.cos(theta) + x0
                    y = r * np.sin(theta) + y0
                    new_arc = plt.plot(x, y, color=node_connection_color, visible=self.visible,
                                       linewidth=self.nodes[node_number].connections[
                                                     locs[:, i][1]] * self.max_linewidth)
                    # new_arrow = plt.arrow(x[-4], y[-4], x[-3] - x[-4], y[-3] - y[-4], color=node_connection_color, visible=self.visible,
                    #           head_width=0.025, head_length=0.02)
                    # self.add_arrow(new_arc[0])
                    self.nodes[node_number].arcs.append(new_arc[0])
                    # self.nodes[node_number].arrows.append(new_arrow)

    def toggle_node(self, node_label):
        node_number = self.labels.index(node_label)
        self.nodes[node_number].visible = not self.nodes[node_number].visible
        new_edge_color = tuple(
            np.abs(np.array(self.nodes[node_number].marker_handle.get_edgecolor()) - np.array([1, 1, 1, 0])))
        self.nodes[node_number].marker_handle.set_edgecolor(new_edge_color)
        # for arc,arrow in zip(c.nodes[node_number].arcs, c.nodes[node_number].arrows):
        #     arc.set_visible(c.nodes[node_number].visible)
        #     arrow.set_visible(c.nodes[node_number].visible)
        for arc in self.nodes[node_number].arcs:
            arc.set_visible(self.nodes[node_number].visible)
        plt.draw()

    def highlight_labels(self, labels=None):
        if labels is None:
            labels = self.highlighted_labels
        if labels is not None:
            for node_label in self.labels:
                node_number = self.labels.index(node_label)
                if node_label in labels:
                    self.nodes[node_number].visible = True
                    self.nodes[node_number].marker_handle.set_edgecolor('black')  # black
                else:
                    self.nodes[node_number].visible = False
                    self.nodes[node_number].marker_handle.set_edgecolor('white')  # white
                for arc in self.nodes[node_number].arcs:
                    arc.set_visible(self.nodes[node_number].visible)
                    plt.draw()

    def onpick(self, event):
        for n in self.nodes:
            if n.marker_handle == event.artist:
                label = n.label
        self.toggle_node(label)

    def showall(self, event):
        for node_label in self.labels:
            node_number = self.labels.index(node_label)
            self.nodes[node_number].visible = True
            self.nodes[node_number].marker_handle.set_edgecolor('black')
            for arc in self.nodes[node_number].arcs:
                arc.set_visible(self.nodes[node_number].visible)
        plt.draw()

    def hideall(self, event):
        for node_label in self.labels:
            node_number = self.labels.index(node_label)
            self.nodes[node_number].visible = False
            self.nodes[node_number].marker_handle.set_edgecolor('white')
            for arc in self.nodes[node_number].arcs:
                arc.set_visible(self.nodes[node_number].visible)
        plt.draw()


if __name__ == "__main__":
    nnodes = 50
    adjacency_matrix = np.random.rand(nnodes, nnodes)
    adjacency_matrix[np.where(adjacency_matrix < 0.75)] = 0
    labels = [str(i).rjust(5, '0') for i in np.arange(1, nnodes + 1)]
    colors = plt.cm.rainbow(np.linspace(0, 1, nnodes))
    c = CircularGraph(adjacency_matrix, labels, connection_colors=colors, node_colors=colors, visible=True)
