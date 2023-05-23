""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`
"""
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits import mplot3d
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.lines as mlines
from src.helper.commons import *
import src.functions.mathlib as mlib


# --------------- This is essential to make the back-end work for Qt and Mac OSx!
# matplotlib.use('TkAgg')

# useful links
# list of colors https://matplotlib.org/stable/gallery/color/named_colors.html
# highlight https://coderedirect.com/questions/253759/add-custom-border-to-certain-cells-in-a-matplotlib-seaborn-plot
# Seaborn plot/color tricks https://towardsdatascience.com/heatmap-basics-with-pythons-seaborn-fb92ea280a6c

def use_style(style_name):
    """
    Instant stylization of the plot (e.g. black background).

    Here is the `reference <https://matplotlib.org/3.5.0/tutorials/introductory/customizing.html#customizing-with-style-sheets>`_.

    :param str style_name: :code:`['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh',
        'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright',
        'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep',
        'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk',
        'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']`
    """
    plt.style.use(style_name)


def plot_1d_array(array, title, plt_show, no_ticks, color=None, interpolate=True, interp_res=10, symmetric_x_axis=True,
                  ax=None, label=None, line_style=None, linewidth=1, swap_axes=False, frame_on=True,
                  limits_x_tight=True, limits_y_tight=False):
    """
    :param 1darray array:
    :param str title:
    :param bool plt_show: force plot if :code:`True`, else delay the plot show
    :param bool no_ticks:
    :param str color: if :code:`None` plot style picks the best one. 'firebrick' is a nice choice.
    :param bool interpolate:
    :param int interp_res: :code:`1` no interpolation, :code:`1>` increased resolution
    :param bool symmetric_x_axis: if :code:`True` add points equally to each side of the array center
    :param `~.axes.Axes` ax:
    :param str label: line label
    :param str line_style:
    :param int linewidth:
    :param bool swap_axes:
    :param bool frame_on: border control
    :param bool limits_x_tight: no horizontal space in the axis
    :param bool limits_y_tight: no vertical space in the axis
    :return: axis
    :rtype: `Axes`:
    """
    if ax is None:
        plt.figure()
        ax = plt.axes()

    if np.size(array) == 0:
        return ax

    if interpolate:
        index_new, val_new = mlib.interp_1darray(arr=array, resolution=interp_res, symmetric_x_axis=symmetric_x_axis)
        if swap_axes:
            x_vals = val_new
            y_vals = index_new
        else:
            x_vals = index_new
            y_vals = val_new

        if line_style is not None:
            ax.plot(x_vals, y_vals, line_style, linewidth=linewidth, color=color)
        else:
            ax.plot(x_vals, y_vals, linewidth=linewidth, color=color)

    else:
        if line_style is not None:
            ax.plot(array, line_style, label=label, linewidth=linewidth)
        else:
            ax.plot(array, label=label, linewidth=linewidth)

    if no_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title)

    # tight plot
    if limits_x_tight:
        ax.autoscale(enable=True, axis='x', tight=True)
    if limits_y_tight:
        ax.autoscale(enable=True, axis='y', tight=True)

    # box border on
    if frame_on:
        ax.set(frame_on=True)

    if plt_show:
        plt.show()

    return ax


def plot_filters(opts_plots, safe_rank, filters, specs_info):
    """Plot 1d Poisson filters.

    :param OptionsPlots opts_plots:
    :param int safe_rank: cumulative rank, i.e. maximum rank to be included; must be safe; \
        "safe" means a rank that does not exceed the actual rank of the kernel
    :param 1darray filters:
    :param str specs_info:
    """
    # grid layout to plot multi modes
    _layout = compute_layout(columns=4, num_elements=safe_rank)

    opts_subplots_filters = OptionsSubPlots(layout=_layout)

    all_filters = []
    for r in range(1, safe_rank + 1):
        all_filters.append(filters[r - 1])  # assuming horizontal and vertical filters are identical

    _subplots_filters = []
    init_subplots(sub_plots=_subplots_filters, opts_subplots=opts_subplots_filters)

    for i in range(safe_rank):
        # filter
        filter_1d = all_filters[i]
        # compute a compact range of +- value where we take tha largest of min and max values in the matrix
        max_val = max(np.abs(np.max(filter_1d)), np.abs(np.min(filter_1d)))
        add_subplot(M=filter_1d,
                    title=f'Rank {i + 1}, $\pm${max_val:.0e}',
                    sub_plots=_subplots_filters, opts_subplots=opts_subplots_filters)

    # ranked filters
    plot_1d_array_grid(G=_subplots_filters, title=specs_info,
                       opts_general=opts_plots, opts_subplots=opts_subplots_filters)


def plot_filters_3d_advanced(processed_filters, axes_1d, axes_2d, fig, visualize_truncation, truncation_method,
                             truncation_value, filter_titles, swap_axes, opts_plots):
    """Plot beautified filters with truncation bars.

    :param ndarray processed_filters:
    :param Axes axes_1d: axes to draw 1d views
    :param Axes axes_2d: axes to draw 2d views
    :param matplotlib.figure.Figure fig:
    :param bool visualize_truncation:
    :param TruncationMode truncation_method: for now only works with :code:`FIXED_THRESHOLD`
    :param float truncation_value:
    :param str filter_titles:
    :param bool swap_axes: swaps x-axis and y-axis (vertical or horizontal)
    :param OptionsPlots opts_plots:
    :return:
    """
    assert len(axes_1d) == len(axes_2d)

    for r in range(len(processed_filters)):
        this_filter = processed_filters[r]
        plot_scalar_field_unpack_opts(M=this_filter, ax_img=axes_1d[r], fig=fig, vrange_im=None, opts=opts_plots)
        plot_1d_array(array=this_filter, ax=axes_2d[r], title=filter_titles[r], plt_show=False, no_ticks=True,
                      interpolate=True, interp_res=10, symmetric_x_axis=True, label=None, line_style=None, linewidth=1,
                      swap_axes=swap_axes, limits_x_tight=True, limits_y_tight=True, frame_on=False)

        # adjusting the thin filter plots position
        pos_1d = axes_1d[r].get_position()
        pos_1d = [pos_1d.x0 + 0.02, pos_1d.y0, pos_1d.width, pos_1d.height]
        axes_1d[r].set_position(pos_1d, which='both')

        # truncation threshold bar
        if visualize_truncation:
            draw_truncation_visuals(truncation_method=truncation_method, truncation_value=truncation_value,
                                    filter_1d=this_filter, ax=axes_2d[r])

    return plt


def draw_truncation_visuals(truncation_method, truncation_value, filter_1d, ax):
    """Add visuals to an existing filter plot displaying adaptive truncation (single filter)

    :param TruncationMode truncation_method: for now only works with :code:`FIXED_THRESHOLD`
    :param float truncation_value:
    :param 1darray filter_1d:
    :param Axes ax:
    """
    assert truncation_method == TruncationMode.FIXED_THRESHOLD

    # bar position: the axis has a unit of integer indices. 0 Is at the center.
    filter_half_size = int(filter_1d.shape[0] / 2)
    cut, cut_indices = mlib.get_truncate_indices(arr=filter_1d, cut_off=truncation_value)
    if cut is None or cut_indices is None:  # everything is truncated, nothing is left
        bar_indicator_pos = 0

    else:
        assert filter_half_size >= cut
        # 0 Is at the center of the axis. so we get the relative position
        bar_indicator_pos = filter_half_size - cut

    plot_orientation = 'vertical'
    assert plot_orientation == 'vertical', "Horizontal plot orientation not implemented yet."
    # bar length is the difference between min and max of the filter values
    min_bar_pos = np.min(filter_1d)
    max_bar_pos = np.max(filter_1d)
    coord_x1 = min_bar_pos if plot_orientation == 'vertical' else bar_indicator_pos
    coord_x2 = max_bar_pos if plot_orientation == 'vertical' else bar_indicator_pos
    coord_y1 = bar_indicator_pos if plot_orientation == 'vertical' else min_bar_pos
    coord_y2 = bar_indicator_pos if plot_orientation == 'vertical' else max_bar_pos

    # top bar indicator
    line = mlines.Line2D([coord_x1, coord_x2], [coord_y1, coord_y2], color='white', linewidth=2.5)
    ax.add_line(line)

    # bottom  bar indicator
    line = mlines.Line2D([coord_x1, coord_x2], [-coord_y1, -coord_y2], color='white', linewidth=2.5)
    ax.add_line(line)

    # add truncation info
    # if cut_indices is None it means there is no remaining part, so it is 0
    remaining_part = len(cut_indices) / (filter_half_size * 2) if cut_indices is not None else 0
    bar_info = f' {int((1 - remaining_part) * 100)}%'
    text_vertical_offset = 2
    # text in axis coordinates. If you want to draw text on exact plot values remove the axis transform
    # and use coordinate values (like drawing the bar indicator)
    # axes_2d[r].text(0.5, 0.5, bar_info, fontsize=14, fontweight='bold', alpha=0.75,
    #                 horizontalalignment='center', transform=axes_2d[r].transAxes)

    # text in value coordinates.
    ax.text((coord_x1 + coord_x2) / 2, bar_indicator_pos + text_vertical_offset, bar_info, fontsize=14,
            fontweight='bold', alpha=0.75, horizontalalignment='center', verticalalignment='center')

    # draw a transparent rectangle on the truncated areas
    patch_alpha = 0.2
    patch_color = 'grey'
    # top
    rect_top = Rectangle((coord_x1, coord_y1),  # anchor point
                         coord_x2 - coord_x1,  # width
                         cut if cut is not None else filter_half_size,  # height
                         fill=True, color=patch_color, alpha=patch_alpha)
    # top
    rect_bottom = Rectangle((coord_x1, -coord_y1),  # anchor point
                            coord_x2 - coord_x1,  # width
                            -cut if cut is not None else -filter_half_size,  # height
                            fill=True, color=patch_color, alpha=patch_alpha)

    ax.add_patch(rect_top)
    ax.add_patch(rect_bottom)


def highlght_cells(highlights, opts, ax, cell_offset_x=0, cell_offset_y=0):
    """Add rectangular patch to matrix cells.

    :param ndarray highlights: row/column of the highlighted cells
    :param OptionsPlots opts:
    :param Axes ax:
    :param int cell_offset_x:
    :param int cell_offset_y:
    """
    hm, hn = highlights.shape
    for i in range(0, hm):
        for j in range(0, hn):
            if highlights[i, j] != 0:
                # for some weird reason the rows and colum indices should be swapped to make this work
                ax.add_patch(Rectangle((j - cell_offset_x, i - cell_offset_y),  # anchor point
                                       1,  # width
                                       1,  # height
                                       fill=opts.highlight_fill,
                                       edgecolor=opts.highlight_edgecolor,
                                       facecolor=opts.highlight_facecolor,
                                       # color=opts.highlight_fillcolor,
                                       lw=opts.highlight_line_width))


def add_object_mask(interior_mask, contour_mask, opts_plots, show_interior_mask, fill_interior_mask, show_contour_mask,
                    fill_contour_mask, ax, interior_facecolor='cornflowerblue', contour_facecolor='cornflowerblue',
                    edge_color='azure', interior_linewidth=1, contour_linewidth=1):
    """Add a mask to visualize an object in a matrix domain. Matrices are 2D numpy arrays.

    Here is a `list of colors <https://matplotlib.org/stable/gallery/color/named_colors.html>`_.

    :param ndarray interior_mask:
    :param ndarray contour_mask:
    :param OptionsPlots opts_plots:
    :param bool show_interior_mask:
    :param bool fill_interior_mask:
    :param bool show_contour_mask:
    :param bool fill_contour_mask:
    :param Axes ax:
    :param str interior_facecolor:
    :param str contour_facecolor:
    :param str edge_color:
    :param int interior_linewidth:
    :param int contour_linewidth:
    """
    # fill the interior cells
    if show_interior_mask:
        super_impose_mask(M=interior_mask, opts=opts_plots, edge_color=edge_color, line_width=interior_linewidth,
                          fill=fill_interior_mask, face_color=interior_facecolor, ax=ax)
    # highlight the contour edges
    if show_contour_mask:
        super_impose_mask(M=contour_mask, opts=opts_plots, edge_color=edge_color, line_width=contour_linewidth,
                          fill=fill_contour_mask, face_color=contour_facecolor, ax=ax)


def add_collision_masks(interior_mask, contour_mask, opts_plots, opts_subplots, axes):
    """Add a mask to visualize a collider in a matrix domain for all subplot axes. Matrices are 2D numpy arrays.

    :param ndarray interior_mask:
    :param ndarray contour_mask:
    :param OptionsPlots opts_plots:
    :param OptionsSubPlots opts_subplots:
    :param Axes axes: all subplot axes
    """
    interior_face_color = 'cornflowerblue'
    contour_face_color = 'cornflowerblue'
    edge_color = 'azure'
    interior_line_width = 0
    contour_line_width = 1

    # just put the following functions in the same order as the subplots

    axes_size = np.ndarray(opts_subplots.layout).size
    num_axes = 1 if axes_size == 1 else len(axes)

    for ax_index in range(num_axes):
        if num_axes != 1:
            ax = axes[ax_index]
        else:
            ax = axes

        add_object_mask(interior_mask=interior_mask, contour_mask=contour_mask, opts_plots=opts_plots,
                        show_interior_mask=True, fill_interior_mask=False,
                        show_contour_mask=True, fill_contour_mask=False, ax=ax,
                        interior_facecolor='black', contour_facecolor=contour_face_color,
                        edge_color=edge_color,
                        interior_linewidth=interior_line_width, contour_linewidth=contour_line_width)


def super_impose_mask(M, opts, edge_color, line_width, ax, fill=False, face_color='pink',
                      allow_mask_interpolation=True):
    """Superimpose a 2D mask onto a 2D domain matrix.

    .. warning::
        We override some variables in the :code:`OptionsPlots`, which will persist after the lifespan of this method.

    :param ndarray M: input matrix
    :param OptionsPlots opts:
    :param str edge_color:
    :param int line_width:
    :param Axes ax:
    :param bool fill:
    :param str face_color:
    :param bool allow_mask_interpolation:
    """
    opts.highlight_edgecolor = edge_color
    opts.highlight_fill = fill
    opts.highlight_facecolor = face_color
    opts.highlight_line_width = line_width

    cell_offset_x = 0.5 if opts.interpolate and allow_mask_interpolation else 0
    cell_offset_y = 0.5 if opts.interpolate and allow_mask_interpolation else 0

    highlght_cells(highlights=M, opts=opts, cell_offset_x=cell_offset_x, cell_offset_y=cell_offset_y, ax=ax)


def draw_line(ax, from_x, from_y, to_x, to_y, color='red', linewidth=2.5, linestyle='--'):
    """Draw line in value coordinates.

    :param Axes ax:
    :param float from_x: starting x position
    :param float from_y: starting y position
    :param float to_x: ending x position
    :param float to_y: ending y position
    :param str color:
    :param int linewidth:
    :param str linestyle:
    """
    line = mlines.Line2D([from_x, to_x], [from_y, to_y], color=color, linewidth=linewidth, linestyle=linestyle)
    ax.add_line(line)


def add_text(ax, coord_x, coord_y, text, fontsize=14, color='red', fontweight='bold', alpha=0.75,
             horizontal_alignment='center', vertical_alignment='center'):
    """Add text in value coordinates.

    :param Axes ax:
    :param float coord_x:
    :param float coord_y:
    :param str text:
    :param int fontsize:
    :param str color:
    :param str fontweight:
    :param float alpha: transparency
    :param str horizontal_alignment: **Default=** :code:`"center"`
    :param str vertical_alignment: **Default=** :code:`"center"`
    """
    ax.text(coord_x, coord_y, text, fontsize=fontsize, fontweight=fontweight, alpha=alpha, color=color,
            horizontalalignment=horizontal_alignment, verticalalignment=vertical_alignment)


def fmt_imshow(x: str):
    """
    :param str x:
    :return:
    """
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} times 10^{{{}}}$'.format(a, b)


class FFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.2f", offset=True, useMathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=useMathText)

    def _set_format(self):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


def plot_matrix_2d_simple(M, title, opts, rotate_ticks=True, highlights=None, cmap=None, cbar_percentiles=None,
                          ax=None):
    """
    :param ndarray M: 2D input matrix
    :param str title:
    :param OptionsPlots opts:
    :param bool rotate_ticks: rotate ticks by 45 degrees
    :param ndarray highlights: if not :code:`None`, highlight the cells with non-zero values. \
        Must be the same size as input matrix :code:`M`.
    :param str cmap: color map. Can be `matplotlib.colors.Colormap` too. If not :code:`None` \
        overwrite the cmap in :code:`opts`, else use :code:`opts.cmap`
    :param list cbar_percentiles: percentile list, e.g. :code:`[95, 99]`
    :param Axes ax: input axis
    :return: axis
    """
    # as the convention in matplotlib any color map has a reversed verion by adding _r

    if ax is None:
        plt.figure()
        ax = plt.axes()

    # force window to a specific position
    # mng = plt.get_current_fig_manager()
    # mng.window.wm_geometry("+1100+100")  # example "+1950+0"

    # little hack to get the cbar in percent...
    M = 100 * M

    if opts.interpolate:
        ax.imshow(M, cmap=cmap if cmap is not None else opts.cmap, interpolation='spline16', alpha=opts.alpha)

        # -- work around for imshow color bar
        # import matplotlib.ticker as ticker
        # ax.set_colorbar(im_plt, format=ticker.FuncFormatter(fmt_imshow))
        # -- NOT WORKING, but the format trick is nice

    else:
        # Seaborn tricks https://towardsdatascience.com/heatmap-basics-with-pythons-seaborn-fb92ea280a6c
        import matplotlib.ticker as ticker
        formatter = FFormatter(useMathText=True)
        if opts.cbar_scientific:
            formatter.set_scientific(opts.cbar_scientific)
            formatter.set_powerlimits((-2, 2))
        # Seaborn cbar kws keyword arguments
        # https: // matplotlib.org / stable / api / _as_gen / matplotlib.pyplot.colorbar.html

        cbar_kws = {"shrink": opts.cbar_shrink,
                    'format': formatter,
                    "orientation": opts.cbar_orientation,
                    "pad": 0.05}

        if opts.cbar_only_min_max:
            min_input = np.min(M)
            max_input = np.max(M)

            pct_input = np.percentile(M, [])
            if opts.cbar_add_percentile and cbar_percentiles is not None:
                pct_input = np.percentile(M, cbar_percentiles)

            cbar_kws["ticks"] = [min_input, *pct_input, max_input]

        ax_sns = sns.heatmap(M, ax=ax, cmap=cmap if cmap is not None else opts.cmap,
                             cbar=opts.cbar, annot=opts.show_values,
                             linewidths=opts.line_widths, fmt=opts.fmt,
                             cbar_kws=cbar_kws)
        if rotate_ticks:
            ax_sns.collections[0].colorbar.ax.tick_params(rotation=45)

    # Highlight cells
    if highlights is not None:
        assert M.shape == highlights.shape
        cell_offset_x = 0.5 if opts.interpolate else 0
        cell_offset_y = 0.5 if opts.interpolate else 0
        highlght_cells(highlights=highlights, opts=opts,
                       cell_offset_x=cell_offset_x, cell_offset_y=cell_offset_y,
                       ax=ax)

    ax.set_title(title)

    if opts.no_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if opts.aspect_ratio is not None:
        # y-unit to x-unit ratio

        # get x and y limits
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()

        # set aspect ratio
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * opts.aspect_ratio)

    if opts.plt_show:
        plt.show()

    return ax


def plot_matrix_2d_advanced(M, title, opts, ax_img=None, cbar_ax=None):
    """With color bar control. By default, auto-generate the figure, image and color bar axes \
    (when all are :code:`None` in the input).

    :param ndarray M: 2D input matrix
    :param str title:
    :param OptionsPlots opts:
    :param Axes ax_img: image axis
    :param Axes cbar_ax: color bar axis
    :return: figure and image axis
    """
    fig = None

    # auto-generate axes if necessary
    if ax_img is None:  # if image axis is None normally there is no cbar axis either
        cbar_height_ratio = 0.01
        img_height_ratio = 0.4
        grid_kws = {"height_ratios": (img_height_ratio, cbar_height_ratio), "hspace": .3}
        fig, (ax_img, ax_cbar_new) = plt.subplots(2, gridspec_kw=grid_kws)
        # ...make and assign cbar axis
        cbar_ax = ax_cbar_new  # overwrite the cbar axis anyway regardless if it is None or not

    # plot
    if cbar_ax is not None:
        ax_img = sns.heatmap(M, ax=ax_img, cmap=opts.cmap, cbar=opts.cbar,
                             cbar_ax=cbar_ax, cbar_kws={"orientation": opts.cbar_orientation},
                             annot=opts.show_values, linewidths=opts.line_widths, fmt=opts.fmt)

    else:  # no cbar axis is given, but an image axis is given (not auto-generated by this function):
        # action: let sns take care of it
        ax_img = sns.heatmap(M, ax=ax_img, cmap=opts.cmap, cbar=opts.cbar,
                             cbar_kws={"orientation": opts.cbar_orientation},
                             annot=opts.show_values, linewidths=opts.line_widths, fmt=opts.fmt)

    ax_img.set_title(title)

    if opts.no_ticks:
        ax_img.set_xticks([])
        ax_img.set_yticks([])

    if opts.aspect_ratio is not None:
        # y-unit to x-unit ratio

        # get x and y limits
        x_left, x_right = ax_img.get_xlim()
        y_low, y_high = ax_img.get_ylim()

        # set aspect ratio
        ax_img.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * opts.aspect_ratio)

    if opts.cbar and cbar_ax is not None:
        # a horizontal bar under the image
        cbar_x_left, cbar_x_right = cbar_ax.get_xlim()
        cbar_y_low, cbar_y_high = cbar_ax.get_ylim()

        # making the bar the same length as the image
        const_hv_ratio = 1 / 40
        cbar_ax.set_aspect(abs((cbar_x_right - cbar_x_left) / (cbar_y_low - cbar_y_high)) * const_hv_ratio)

        # forcing only 2 ticks for the start and the end
        cbar_ax.xaxis.set_major_locator(plt.LinearLocator(2))

    if opts.axis_background_color is not None:
        ax_img.set_facecolor(opts.axis_background_color)  # must be string
        # ax.set_facecolor((1.0, 0.47, 0.42)) # or a tuple

    if opts.plt_show:
        plt.show()

    return fig, ax_img


def plot_matrix_3d(M, title, opts, cmap=None, ax=None, cbar_show=False, cbar_ax=None, fig=None):
    """ By default, auto-generate the figure, image and color bar axes (when all are :code:`None` in the input).

    :param ndarray M: 3D input array (tensor)
    :param str title:
    :param OptionsPlots opts:
    :param str cmap: color map. Can be `matplotlib.colors.Colormap` too. If not :code:`None` \
        overwrite the cmap in :code:`opts`, else use :code:`opts.cmap`
    :param Axes ax:
    :param bool cbar_show:
    :param Axes cbar_ax:
    :param matplotlib.figure.Figure fig:
    :return: axis
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if opts.interp_3d:
        from scipy import interpolate

        x = np.linspace(1, M.shape[0], M.shape[0])
        y = np.linspace(1, M.shape[1], M.shape[1])

        X, Y = np.meshgrid(x, y)
        tck = interpolate.bisplrep(X, Y, M, s=0)
        M_new = interpolate.bisplev(x[:, 0], y[0, :], tck)

        ax.plot_surface(x, y, M_new, rstride=1, cstride=1,
                        cmap=cmap if cmap is not None else opts.cmap,
                        edgecolor='none')

    else:

        x = np.linspace(1, M.shape[0], M.shape[0])
        y = np.linspace(1, M.shape[1], M.shape[1])
        X, Y = np.meshgrid(x, y)

        ax.plot_surface(X, Y, M, rstride=1, cstride=1,
                        cmap=cmap if cmap is not None else opts.cmap,
                        edgecolor='none')

    # color bar
    if (opts.cbar or cbar_show) and fig is not None and cbar_ax is not None:
        scalar_mappable = cm.ScalarMappable(cmap=cmap if cmap is not None else opts.cmap)
        cb = fig.colorbar(scalar_mappable, cax=cbar_ax, orientation=opts.cbar_orientation)
        cb.outline.set_visible(False)  # Remove outline

        if opts.cbar_orientation == 'horizontal':
            # a horizontal bar under the image
            cbar_x_left, cbar_x_right = cbar_ax.get_xlim()
            cbar_y_low, cbar_y_high = cbar_ax.get_ylim()

            # making the bar the same length as the image
            const_hv_ratio = 1 / 40
            cbar_ax.set_aspect(abs((cbar_x_right - cbar_x_left) / (cbar_y_low - cbar_y_high)) * const_hv_ratio)

            # forcing only 2 ticks for the start and the end
            cbar_ax.xaxis.set_major_locator(plt.LinearLocator(2))

    if opts.axis_background_color is not None:
        ax.set_facecolor(opts.axis_background_color)  # must be string
        # ax.set_facecolor((1.0, 0.47, 0.42)) # or a tuple

    ax.set_aspect('auto')

    if opts.beautify:
        beautify_axis(ax)

    ax.set_title(title)

    if opts.plt_show:
        plt.show()

    return ax


def plot_vector_field_2d(Mx, My, title, plt_show=False, interp=False, flip_vertically=True, cmap=None,
                         vector_variable_cmap=True, variable_cmap_mode=ColorMapModesVectorField.VECTOR_ANGLE,
                         show_cbar=False, ax_no_ticks=False,
                         down_sample_rate=None, alpha=1.0, background_tone=.0, scale=None, ax=None):
    """
    :param ndarray Mx: x components of the input matrix
    :param ndarray My: y components of the input matrix
    :param str title:
    :param bool plt_show: force plot if :code:`True`, else delay the plot show
    :param bool interp: interpolate values
    :param bool flip_vertically: flip vectors
    :param str cmap: color map. Can be `matplotlib.colors.Colormap` too. If not :code:`None` \
        overwrite the cmap in :code:`opts`, else use :code:`opts.cmap`
    :param bool vector_variable_cmap: apply custom dynamic vector color
    :param ColorMapModesVectorField variable_cmap_mode: vector angle or vector length color map
    :param bool show_cbar:
    :param bool ax_no_ticks:
    :param int down_sample_rate:
    :param float alpha:
    :param float background_tone: background color tone
    :param float scale: scales the arrow. Smaller values make the arrows longer. Good value :code:`1e4`. \
        **Default=** :code:`None` (auto)
    :param Axes ax:
    :return:
    """
    if ax is None:
        plt.figure()
        ax = plt.axes()

    if down_sample_rate is not None:
        assert down_sample_rate >= 1
        down_sample_rate = int(down_sample_rate)  # making sure it is always integer
        every_nth_arrow = down_sample_rate
        Mx = np.copy(Mx[::every_nth_arrow, ::every_nth_arrow])
        My = np.copy(My[::every_nth_arrow, ::every_nth_arrow])

    assert Mx.shape == My.shape

    rows, cols = Mx.shape
    x = np.linspace(0, cols - 1, cols)  # cols - horizontal - X
    y = np.linspace(0, rows - 1, rows)  # rows - vertical - Y
    # vector coordinates adjustment: cell center
    x -= 0.5
    y -= 0.5
    # additional coordinate offset when using imshow instead of seaborn heatmap
    if interp:
        x -= 0.5
        y -= 0.5
    X, Y = np.meshgrid(x, y)

    # flip vertically if needed
    U = np.flipud(Mx) if flip_vertically else Mx
    V = np.flipud(My) if flip_vertically else My

    if cmap is None:
        favorite_cmaps = ['cool', 'binary', 'spring', 'jet', 'Wistia', 'summer', 'coolwarm_r']
        cmap = favorite_cmaps[0]

    if vector_variable_cmap:
        if variable_cmap_mode == ColorMapModesVectorField.VECTOR_ANGLE:

            # original
            q = ax.quiver(X, Y, U, V,
                          np.arctan2(U, V),  # this is C: arrow colors (look up the documentation
                          # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.quiver.html)
                          units='width', angles='xy', cmap=cmap, alpha=alpha, scale=scale)
            if show_cbar:
                plt.colorbar(q, orientation='vertical')

        elif variable_cmap_mode == ColorMapModesVectorField.VECTOR_LENGTH:
            data_1d_x = U.flatten()
            data_1d_y = V.flatten()
            data_1d = 0.5 * (data_1d_x * data_1d_y)
            occurrence = U / np.sum(data_1d)
            norm = matplotlib.colors.Normalize()
            norm.autoscale(occurrence)
            ""
            # color map
            # https: // matplotlib.org / stable / tutorials / colors / colormaps.html
            # https://seaborn.pydata.org/tutorial/color_palettes.html

            # option 1
            # cmap = matplotlib.cm.spring
            cmap_obj = matplotlib.cm.get_cmap(cmap)

            # option 2
            # cm = matplotlib.cm.bwr # simple but zeros on the boundary are white

            # option 3
            # cm = matplotlib.cm.brg  # goes well with a salmon backgroundcm = matplotlib.cm.brg
            # # set background
            # ax.set_facecolor((1.0, 0.47, 0.42)) # salmon

            # other options
            # cm = matplotlib.cm.RdBu
            # cm = matplotlib.cm.RdYlBu_r
            # cm = matplotlib.cm.coolwarm

            sm = matplotlib.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])

            plt.quiver(X, Y, U, V, color=cmap_obj(norm(data_1d)), alpha=alpha, scale=scale)
            if show_cbar:
                plt.colorbar(sm)

    else:
        q = plt.quiver(X, Y, U, V, color='blue', alpha=alpha, scale=scale)
        if show_cbar:
            plt.colorbar(q, orientation='vertical')

    # set background
    ax.set_facecolor((background_tone, background_tone, background_tone))

    if ax_no_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    ax.set_aspect(1.0)
    ax.set_title(title)

    if plt_show:
        plt.show()


def add_subplot(M, title, sub_plots, opts_subplots, highlights=None, cbar_percentile_list=None, cmap=None):
    """Add an element to a list of subplots

    :param ndarray M: input matrix
    :param List[str] title:
    :param List[ndarray] sub_plots: list of subplot matrices
    :param OptionsSubPlots opts_subplots:
    :param List[ndarray] highlights: in case of plotting a matrix, the index values of the highlighted cells.
        This will be ignored if :code:`None`.
    :param List[float] cbar_percentile_list:
    :param List[str] cmap:
    """
    sub_plots.append(M)
    opts_subplots.titles.append(title)
    opts_subplots.highlights.append(highlights)
    opts_subplots.cbar_percentiles.append(cbar_percentile_list)
    opts_subplots.cmaps.append(cmap)


def init_subplots(sub_plots, opts_subplots):
    """
    :param List[ndarray] sub_plots: list of subplot matrices
    :param OptionsSubPlots opts_subplots:
    """
    sub_plots.clear()
    opts_subplots.titles.clear()
    opts_subplots.highlights.clear()
    opts_subplots.cbar_percentiles.clear()
    opts_subplots.cmaps.clear()


def plot_1d_array_grid(G, title, opts_general, opts_subplots):
    """Plot 1d arrays (usually filters) in a grid structure with automatic handling of its properties.

    :param List[ndarray] G: list of 1d arrays
    :param str title:
    :param OptionsPlots opts_general:
    :param OptionsSubPlots opts_subplots:
    :return:
    """
    rows = opts_subplots.layout[0]
    cols = opts_subplots.layout[1]

    fig, axes = plt.subplots(rows, cols)

    axes = axes.reshape((-1,))

    next_element = 0
    for r in range(rows):
        for c in range(cols):

            if next_element < len(G):
                plot_1d_array(array=G[next_element], title="" if (opts_subplots.titles) is None else
                              opts_subplots.titles[next_element],
                              plt_show=opts_general.plt_show, no_ticks=opts_general.no_ticks,
                              interpolate=opts_general.interp_1d, interp_res=opts_general.interp_1d_res,
                              ax=axes[next_element], )

            else:  # clear the remaining empty subplots
                clear_axis(axes[next_element])

            next_element += 1

    fig.suptitle(title, fontsize=14)


def plot_matrix_grid(G, projection, opts_general, opts_subplots, title=None, cmaps=None, fig_size=None,
                     clear_unused=True):
    """Plot 2d matrices in a grid structure with automatic handling of its properties.

    :param List[ndarray] G: list of 2d matrices
    :param str title:
    :param str projection: :code:`"2d"` or :code:`"3d"`
    :param OptionsPlots opts_general:
    :param OptionsSubPlots opts_subplots:
    :param cmaps: color map. Can be `matplotlib.colors.Colormap` too. If not :code:`None` \
        overwrite the cmap in :code:`opts`, else use :code:`opts.cmap`
    :param (float, float) fig_size: (width, height) in inches
    :param bool clear_unused: clear unused subplots. Do not clean if the remaining subplots are used
        by an external function.
    :return: axes
    """
    # when using this function make sure to disable plt_show in the options for proper behaviour.
    # you should call plt.show() explicitly after this function.
    assert not opts_general.plt_show, "If set true, only the first subplot is shown."

    rows = opts_subplots.layout[0]
    cols = opts_subplots.layout[1]

    if projection == '3d':
        fig, axes = plt.subplots(rows, cols, subplot_kw=dict(projection="3d"))
    else:
        fig, axes = plt.subplots(rows, cols)

    if fig_size is not None:
        fig.set_size_inches(fig_size[0], fig_size[1])

    if np.ndarray(opts_subplots.layout).size != 1:
        axes = axes.reshape((-1,))

    next_element = 0
    for r in range(rows):
        for c in range(cols):

            if next_element < len(G):
                if np.ndarray(opts_subplots.layout).size != 1:
                    ax = axes[next_element]
                else:
                    ax = axes

                if projection == '3d':
                    plot_matrix_3d(M=G[next_element], title="" if opts_subplots.titles is None else
                                   opts_subplots.titles[next_element], opts=opts_general,
                                   cmap=cmaps[next_element] if cmaps is not None else None,
                                   ax=ax)
                else:
                    plot_matrix_2d_simple(M=G[next_element], title="" if opts_subplots.titles is None else
                                          opts_subplots.titles[next_element], opts=opts_general,
                                          cbar_percentiles=opts_subplots.cbar_percentiles[next_element],
                                          highlights=opts_subplots.highlights[next_element],
                                          cmap=cmaps[next_element] if cmaps is not None else None,
                                          ax=ax)

            elif clear_unused:  # clear the remaining empty subplots
                clear_axis(axes[next_element])

            next_element += 1

    if title is not None:
        fig.suptitle(title, fontsize=14)

    return axes


def plot_scalar_field_unpack_opts(M, ax_img, fig, opts, vrange_im, title='', ax_cbar_image=None, ax_cbar_contour=None,
                                  vrange_contour=None, bar_indicator_val=None):
    """Plot scalar field. This is a helper function to unpack options when calling the main plot function.

    :param ndarray M: input matrix
    :param Axes ax_img: image axis
    :param matplotlib.figure.Figure fig:
    :param OptionsPlots opts:
    :param (float, float) vrange_im: (min, max); used to force the image color bar to a min and max
    :param str title:
    :param Axes ax_cbar_image: image color bar axis
    :param Axes ax_cbar_contour: contour color bar axis
    :param (float, float) vrange_contour: (min, max); used to force the contour color bar to a min and max
    :param float bar_indicator_val: if not :code:`None`, draw a little marker line on the color bar. Must be normalized.
    """
    plot_scalar_field(M=M, ax=ax_img, ax_cbar_image=ax_cbar_image, ax_cbar_contour=ax_cbar_contour, fig=fig,
                      ax_noticks=opts.no_ticks, cmap=opts.cmap, show_cbar=opts.cbar,
                      cbar_location=opts.cbar_location, cbar_orientation=opts.cbar_orientation,
                      blend_color=opts.blend_cmaps, cmap_secondary_blend=opts.cmap_secondary_blend,
                      interp_mode=opts.interp_2d_mode,
                      show_contour=opts.show_contour, show_contour_values=opts.contour_show_values,
                      contour_res=opts.contour_res, show_contour_cbar=opts.contour_cbar,
                      cmap_contour=opts.contour_cmap, contour_cbar_location=opts.contour_cbar_location,
                      contour_cbar_orientation=opts.contour_cbar_orientation,
                      contour_cbar_border=opts.contour_cbar_border, contour_cbar_no_ticks=opts.contour_cbar_no_ticks,
                      vrange_contour=vrange_contour, bar_indicator_val=bar_indicator_val, vrange_im=vrange_im,
                      title=title)


def plot_scalar_field(M, ax, fig, title, show_contour, interp_mode, cmap, blend_color, cmap_secondary_blend='flag',
                      show_cbar=True, ax_cbar_image=None, cbar_location='right', cbar_orientation='vertical',
                      show_contour_values=False, contour_res=100, cmap_contour='gist_heat',
                      ax_cbar_contour=None, show_contour_cbar=True,
                      contour_cbar_location='right', contour_cbar_orientation='vertical',
                      bar_indicator_val=None, vrange_im=None, vrange_contour=None,
                      contour_cbar_border=False, contour_cbar_no_ticks=True, ax_noticks=False,
                      cbar_border=False, cbar_outline_color='black', cbar_label=None, cbar_label_color='white',
                      cbar_ticks=True, cbar_tick_color=None):
    """Plot a scalar field with contours.

    :param ndarray M: input matrix
    :param Axes ax: image axis
    :param matplotlib.figure.Figure fig:
    :param str title:
    :param bool show_contour:
    :param str interp_mode: :code:`[None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']`. \
           Here is the\
            `full list <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_\
             with examples.
    :param str cmap: colormap
    :param bool blend_color: blend between the image cmap and an additional one given in :code:`cmap_secondary_blend` \
        (usually to make a more detailed cmap)
    :param str cmap_secondary_blend: when blending, blend image cmap with this cmap
    :param bool show_cbar:
    :param Axes ax_cbar_image: image axis. :code:`None` by default (if show cbar, this will automatically create one \
        by dividing the image axis)
    :param str cbar_location: :code:`'right', 'left', 'top', 'bottom'`
    :param str cbar_orientation: :code:`'horizontal', 'vertical'`
    :param bool show_contour_values:
    :param int contour_res: contour resolution
    :param str cmap_contour: contour colormap
    :param ax_cbar_contour: contour color bar axis. :code:`None` by default (if show contour cbar, this will automatically \
        create one by dividing
        the image axis)
    :param bool show_contour_cbar:
    :param float bar_indicator_val: if not :code:`None`, path indicator in the color bar. \
        Bar indicator sits on the color bar. Must be normalized.
    :param (float, float) vrange_im: (min, max); used to force the image color bar to a min and max
        Can be left :code:`None` and `imshow` will automatically compute it
    :param (float, float) vrange_contour: (min, max); used to force the contour color bar to a min and max
    :param str contour_cbar_location: :code:`'right', 'left', 'top', 'bottom'`
    :param contour_cbar_orientation: for now only :code:`'vertical'` works
    :param bool contour_cbar_border: remove contour cbar borders when :code:`False`
    :param bool contour_cbar_no_ticks: remove contour cbar ticks
    :param bool ax_noticks:
    :param bool cbar_border:
    :param str cbar_outline_color:
    :param str cbar_label:
    :param str cbar_label_color:
    :param bool cbar_ticks:
    :param str cbar_tick_color:
    :return: image and contour handles
    """

    # auto-generate axes if necessary
    if ax is None:  # if image axis is None normally there is no cbar axis either
        assert fig is not None, \
            'Not sure where to add the new axis for an existing figure.. better create it in the figure.'
        fig, ax = plt.subplots(1, 1)

    # ====== Blend Colors =========
    # The blending of two color maps was inspired by
    # https://stackoverflow.com/questions/10127284/overlay-imshow-plots-in-matplotlib

    if blend_color:
        cmap1 = cm.get_cmap(cmap)
        cmap2 = cm.get_cmap(cmap_secondary_blend)

        cmap2._init()  # create the _lut array, with rgba values
        # create your alpha array and fill the colormap with them.
        # here it is progressive, but you can create whatever you want
        alphas = np.linspace(0, 0.415, cmap1.N + 3)  # max alpha = 0.115
        cmap2._lut[:, -1] = alphas
        if vrange_im is not None:
            ax.imshow(M, interpolation=interp_mode, cmap=cmap1, vmin=vrange_im[0], vmax=vrange_im[1])
            im = ax.imshow(M, interpolation=interp_mode, cmap=cmap2, vmin=vrange_im[0], vmax=vrange_im[1])
        else:
            ax.imshow(M, interpolation=interp_mode, cmap=cmap1)
            im = ax.imshow(M, interpolation=interp_mode, cmap=cmap2)
    else:
        if vrange_im is not None:
            im = ax.imshow(M, interpolation=interp_mode, cmap=cmap, vmin=vrange_im[0], vmax=vrange_im[1])
        else:
            im = ax.imshow(M, interpolation=interp_mode, cmap=cmap)

    if ax_noticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    # ====== Contours =========
    # diving axes for better control over the color bars
    divider = make_axes_locatable(ax)

    contours = []
    if show_contour:
        rows, cols = M.shape
        X = np.linspace(0, cols - 1, cols)
        Y = np.linspace(0, rows - 1, rows)
        X, Y = np.meshgrid(X, Y)

        # ==== New color map to make a section of high value colors fixed ===
        # half range cut
        cut = int(255 / 2)
        cmap_edit = cm.get_cmap(cmap_contour, 256)
        new_colors = cmap_edit(np.linspace(0, 1, 256))
        fixed_color = np.array(cmap_edit(cut))
        new_colors[cut:, :] = fixed_color
        cmap_cont = ListedColormap(new_colors)

        # forcing range on contours
        if vrange_contour is not None:
            contours = ax.contour(X, Y, M, contour_res, cmap=cmap_cont, linewidths=0.75,
                                  vmin=vrange_contour[0], vmax=vrange_contour[1])  # 'RdGy' 'gist_heat'
        else:
            contours = ax.contour(X, Y, M, contour_res, cmap=cmap_cont, linewidths=0.75)  # 'RdGy' 'gist_heat'

        # Show contour values
        if show_contour_values:
            import matplotlib.ticker as ticker
            fmt = ticker.LogFormatterSciNotation()
            fmt.create_dummy_axis()
            ax.clabel(contours, inline=True, fmt=fmt, fontsize=10)

        # contours color bar
        if fig is not None and show_contour_cbar:
            # if a contour cbar axis is not given, make one
            if ax_cbar_contour is None:
                ax_cbar_contour = divider.append_axes(contour_cbar_location, size='5%', pad=0.05)
            assert contour_cbar_orientation == 'vertical', "For now only vertical works; matplotlib complains otherwise"
            cb_cont = fig.colorbar(contours, cax=ax_cbar_contour, orientation=contour_cbar_orientation)
            cb_cont.outline.set_visible(contour_cbar_border)  # remove borders when False
            if contour_cbar_no_ticks:
                cb_cont.set_ticks([])

    ax.set_title(title)

    # image color bar
    if show_cbar:
        # if a cbar axis is not given, make one
        if ax_cbar_image is None:
            ax_cbar_image = divider.append_axes(cbar_location, size='5%', pad=0.05)

        if fig is not None:
            cb_im = fig.colorbar(im, cax=ax_cbar_image, orientation=cbar_orientation)

            # set color bar edgecolor
            cb_im.outline.set_visible(cbar_border)  # remove borders
            if cbar_border:
                cb_im.outline.set_edgecolor(cbar_outline_color)

            # set color bar label plus label color
            if cbar_label is not None:
                cb_im.set_label(cbar_label, color=cbar_label_color)

            # set color bar tick color
            if cbar_ticks and cbar_tick_color is not None:
                cb_im.ax.xaxis.set_tick_params(color=cbar_tick_color)
                cb_im.ax.yaxis.set_tick_params(color=cbar_tick_color)
            else:
                cb_im.set_ticks([])

        # bar indicator on the color bar
        if bar_indicator_val is not None and ax_cbar_image is not None:
            bar_length = 10
            coord_x1 = 0 if cbar_orientation == 'vertical' else bar_indicator_val
            coord_x2 = bar_length if cbar_orientation == 'vertical' else bar_indicator_val
            coord_y1 = bar_indicator_val if cbar_orientation == 'vertical' else 0
            coord_y2 = bar_indicator_val if cbar_orientation == 'vertical' else bar_length

            # bar indicator sits on the bar color. It must be normalized
            # background
            lines = mlines.Line2D([coord_x1, coord_x2], [coord_y1, coord_y2], color='white', linewidth=5.0)
            ax_cbar_image.add_line(lines)
            # bar itself
            lines = mlines.Line2D([coord_x1, coord_x2], [coord_y1, coord_y2], color='red', linewidth=2.5)
            ax_cbar_image.add_line(lines)

    return im, contours


def clear_axis(axis):
    """
    :param Axes axis:
    :return:
    """
    axis.clear()
    axis.axis("off")
    axis.set_visible(False)
    axis.remove()


def compute_layout(columns, num_elements):
    """Compute the layout rows and columns for a desired number of columns and the total number of elements.

    :param int columns:
    :param int num_elements: total number of elements
    :return: (rows, columns)
    """
    rows = int(num_elements / columns)
    rows += 0 if num_elements % columns == 0 else 1
    return rows, columns


def beautify_axis(ax):
    """ Beautify axis by hiding the axis, ticks, axis lines, and setting the panes to black.

    :param Axes ax:
    """
    ax.set_axis_off()
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)


def correct_color_bar_2d(M, interior_mask):
    """Due to the likely presence of a solid object inside the domain the color bar might be wrongly shifted.
    This is because the interior part of it is not updated during the solve and could remain zero or some other
    initial const value. To fix this, we find and set them to be the average of min and max of the \
    domain, so they do not shift the color bar.

    :param ndarray M: input 2d matrix
    :param ndarray interior_mask: mask matrix
    :return: adjusted matrix
    """
    avg = 0
    count = 0
    m, n = M.shape
    for i in range(0, m):
        for j in range(0, n):
            if interior_mask[i, j] != 1:
                avg += M[i, j]
                count += 1

    avg /= count

    for i in range(0, m):
        for j in range(0, n):
            if interior_mask[i, j] == 1:
                M[i, j] = avg

    return M


def simple_animation_examples():
    """Contains two animations. The first one is a random walk plot. The second is an image animation.

    Other examples are found
    `here <https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/>`_.
    """

    import matplotlib.animation as animation

    # pass this to animation.FuncAnimation, if needed
    def init_func():
        # This is not necessary, but a good placeholder if you want to have pre-process steps for
        # setting up the plot (axis adjustments etc).
        # Might need to be careful with 'blit'-- setting it False would make things easier
        blah = 2

    def update_line(num, data, line):
        line.set_data(data[..., :num])
        return line,

    fig1 = plt.figure()

    data = np.random.rand(2, 25)
    l, = plt.plot([], [], 'r-')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.title('test')
    line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l), init_func=init_func,
                                       interval=50, blit=False)

    # save
    # To save the animation, use the command: line_ani.save('lines.mp4')

    fig2 = plt.figure()

    x = np.arange(-9, 10)
    y = np.arange(-9, 10).reshape(-1, 1)
    base = np.hypot(x, y)
    ims = []
    for add in np.arange(15):
        ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=300, blit=True)
    # To save this second animation with some metadata, use the following command:
    # im_ani.save('im.mp4', metadata={'artist':'Guido'})

    # Saving the animation...

    # FFwriter = animation.FFMpegWriter()
    # ani.save('basic_animation.mp4', writer = FFwriter)

    plt.show()


def adjust_axis_projection(fig, ax, grid_spec, projection):
    """Change the axis projection by removing and re-adding the axis.

    :param matplotlib.figure.Figure fig:
    :param Axes ax:
    :param matplotlib.GridSpec grid_spec:
    :param str projection: :code:`'2d'` or :code:`'3d'`
    :return: axis
    """
    ax.remove()
    ax = fig.add_subplot(grid_spec, projection=projection)
    return ax


def initialize_grid_layout_gridspec(fig_size, widths_ratios, heights_ratios, constrained_layout=True,
                                    canvas_background_color=None):
    """

    :param (float, float) fig_size: (width, height)
    :param List[float] widths_ratios:
    :param List[float] heights_ratios:
    :param bool constrained_layout: nice and tidy layout (can't manually tweak it though).

        .. warning::
            this is not compatible with fine adjustments of GridSpec.
    :param str canvas_background_color:
    :return: figure, and the created grid spec
    """
    fig = plt.figure(figsize=fig_size, constrained_layout=constrained_layout, facecolor=canvas_background_color)
    nrows = len(heights_ratios)
    ncols = len(widths_ratios)
    grid_spec = fig.add_gridspec(nrows=nrows, ncols=ncols,
                                 width_ratios=widths_ratios, height_ratios=heights_ratios)

    return fig, grid_spec


def add_grid_subplots_gridspec(fig, grid_spec, annot=False):
    """Add axes to an already created figure layout using GridSpec.

    :param matplotlib.figure.Figure fig:
    :param matplotlib.GridSpec grid_spec: created GridSpec
    :param bool annot: show axis info annotations
    :return: axes
    """
    # example of accessing params in the grid spec.
    widths_ratios = grid_spec._col_width_ratios
    heights_ratios = grid_spec._row_height_ratios

    nrows = len(heights_ratios)
    ncols = len(widths_ratios)

    axes = []
    for row in range(nrows):
        axes_row = []
        for col in range(ncols):
            ax = fig.add_subplot(grid_spec[row, col])
            axes_row.append(ax)
            if annot:
                label = 'Width: {}\nHeight: {}'.format(widths_ratios[col], heights_ratios[row])
                ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')
        axes.append(axes_row)

    return axes


def make_grid_layout_gridspec(fig_size, widths_ratios, heights_ratios, constrained_layout=True, annot=False):
    """Create a detailed grid subplots with full control over the layout

    :param (float, float) fig_size: (width, height)
    :param List[float] widths_ratios:
    :param List[float] heights_ratios:
    :param bool constrained_layout: nice and tidy layout (can't manually tweak it though).

        .. warning::
            this is not compatible with fine adjustments of GridSpec.
    :param annot: show axis info annotations
    :return: fig, axes list in the shape of [nrows][ncols], and the created grid spec
    """
    # init layout
    fig, grid_spec = initialize_grid_layout_gridspec(fig_size=fig_size,
                                                     widths_ratios=widths_ratios, heights_ratios=heights_ratios,
                                                     constrained_layout=constrained_layout)
    # add axes
    axes = add_grid_subplots_gridspec(fig=fig, grid_spec=grid_spec, annot=annot)

    return fig, axes, grid_spec


def grid_layout_example_1():
    """An example of making complex grids using manual widths and heights."""
    # ratios
    widths = [2, 3, 1.5]
    heights = [1, 3, 2]

    constrained_layout = True  # nice and tidy layout --> [only use it when NOT fine adjusting the layout!]

    fig, axes, grid_spec = make_grid_layout_gridspec(fig_size=(10, 10),
                                                     widths_ratios=widths, heights_ratios=heights,
                                                     constrained_layout=constrained_layout,
                                                     annot=True  # if you want to print info (example of accessing
                                                     # params in grid_spec)
                                                     )
    # fine-tuning the axes if desired
    # Note: if we are making fine adjustments to the layout (like below) we can't use constraint layout
    if not constrained_layout:
        fig.subplots_adjust(hspace=0, wspace=0.2)

    # adjust the projection mode AFTER the call to add subplot axes.
    row = 0
    col = 0
    adjust_axis_projection(fig=fig, ax=axes[row][col], grid_spec=grid_spec[row, col], projection='3d')

    plt.show()


def make_grid_layout_mosaic(layout_str, fig_size, constrained_layout=True, canvas_background_color=None,
                            subplot_kw=None, grid_spec_kw=None):
    """Make a figure and its mosaic subplots.\
    Check out this `example <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot_mosaic.html#matplotlib.pyplot.subplot_mosaic>`_\
    or `this one <https://matplotlib.org/stable/tutorials/provisional/mosaic.html>`_ to learn how to gain\
    more control over mosaic subplots.

    :param List[List[str]] layout_str: :code:`dict[label, Axes]` A dictionary mapping the labels to the Axes objects.
        The order of the axes is left-to-right and top-to-bottom of their position in the total layout.
    :param (float, float) fig_size: (width, height)
    :param bool constrained_layout: nice and tidy layout (can't manually tweak it though).

        .. warning::
            this is not compatible with fine adjustments of GridSpec.
    :param str canvas_background_color:
    :param subplot_kw: See :func:`add_grid_subplots_mosaic`
    :param grid_spec_kw: See :func:`add_grid_subplots_mosaic`
    :return:
        - **fig**: new figure
        - **axs**: :code:`dict[label, Axes]`: A dictionary mapping the labels to the Axes objects. The order of the axes is \
            left-to-right and top-to-bottom of their position in the total layout.

    """
    fig = plt.figure(figsize=fig_size, constrained_layout=constrained_layout, facecolor=canvas_background_color)
    axs = add_grid_subplots_mosaic(fig=fig, layout_str=layout_str, subplot_kw=subplot_kw, grid_spec_kw=grid_spec_kw)
    return fig, axs


def add_grid_subplots_mosaic(fig, layout_str, subplot_kw=None, grid_spec_kw=None):
    """Add subplots to an already existing figure using mosaics.

    Check out this `example <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot_mosaic.html#matplotlib.pyplot.subplot_mosaic>`_\
    or `this one <https://matplotlib.org/stable/tutorials/provisional/mosaic.html>`_ to learn how to gain \
    more control over mosaic subplots.

    :param matplotlib.figure.Figure fig:
    :param List[List[str]] layout_str: :code:`dict[label, Axes]` A dictionary mapping the labels to the Axes objects.
        The order of the axes is left-to-right and top-to-bottom of their position in the total layout.
    :param subplot_kw: check out `subplot_kw doc <https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.add_subplot>`_.
    :param grid_spec_kw: check out `grid_spec_kw doc <https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.gridspec.GridSpec.html#matplotlib.gridspec.GridSpec>`_.
    :return: axes
    """
    axs = fig.subplot_mosaic(layout_str, subplot_kw=subplot_kw, gridspec_kw=grid_spec_kw)
    return axs


def grid_layout_example_2():
    """An example of creating a grid using mosaic. String based layout (easier control when combining axes).

    Check out this `example <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot_mosaic.html#matplotlib.pyplot.subplot_mosaic>`_\
    or `this one <https://matplotlib.org/stable/tutorials/provisional/mosaic.html>`_ to learn how to gain \
    more control over mosaic subplots.
    """
    import matplotlib.transforms as mtransforms

    layout_str = [['a)', 'c)', 'c)'],
                  ['b)', 'c)', 'c)'],
                  ['d)', '.', 'e)']]  # '.' means empty slot

    fig, axs = make_grid_layout_mosaic(fig_size=(10, 10), layout_str=layout_str, constrained_layout=True)

    # display labels...
    for label, ax in axs.items():
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

    plt.show()


if __name__ == '__main__':
    # --------------------- UNCOMMENT TO RUN THE DEMOS...

    # Basic tester functions...

    # Simple animation
    # UNCOMMENT------------------------------------------------
    # simple_animation_examples()

    # Making complex grids using GridSpec: manual widths and heights
    # UNCOMMENT------------------------------------------------
    grid_layout_example_1()

    # Making complex grids using Mosaic: string based layout (easier control when combining axes)
    # UNCOMMENT------------------------------------------------
    # grid_layout_example_2()
