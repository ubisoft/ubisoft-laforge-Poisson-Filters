""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`
"""

import matplotlib.pyplot as plt
import numpy as np
import src.helper.commons as com
import src.helper.iohandler as io
import src.helper.visualizer as vis
import src.functions.decompositions as dec
import src.functions.generator as gen
import src.functions.mathlib as mlib
import sys
sys.path.insert(0, './')


def plot_and_csv_export_truncated_filters_and_modes_3d(order, rank, export_csv, filter_trim_zeros):
    """Generates and plots the Poisson filters for the 3D case, for a given rank and truncation method/value.

    Option to export as csv. CSV export is useful, for instance, to load and plot filters in Mathematica.

    .. note::
        Filter and mode csv files can be found in :code:`data/preprocess/components/`.

    :param int order: filter order (target iteration)
    :param int rank: how many ranks in decomposition (8 or less is usually enough)
    :param int export_csv: exporting the modes and filters as .csv.
    :param bool filter_trim_zeros: :code:`True` if you want to trim the already zeroed-out elements due to adaptive truncation
    """

    # =============== Options ===============

    # kernel Parameters
    dim = com.SolverDimension.D3
    itr = order
    truncation_method = com.TruncationMode.FIXED_THRESHOLD
    truncation_value = 1e-5 # safety code to prohibit varying size filters (solve, low rank kernel).
    # for visualization purposes however you can turn this off safely when low rank is not computed
    preserve_shape = True if truncation_method == com.TruncationMode.FIXED_THRESHOLD else False

    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 100.  # diffusivity - only for diffusion

    # solver
    solver_type = com.PoissonSolverType.INVERSE
    # solver_type = PoissonSolverType.FORWARD

    # kernel/solver parameters
    kernel_type = com.PoissonKernelType.UNIFIED # compatible with shader code with negative divergence scale
    # kernel_type = PoissonKernelType.STANDARD # General format with no assumption about the input
    zero_init = gen.is_ok_to_zero_init(solver_type=solver_type) # cannot use zero init for diffusion
    # warm_start = False  # True : forces to use STANDARD kernel, otherwise UNIFIED kernel
    # kernel_type, zero_init = i_want_to_warm_start(warm_start=warm_start, solver_type=solver_type)

    # Decomposition Methods
    decomp_method = com.DecompMethod.SYM_CP_3D # leave it as default for 3D
    # Output Format
    output_format = com.OutputFormat.ABSORBED_FILTERS
    # output_format = OutputFormat.ALL_COMPONENTS

    opts_solver = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init)

    # --- functions Parameters ----
    # both for diffusion and pressure
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)
    opts_kernel = com.OptionsKernel(kernel_type=kernel_type,
                                    itr=itr,
                                    dx=dx, dt=dt, kappa=kappa, alpha=alpha, beta=beta,
                                    clear_memo=True)

    opts_reduction = com.OptionsReduction(decomp_method=decomp_method,
                                          reduce=True,
                                          use_separable_filters=True,
                                          rank=rank,
                                          truncation_method=truncation_method,
                                          truncation_value=truncation_value,
                                          preserve_shape=True if truncation_method == com.TruncationMode.FIXED_THRESHOLD
                                          else False,
                                          output_format=output_format)

    opts_input = com.OptionsDataMatrix(shape=(11, 11, 11),
                                       mode=com.DataMatrixMode.RANDOMIZE_INT,
                                       rand_range=(1, 10),  # integer, excluding the high
                                       const_input=1.0)
    #
    opts_boundary = com.OptionsBoundary(enforce=False,
                                        condition=com.BoundaryType.NEUMANN_EDGE,
                                        obj_collide=False,
                                        post_solve_enforcement=False,
                                        val=0,
                                        dynamic_padding=False,
                                        padding_size=com.get_default_padding(),
                                        left_wall=True,
                                        right_wall=True,
                                        up_wall=True,
                                        down_wall=True)

    # packing all..
    opts = com.OptionsGeneral(solver=opts_solver, kernel=opts_kernel, reduction=opts_reduction,
                              boundary=opts_boundary, input=opts_input)

    # =============== Solve and Output ===============
    # NOTE --> make sure you use rank_filter_reorder=True for 3D to get [rank, filter] tuple order
    all_filters, all_modes, low_rank, full_kernel, safe_rank = \
        dec.poisson_compute_modes_trim_filters_3d(opts=opts, rank_filter_reorder=True, filter_trim_zeros=filter_trim_zeros,
                                                  preserve_shape=preserve_shape)

    # =============== Export CSV ===============
    if export_csv:
        io.export_components_csv(dim=dim, solver_type=solver_type, order=itr, safe_rank=safe_rank,
                                 filters=all_filters, modes=all_modes, full_kernel=full_kernel)

    # =============== Plots ===============

    # general kernel info
    specs_info = f' Poisson Filters: 3D, {solver_type.name}, {kernel_type.name}, Order={itr}, Rank={opts.reduction.rank}, ' \
        f'{opts.reduction.truncation_method}, TruncationValue={opts.reduction.truncation_value}'

    # plot options..
    opts_plots = com.OptionsPlots(show_values=False, no_ticks=True, aspect_ratio=1.,
                                  cbar_orientation="horizontal", line_widths=0.0, fmt=".0e",
                                  plt_show=False, beautify=True, cmap="rocket_r", cbar=False,
                                  interp_1d=True, interp_1d_res=10)

    vis.plot_filters(opts_plots=opts_plots, safe_rank=opts.reduction.rank, filters=all_filters, specs_info=specs_info)

    plt.show()


def quick_process_filters_3d(single_itr_num, safe_rank, filter_trim_zeros, filter_gen_method,
                             opts, preserve_shape=True):
    """
    :param int single_itr_num: target iteration
    :param int safe_rank: how many ranks in decomposition (8 or less is usually enough)
    :param bool filter_trim_zeros: :code:`True` if you want to trim the already zeroed-out elements due to adaptive truncation
    :param str filter_gen_method: :code:`'load'` or :code:`'generate'`
    :param OptionsGeneral opts:
    :param bool preserve_shape: if :code:`True` keep the original shape and fill them with zeros,
        else return the shrunk filter (Default= :code:`True`)
    :return: processed filters and filter titles
    """

    if filter_gen_method == 'load':
        database_max_itr = 100  # ---> NOTE! this database must exist
        database = io.load_filter_database(max_itr=database_max_itr, dx=opts.kernel.dx, kappa=opts.kernel.kappa,
                                           dim=opts.solver.dim, solver_type=opts.solver.solver_type,
                                           kernel_type=opts.kernel.kernel_type)

        assert len(database) >= single_itr_num
        name = "arr_{}".format(single_itr_num - 1)
        filters = database[name]
        # filters = database[single_itr_num - 1]

    elif filter_gen_method == 'generate':
        # generating filters fresh
        filters, low_rank, full_kernel, safe_rank = dec.poisson_filters_3d(opts=opts, rank_filter_reorder=True,
                                                                           preserve_shape=preserve_shape)
    else:
        assert False, "Unknown filter generation method"

    # extra process to make them nice and ready
    processed_filters = []
    filter_titles = []
    for r in range(1, safe_rank + 1):
        # filter
        ranked_filter = filters[r - 1]

        if filter_gen_method == 'generate':
            ranked_filter = ranked_filter[np.nonzero(ranked_filter)] if filter_trim_zeros else ranked_filter
        else:
            assert filter_gen_method == 'load', "Unknown filter generation method"

        if np.size(ranked_filter) == 0:
            ranked_filter = np.zeros(5)  # fixed number just to make the interpolation work, "if" we are interpolating
        ranked_filter = ranked_filter.reshape(-1, 1)
        processed_filters.append(ranked_filter)  # assuming horizontal and vertical filters are identical
        # title: show min/max range
        max_val = max(np.abs(np.max(ranked_filter)), np.abs(np.min(ranked_filter)))
        filter_titles.append(f'Rank {r}, $\pm${max_val:.1e} \n')

    return processed_filters, filter_titles


def demo_plot_filters_adaptive_truncation_beautified_3d(order=60, rank=8, filter_gen_method='load',
                                                        use_custom_style=True, visualize_truncation=True):
    """Plot beautified filters with added visuals for the truncated areas. This works only for
    :code:`com.TruncationMode.FIXED_THRESHOLD` (adaptive truncation).

    :param int order: filter order (target iteration)
    :param int rank: how many ranks in decomposition (8 or less is usually enough)
    :param str filter_gen_method: :code:`'load'` or :code:`'generate'`. A valid database containing the filters must
        already exist in :code:`data/preprocess/filters/` if choosing to load; else generate them fresh.
    :param bool use_custom_style: allows for custom background
    :param bool visualize_truncation: draw the threshold
    """
    dim = com.SolverDimension.D3
    decomp_method = com.DecompMethod.SYM_CP_3D

    # =============== Options ===============

    # solver
    solver_type = com.PoissonSolverType.INVERSE
    # solver_type = PoissonSolverType.FORWARD

    # kernel/solver parameters
    warm_start = False  # True : forces to use STANDARD kernel, otherwise UNIFIED kernel
    kernel_type, zero_init = gen.i_want_to_warm_start(warm_start=warm_start, solver_type=solver_type)
    opts_solver = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init)

    # both for diffusion and pressure
    itr = order
    # reduction
    truncation_method = com.TruncationMode.FIXED_THRESHOLD
    truncation_value = 1e-2  # truncate everything below this value if FIXED_THRESHOLD
    # NOTE: only use preserve_shape when generating fresh... does not apply to loaded filters from database.
    # safety code to prohibit varying size filters (solve, low rank kernel).
    # for visualization purposes however you can turn this safely off when low rank is not computed
    preserve_shape = True if truncation_method == com.TruncationMode.FIXED_THRESHOLD else False
    filter_trim_zeros = True

    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 1  # diffusivity - only for diffusion
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)
    opts_kernel = com.OptionsKernel(kernel_type=kernel_type, itr=itr, dx=dx, dt=dt, kappa=kappa,
                                    alpha=alpha, beta=beta, clear_memo=True)

    opts_reduction = com.OptionsReduction(decomp_method=decomp_method,
                                          reduce=True,
                                          use_separable_filters=True,
                                          rank=rank,
                                          truncation_method=truncation_method,
                                          truncation_value=truncation_value,
                                          preserve_shape=True if truncation_method == com.TruncationMode.FIXED_THRESHOLD
                                          else False)
    # packing all..
    opts = com.OptionsGeneral(solver=opts_solver, kernel=opts_kernel, reduction=opts_reduction,
                              boundary=com.OptionsBoundary(), input=com.OptionsDataMatrix())

    # =============== Compute OR Load from file ===============
    safe_rank = rank
    processed_filters, filter_titles = quick_process_filters_3d(single_itr_num=itr, safe_rank=safe_rank,
                                                                filter_trim_zeros=filter_trim_zeros,
                                                                filter_gen_method=filter_gen_method,
                                                                opts=opts, preserve_shape=preserve_shape)

    # =============== Plots ===============

    constrained_layout = False
    swap_axes = True  # for plotting vertical 1d filters arrays
    if use_custom_style:
        vis.use_style(style_name="dark_background")

    # generating layout and titles...

    # general kernel info
    _specs_info = f' (Itr={itr}, Rank={safe_rank}, Trunc={opts.reduction.truncation_value})'

    # plot options..
    opts_plots = com.OptionsPlots(show_values=False, no_ticks=True, aspect_ratio=1.,
                                  cbar_orientation="horizontal", line_widths=0.0, fmt=".0e",
                                  plt_show=False, beautify=True, cmap="rocket_r", cbar=False,
                                  interpolate=True, interp_1d=True, interp_1d_res=100)

    # ===== plot all filters =====

    def get_key_rank(r):
        assert 0 < r <= 8
        return 'R' + str(r + 1) + '_'

    def make_layout():
        # ----- init layout -------
        layout_str = []
        this_row = []
        for r in range(safe_rank):
            key_rank = get_key_rank(r + 1)
            this_row.append(key_rank + '1d')
            this_row.append(key_rank + '2d')
        # sanity check
        layout_cols = len(this_row)
        assert layout_cols == 2 * safe_rank
        layout_str.append(this_row)

        figsize = (18, 12)

        return layout_str, figsize

    def assign_subplots():
        axes_grid = vis.add_grid_subplots_mosaic(fig=fig, layout_str=layout_str, subplot_kw=subplot_kw,
                                                 grid_spec_kw=gridspec_kw)
        axes_1d = []
        axes_2d = []
        for r in range(safe_rank):
            key_rank = get_key_rank(r + 1)
            axes_1d.append(axes_grid[key_rank + '1d'])
            axes_2d.append(axes_grid[key_rank + '2d'])

        return axes_1d, axes_2d

    # ===================== Final plotting calls ==================
    layout_str, figsize = make_layout()
    subplot_kw = {'frameon': False}
    gridspec_kw = None
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    axes_1d, axes_2d = assign_subplots()
    assert len(axes_1d) == safe_rank
    assert len(axes_2d) == safe_rank

    vis.plot_filters_3d_advanced(processed_filters=processed_filters, axes_1d=axes_1d, axes_2d=axes_2d, fig=fig,
                                 visualize_truncation=visualize_truncation, truncation_method=truncation_method,
                                 truncation_value=truncation_value, filter_titles=filter_titles, swap_axes=swap_axes,
                                 opts_plots=opts_plots)

    title = f'$\delta = ${truncation_value:.0e}'
    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.show()


if __name__ == '__main__':
    # UNCOMMENT TO RUN THE DEMOS...

    # 1. Compute and plot all components of 3D filter decomposition : CSV export is available,
    # UNCOMMENT------------------------------------------------
    # plot_and_csv_export_truncated_filters_and_modes_3d(order=30, rank=8, export_csv=True, filter_trim_zeros=True)

    # 2. Plot beautified adaptive truncation of filters
    # UNCOMMENT------------------------------------------------
    demo_plot_filters_adaptive_truncation_beautified_3d(order=60, rank=8)
