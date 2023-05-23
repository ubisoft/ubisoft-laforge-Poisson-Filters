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


def plot_and_csv_export_truncated_filters_and_modes_2d(order=50, rank=8, export_csv=True, filter_trim_zeros=True):
    """Compute the full Poisson kernel, reduce it based on a desired rank, and plot \
    the full kernel, the reduced one, the modes and separable filters.

    Option to export as csv. CSV export is useful, for instance, to load and plot filters in Mathematica.

    .. note::
        Filter and mode csv files can be found in :code:`data/preprocess/components/`.

    :param int order: filter order (target iteration)
    :param int rank: how many ranks in decomposition (8 or less is usually enough)
    :param bool export_csv: exporting the modes and filters as .csv
    :param bool filter_trim_zeros: :code:`True` if you want to trim the already zeroed-out elements due to adaptive truncation
    """
    dim = com.SolverDimension.D2

    # =============== Options ===============

    # solver
    solver_type = com.PoissonSolverType.INVERSE
    # solver_type = PoissonSolverType.FORWARD

    # kernel/solver parameters
    warm_start = False # True : forces to use STANDARD kernel, otherwise UNIFIED kernel
    kernel_type, zero_init = gen.i_want_to_warm_start(warm_start=warm_start, solver_type=solver_type)
    opts_solver = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init)

    # both for diffusion and pressure
    itr = order
    # reduction
    truncation_method = com.TruncationMode.FIXED_THRESHOLD
    truncation_value = 1e-6
    include_low_rank_kernel = False
    # safety code to prohibit varying size filters (solve, low rank kernel).
    # for visualization purposes however you can turn this off safely when low rank is not computed
    preserve_shape = True if truncation_method == com.TruncationMode.FIXED_THRESHOLD else False

    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 1  # diffusivity - only for diffusion
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)
    opts_kernel = com.OptionsKernel(kernel_type=kernel_type, itr=itr, dx=dx, dt=dt, kappa=kappa,
                                    alpha=alpha, beta=beta, clear_memo=True)

    opts_reduction = com.OptionsReduction(decomp_method=com.DecompMethod.SVD_2D,
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

    # =============== Compute ===============

    # Full Poisson kernel solve
    poisson_kernel = gen.poisson_kernel_2d(opts=opts)

    # reduce kernel and its separable filters
    safe_rank = dec.rank_safety_clamp_2d(itr=itr, rank=opts.reduction.rank, zero_init=zero_init)
    U, S, VT, low_rank = dec.poisson_svd_2d(P=poisson_kernel, rank=safe_rank)
    v_hor, v_ver, safe_rank = dec.compute_separable_filters_truncated_2d(U=U, S=S, VT=VT,
                                                                         rank=safe_rank,
                                                                         trunc_method=truncation_method,
                                                                         trunc_factor=opts.reduction.truncation_value,
                                                                         preserve_shape=preserve_shape)
    if include_low_rank_kernel:
        low_rank_truncated = dec.compute_low_rank_kernel_from_filters_2d(hor=v_hor, ver=v_ver, safe_rank=safe_rank)
    else:
        low_rank_truncated = None

    all_modes = []
    all_filters = []
    for r in range(1, safe_rank + 1):
        # mode
        nth_mode = dec.compute_nth_kernel_mode_2d(hor=v_hor, ver=v_ver, rank=r)
        all_modes.append(nth_mode)
        # filter
        ranked_filter = v_hor[r - 1]
        ranked_filter = ranked_filter[np.nonzero(ranked_filter)] if filter_trim_zeros else ranked_filter
        if np.size(ranked_filter) == 0:
            ranked_filter = np.zeros(5) # fixed number just to make the interpolation work, "if" we are interpolating
        all_filters.append(ranked_filter)  # assuming horizontal and vertical filters are identical

    # =============== Export CSV ===============
    if export_csv:
        io.export_components_csv(dim=dim, solver_type=solver_type, order=itr, safe_rank=safe_rank,
                                filters=all_filters, modes=all_modes, full_kernel=poisson_kernel)

    # =============== Plots ===============

    _show_values = False
    _no_ticks = True
    _aspect_ratio = 1  # y-unit to x-unit ratio
    _linewidths = 0.0

    # generating layout and titles...

    # general kernel info
    _specs_info = f' {solver_type.name}, {kernel_type.name}, Order={itr}, Rank={opts.reduction.rank}, ' \
        f'{opts.reduction.truncation_method}, TruncationValue={opts.reduction.truncation_value}'

    # plot options..
    opts_plots = com.OptionsPlots(show_values=False, no_ticks=True, aspect_ratio=1.,
                                  cbar_orientation="horizontal", line_widths=0.0, fmt=".0e",
                                  plt_show=False, beautify=True, cmap="rocket_r", cbar=False,
                                  interpolate=True, interp_1d=True, interp_1d_res=100)

    # ===== plot all modes and filters =====
    plot_modes_filters_2d(opts_plots=opts_plots, safe_rank=safe_rank,
                          all_modes=all_modes, all_filters=all_filters, specs_info=_specs_info)

    # ===== full and reduce kernels =====
    opts_plots.cbar = True
    opts_plots.fmt = ".2e"
    opts_plots.cmap = "rocket" if kernel_type == com.PoissonKernelType.STANDARD \
                                  and solver_type == com.PoissonSolverType.INVERSE else "rocket_r"
    # poisson kernel
    vis.plot_matrix_2d_advanced(M=poisson_kernel, title=f'Full Poisson Kernel Order={itr}', opts=opts_plots)
    # reduced kernel
    vis.plot_matrix_2d_advanced(M=low_rank, title=f'Reduced Poisson Kernel' + _specs_info, opts=opts_plots)
    # reduced kernel with truncation
    if low_rank_truncated is not None:
        vis.plot_matrix_2d_advanced(M=low_rank_truncated, title=f'Truncated Reduced Poisson Kernel' + _specs_info, opts=opts_plots)
    # 3D full kernel
    vis.plot_matrix_3d(M=poisson_kernel, title=f'Full Poisson Kernel Order={itr}', opts=opts_plots)

    plt.show()


def plot_modes_filters_2d(opts_plots, all_modes, all_filters, safe_rank, specs_info):
    """
    :param OptionsPlots opts_plots:
    :param List[ndarray] all_modes:
    :param List[ndarray] all_filters:
    :param int safe_rank:
    :param str specs_info:
    """
    # grid layout to plot multi modes
    _layout = vis.compute_layout(columns=4, num_elements=safe_rank)

    # 2d mode plots
    opts_subplots_modes = com.OptionsSubPlots(layout=_layout)
    opts_subplots_filters = com.OptionsSubPlots(layout=_layout)

    _subplots_modes = []
    _subplots_filters = []
    vis.init_subplots(sub_plots=_subplots_modes, opts_subplots=opts_subplots_modes)
    vis.init_subplots(sub_plots=_subplots_filters, opts_subplots=opts_subplots_filters)

    for i in range(safe_rank):
        # mode
        mode = all_modes[i]
        # compute a compact range of +- value where we take tha largest of min and max values in the matrix
        max_val = max(np.abs(np.max(mode)), np.abs(np.min(mode)))
        vis.add_subplot(M=mode,
                        title=f'Mode {i + 1}, $\pm${max_val:.0e}',
                        sub_plots=_subplots_modes, opts_subplots=opts_subplots_modes)
        # filter
        filtr = all_filters[i]
        # compute a compact range of +- value where we take tha largest of min and max values in the matrix
        max_val = max(np.abs(np.max(filtr)), np.abs(np.min(filtr)))
        vis.add_subplot(M=filtr,
                        title=f'Rank {i + 1}, $\pm${max_val:.0e}',
                        sub_plots=_subplots_filters, opts_subplots=opts_subplots_filters)

    # subplot options
    vis.plot_matrix_grid(G=_subplots_modes, title=specs_info, projection='2d',
                         opts_general=opts_plots, opts_subplots=opts_subplots_modes)

    # ---------- HARD CODED for paper ---------
    opts_plots.interp_3d = False # force overwriting the options
    # ---------- HARD CODED for paper ---------
    # 3d mode plots
    vis.plot_matrix_grid(G=_subplots_modes, title=specs_info, projection='3d',
                         opts_general=opts_plots, opts_subplots=opts_subplots_modes)
    # ranked filters
    vis.plot_1d_array_grid(G=_subplots_filters, title=specs_info,
                           opts_general=opts_plots, opts_subplots=opts_subplots_filters)


def demo_compare_standard_and_unified_kernels_unreduced_2d(order=50):
    """Visually inspecting the differences and similarities between two ways of kernel generation, namely
    :code:`Standard` and :code:`Unified`. Both are full kernels (unreduced) of the same order (same target iteration).

    For more info see :func:`helper.common.PoissonKernelType`.

    .. note::
        Observe how the two kernels are the inverse of each other when \
        :code:`solver_type = com.PoissonSolverType.INVERSE`, while they are the same for :code:`FORWARD`. \
        When solving the *inverse* Poisson equation, you should multiply the right hand side :math:`b` in :math:`Ax=b` \
        by :math:`-1`.

    :param int order: filter order (target iteration)
    """

    dim = com.SolverDimension.D2

    # =============== Options ===============

    solver_type = com.PoissonSolverType.INVERSE
    # solver_type = com.PoissonSolverType.FORWARD

    # standard kernel
    kernel_type_standard = com.PoissonKernelType.STANDARD
    zero_init_standard = True
    opts_solver_standard = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init_standard)

    # unified kernel
    kernel_type_unified = com.PoissonKernelType.UNIFIED
    zero_init_unified = True
    opts_solver_unified = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init_unified)

    # =============== functions Parameters ===============
    # both for diffusion and pressure
    itr = order
    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 100.  # diffusivity - only for diffusion
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)

    # standard
    opts_kernel_standard = com.OptionsKernel(kernel_type=kernel_type_standard, itr=itr, dx=dx, dt=dt, kappa=kappa,
                                             alpha=alpha, beta=beta, clear_memo=True)

    opts_standard = com.OptionsGeneral(solver=opts_solver_standard, kernel=opts_kernel_standard,
                                       reduction=com.OptionsReduction(), boundary=com.OptionsBoundary(),
                                       input=com.OptionsDataMatrix())

    # unified
    opts_kernel_unified = com.OptionsKernel(kernel_type=kernel_type_unified, itr=itr, dx=dx, dt=dt, kappa=kappa,
                                            alpha=alpha, beta=beta, clear_memo=True)

    opts_unified = com.OptionsGeneral(solver=opts_solver_unified, kernel=opts_kernel_unified,
                                      reduction=com.OptionsReduction(), boundary=com.OptionsBoundary(), input=com.OptionsDataMatrix())

    poisson_standard = gen.poisson_kernel_2d(opts=opts_standard)
    poisson_unified = gen.poisson_kernel_2d(opts=opts_unified)

    # =============== Plots ===============

    _specs_info = f'{solver_type.name} Poisson Kernel Types' + f' Order={itr}'

    opts_plots = com.OptionsPlots(show_values=False, no_ticks=True, aspect_ratio=1., cbar=True,
                                  cbar_orientation="horizontal",
                                  line_widths=0.0, cmap="rocket", fmt=".2f", plt_show=False)

    opts_subplots = com.OptionsSubPlots(layout=(1, 2))

    sub_plots = []
    vis.init_subplots(sub_plots=sub_plots, opts_subplots=opts_subplots)

    vis.add_subplot(M=poisson_standard, title=f'Standard Kernel, {solver_type.name},' + f' Order={itr}',
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    vis.add_subplot(M=poisson_unified, title=f'Unified Kernel, {solver_type.name},' + f' Order={itr}',
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # plot all
    vis.plot_matrix_grid(G=sub_plots, title=_specs_info, projection='2d',
                         opts_general=opts_plots, opts_subplots=opts_subplots)

    plt.show()


if __name__ == '__main__':
    # UNCOMMENT TO RUN THE DEMOS...

    # 1. Compute the full Poisson kernel, reduce it based on a desired rank, and plot
    # the full kernel, the reduced one, the modes and separable filters. Option to export as csv.
    # UNCOMMENT------------------------------------------------
    plot_and_csv_export_truncated_filters_and_modes_2d(order=50, rank=8, export_csv=True, filter_trim_zeros=True)

    # 2. Comparing standard and unified kernel types
    # UNCOMMENT------------------------------------------------
    # demo_compare_standard_and_unified_kernels_unreduced_2d(order=50)


