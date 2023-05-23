""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`

    .. _convergence_notes:

    Technicals
    ============

    - *Adaptive truncation*:
        Adaptive truncation is based on the ceiling of filter sizes, i.e. maximum filter size needed for a given
        rank group given a fixed filter value threshold. \
        We can further optimize this by doing per rank filter truncation per iteration as there might \
        certain ranks whose filters can still have elements below the threshold. \
        In general this is a minor optimization and does not have a
        very significant effect on the performance, but it can make the implementation more complicated, \
        so we did not use it in our results in the paper.

        See :func:`demo_adaptive_truncation_analysis_2d` for adaptive truncation per individual rank as well as \
        Poisson filters memory footprint.

    - When solving the *inverse* Poisson equation, you can generate Poisson kernels with :math:`\\alpha=1` \
        then just scale the right hand side :math:`b` in :math:`Ax=b`.
"""
import numpy as np
import matplotlib.pyplot as plt
import src.functions.analytics as an
import src.helper.commons as com
import src.helper.visualizer as vis
import src.functions.generator as gen
import src.functions.mathlib as mlib
import sys
sys.path.append('../../')


# ============================== KEEP! ==============================
def demo_3methods_comparison_no_bc_2d(order=30, rank=8, use_full_poisson_kernel=True):
    """*Sanity check*: compare Jacobi solutions, one from the flattened :math:`Ax=b` (*A.K.A. vector-form*) and \
    one for the *matrix-form*. Also compare them with the Poisson kernel solution achieved \
    through convolution :math:`L*X=B`.

    This is without Neumann boundary enforcement. This effectively means an *infinite domain* setup. \
    Infinite domain is achieved by dynamically padding the input data based on the desired \
    Jacobi iteration. In other words, we pad the data with sufficiently large blocks so the \
    effects of the wall boundaries never reach inside the domain, where the original much \
    smaller input data exists.

    .. note::
        The Poisson kernel (and subsequently its separable filters) are computed based on the *matrix-form*, \
        so for an infinite domain without any boundary treatment the solution using the Poisson kernel and the \
        one from the *matrix-form* of Jacobi must match to the machine precision.

    In case of using the full kernel, there is no reduction, so we expect an exact match \
    between Jacobi *matrix-form* and the Poisson filters solution. You can however experiment using Poisson \
    filters by setting :code:`use_full_poisson_kernel=False`.

    :param int order: filter order (target iteration)
    :param int rank: how many ranks in decomposition (8 or less is usually enough).
        Only used when in reduction mode.
    :param bool use_full_poisson_kernel: if :code:`True` there is no reduction, full kernel convolution.
        If :code:`False`, we are either using the reduced square Poisson kernel, or
        just using the separable filters. Look inside the code for comments on :code:`reduce` and
        :code:`use_separable_filters` variables.
    """
    dim = com.SolverDimension.D2

    # =============== Options ===============

    # solver
    solver_type = com.PoissonSolverType.INVERSE
    # solver_type = PoissonSolverType.FORWARD

    # kernel/solver parameters
    kernel_type = com.PoissonKernelType.STANDARD
    zero_init = True

    print(f'kernel type: {kernel_type}')
    print(f'solver type: {solver_type}')
    print(f'zero init: {zero_init}')
    opts_solver = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init)

    # =============== functions Parameters ===============
    # both for diffusion and pressure
    itr = order
    domain_size = (21, 21)  # use ODD numbers
    assert domain_size[0] % 2 != 0 and domain_size[1] % 2 != 0, "Use odd domain shape. Data needs ot be symmetrical"

    # 'reduce': if True use reduced kernel, else use the full unreduced kernel
    # 'use_separable_filters': if False use the full 'reduced' kernel, if 'reduce=False' this will be ignored
    if use_full_poisson_kernel:
        reduce = False
        use_separable_filters = False
    else:
        reduce = True
        use_separable_filters = True  # if set False, a 'reduced' kernel will be used, else, filters will be used.

    truncation_method = com.TruncationMode.PERCENTAGE  # only used for filters
    truncation_value = 0.  # 0 means no truncation, otherwise values in [0,1] for percentage

    poisson_choice_str = 'Poisson Full Kernel' + (' (reduced)' if reduce else '')
    if reduce and use_separable_filters:
        poisson_choice_str = 'Poisson Filter'
    if reduce:
        poisson_choice_str += f' rank {rank}'

    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 100.  # diffusivity - only for diffusion
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)
    opts_kernel = com.OptionsKernel(kernel_type=kernel_type, itr=itr, dx=dx, dt=dt, kappa=kappa,
                                    alpha=alpha, beta=beta, clear_memo=True)
    # reduction
    opts_reduction = com.OptionsReduction(decomp_method=com.DecompMethod.SVD_2D,
                                          reduce=reduce,
                                          use_separable_filters=use_separable_filters,
                                          rank=rank,
                                          truncation_method=truncation_method,
                                          truncation_value=truncation_value,
                                          preserve_shape=True if truncation_method == com.TruncationMode.FIXED_THRESHOLD
                                          else False)
    # example data matrix
    opts_input = com.OptionsDataMatrix(shape=domain_size,
                                       mode=com.DataMatrixMode.RANDOMIZE_FLOAT_UNIFORM,
                                       rand_range=(1, 10),  # integer, excluding the high
                                       const_input=0.)
    # boundary
    opts_boundary = com.OptionsBoundary(enforce=False, val=1, padding_size=com.get_default_padding())

    # packing all..
    opts_general = com.OptionsGeneral(solver=opts_solver, kernel=opts_kernel, reduction=opts_reduction,
                                      boundary=opts_boundary, input=opts_input)

    # =============== Solve ===============
    print('Solving..')
    error_abs_j_vs_j, error_rel_j_vs_j, error_abs_jv_vs_pk, error_rel_jv_vs_pk, error_abs_jm_vs_pk, error_rel_jm_vs_pk, \
    jacobi_solution_matrix_form, jacobi_solution_vector_form, poisson_solution, data \
        = an.compare_jacobi_3methods_2d(opts=opts_general)

    # =============== Plots ===============
    print('Plotting..')
    specs_info = f'Jacobi Matrix Form vs Jacobi Vector Form vs ' + poisson_choice_str + f' (Itr={itr}, Data={opts_input.shape})'

    opts_plots = com.OptionsPlots(show_values=False, no_ticks=True, aspect_ratio=1., cbar=True,
                                  cbar_orientation="horizontal",
                                  line_widths=0.0, cmap="rocket", fmt=".2f", plt_show=False)

    opts_subplots = com.OptionsSubPlots(layout=(2, 3))

    #data
    vis.plot_matrix_2d_advanced(M=data, title="Dynamically Padded Input Matrix", opts=opts_plots)

    # using divergent colormap for the error plots
    manual_cmaps = ["rocket", "rocket", "rocket", "twilight", "twilight", "twilight"]

    sub_plots = []
    vis.init_subplots(sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Jacobi solution vector-form
    vis.add_subplot(M=jacobi_solution_vector_form, title="Jacobi Solution Vector Form",
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Jacobi solution matrix-form
    vis.add_subplot(M=jacobi_solution_matrix_form, title="Jacobi Solution Matrix Form",
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Full Poisson kernel solution
    vis.add_subplot(M=poisson_solution, title="Poisson Solution",
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Rel. Err. Jacobi matrix vs vector-form
    vis.add_subplot(M=error_rel_j_vs_j, title="Jacobi Matrix vs vector-form Rel. Err. %",
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Rel. Err. Jacobi matrix-form vs poisson kernel
    vis.add_subplot(M=error_rel_jm_vs_pk, title="Jacobi Matrix Form vs Poisson Solution Rel. Err. %",
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Rel. Err. Jacobi vector-form vs poisson kernel
    vis.add_subplot(M=error_rel_jv_vs_pk, title="Jacobi Vector Form vs Poisson Solution Rel. Err. %",
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # plot all
    vis.plot_matrix_grid(G=sub_plots, title=specs_info, projection='2d',
                         opts_general=opts_plots, opts_subplots=opts_subplots,
                         cmaps=manual_cmaps)

    # This line will act on behalf of all functions with plots.
    plt.show()


def demo_3methods_comparison_with_wall_neumann_bc_2d(order=20, rank=8):
    """*Sanity check*: compare Jacobi solutions, one from the flattened :math:`Ax=b` (*A.K.A. vector-form*) and \
    one for the *matrix-form*. Also compare them with the Poisson kernel solution achieved \
    through convolution :math:`L*X=B`.

    This is with Neumann boundary enforcement. Only for walls. No complex object.

    .. note::
        The solutions to Poisson filters and Jacobi *matrix-form* are supposed to match, but we should
        expect slight difference between Jacobi *matrix-form* and Jacobi *vector-form* due to the way that
        they deal with corner domain boundary treatment. The results in the paper are achieved with a \
        GPU implementation, where we use Jacobi *matrix-form*, so as long as \
        Poisson filters and Jacobi *matrix-form* match we are good.

    :param int order: filter order (target iteration)
    :param int rank: how many ranks in decomposition (8 or less is usually enough).
        Only used when in reduction mode.
    """

    dim = com.SolverDimension.D2

    # =============== Options ===============

    # solver
    solver_type = com.PoissonSolverType.INVERSE
    # solver_type = PoissonSolverType.FORWARD

    # kernel/solver parameters

    kernel_type = com.PoissonKernelType.STANDARD
    zero_init = True

    print(f'kernel type: {kernel_type}')
    print(f'solver type: {solver_type}')
    print(f'zero init: {zero_init}')
    opts_solver = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init)

    # =============== functions Parameters ===============
    # both for diffusion and pressure
    itr = order
    truncation_method = com.TruncationMode.PERCENTAGE
    truncation_value = 0.
    domain_size = (51, 51) # use ODD numbers
    assert domain_size[0] % 2 != 0 and domain_size[1] % 2 != 0, "Use odd domain shape. Data needs ot be symmetrical"
    obj_collide = True  # if False only walls will be taken into account

    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 100.  # diffusivity - only for diffusion
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)
    opts_kernel = com.OptionsKernel(kernel_type=kernel_type, itr=itr, dx=dx, dt=dt, kappa=kappa,
                                    alpha=alpha, beta=beta, clear_memo=True)
    # reduction
    opts_reduction = com.OptionsReduction(decomp_method=com.DecompMethod.SVD_2D,
                                          reduce=True,
                                          use_separable_filters=True,
                                          rank=rank,
                                          truncation_method=truncation_method,
                                          truncation_value=truncation_value,
                                          preserve_shape=True if truncation_method == com.TruncationMode.FIXED_THRESHOLD
                                          else False)
    # example data matrix
    opts_input = com.OptionsDataMatrix(shape=domain_size,
                                       mode=com.DataMatrixMode.RANDOMIZE_FLOAT_UNIFORM,
                                       rand_range=(-1, 1),  # integer, excluding the high
                                       const_input=0.)

    # boundary
    opts_boundary = com.OptionsBoundary(enforce=True,
                                        condition=com.BoundaryType.NEUMANN_EDGE,
                                        obj_collide=obj_collide,
                                        post_solve_enforcement=True,
                                        val=0,
                                        dynamic_padding=False,
                                        padding_size=com.get_default_padding(),
                                        left_wall=True,
                                        right_wall=True,
                                        up_wall=True,
                                        down_wall=True,
                                        front_wall=True, # does not matter in 2D
                                        back_wall=True, # does not matter in 2D
                                        )

    # packing all..
    opts_general = com.OptionsGeneral(solver=opts_solver, kernel=opts_kernel, reduction=opts_reduction,
                                      boundary=opts_boundary, input=opts_input)

    # =============== Solve ===============

    error_abs_j_vs_j, error_rel_j_vs_j, error_abs_jv_vs_pk, error_rel_jv_vs_pk, error_abs_jm_vs_pk, error_rel_jm_vs_pk, \
    jacobi_solution_matrix_form, jacobi_matrix_form_residuals, jacobi_solution_vector_form, jacobi_vector_form_residuals, \
    poisson_solution, poisson_residual, data \
        = an.compare_jacobi_3methods_neumann_2d(opts=opts_general)

    # =============== Plots ===============

    specs_info = f'Jacobi Matrix Form vs Jacobi Vector Form vs Poisson Kernel (Itr={itr}, Data={opts_input.shape})'

    opts_plots = com.OptionsPlots(show_values=False, no_ticks=True, aspect_ratio=1., cbar=True,
                                  cbar_orientation="horizontal",
                                  line_widths=0.0, cmap="rocket", fmt=".2f", plt_show=False)

    opts_subplots = com.OptionsSubPlots(layout=(2, 3))
    # using divergent colormap for the error plots
    manual_cmaps = ["rocket", "rocket", "rocket", "twilight", "twilight", "twilight"]

    sub_plots = []
    vis.init_subplots(sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Jacobi solution vector-form
    vis.add_subplot(M=jacobi_solution_vector_form, title="Jacobi Solution Vector Form",
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Jacobi solution matrix-form
    vis.add_subplot(M=jacobi_solution_matrix_form, title="Jacobi Solution Matrix Form",
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Full Poisson kernel solution
    vis.add_subplot(M=poisson_solution, title="Poisson Solution",
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Abs. Err. Jacobi matrix vs vector-form
    avg_err_abs_j_vs_j = np.mean(error_abs_j_vs_j)  # get it before possible manipulation..
    vis.add_subplot(M=error_abs_j_vs_j, title="Jacobi Matrix vs Vector Form Abs. Err. " +
                                          ', mean {:3.2f}'.format(avg_err_abs_j_vs_j),
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Abs. Err. Jacobi matrix-form vs poisson kernel
    avg_err_abs_jm_vs_pk = np.mean(error_abs_jm_vs_pk)  # get it before possible manipulation..
    vis.add_subplot(M=error_abs_jm_vs_pk, title="Jacobi Matrix Form vs Poisson Kernel Abs. Err. " +
                                            ', mean {:3.2f}'.format(avg_err_abs_jm_vs_pk),
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Abs. Err. Jacobi vector-form vs poisson kernel
    avg_err_abs_jv_vs_pk = np.mean(error_abs_jv_vs_pk)  # get it before possible manipulation..
    vis.add_subplot(M=error_abs_jv_vs_pk, title="Jacobi Vector Form vs Poisson Kernel Abs. Err. " +
                                            ', mean {:3.2f}'.format(avg_err_abs_jv_vs_pk),
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # plot all
    vis.plot_matrix_grid(G=sub_plots, title=specs_info, projection='2d',
                         opts_general=opts_plots, opts_subplots=opts_subplots,
                         cmaps=manual_cmaps)

    #data
    vis.plot_matrix_2d_advanced(M=data, title="Input Data Matrix", opts=opts_plots)

    # --- Residuals
    fig = plt.figure()
    # Jacobi matrix-form residuals
    plt.semilogy(jacobi_matrix_form_residuals, label=f'Jaocbi Matrix Residuals')
    # Jacobi vector-form residuals
    plt.semilogy(jacobi_vector_form_residuals, label=f'Jaocbi Vector Residuals')
    plt.title('Residuals - Poisson Residual={:3.2f}'.format(poisson_residual))
    plt.legend()
    # This line will act on behalf of all functions with plots.
    plt.show()


def demo_adaptive_truncation_analysis_2d(order=100, rank=8):
    """Plot maximum rank needed per target iteration given a desired adaptive truncation value, \
    as well as the Poisson filters memory footprint.

    .. note::
        In practice little improvement on convergence is observed by using different ranks for different \
        adaptive truncation values. It is safe to use a sufficiently large rank for all target iterations. \
        This helps avoid complicating the implementation. A rank range of :math:`6 \\cdots 8` for 3D, and \
        :math:`1 \\cdots 4` for 2D is usually sufficient regardless of the truncation value.

    .. note::
        Note the sub-linear memory footprint of the filters as the target iteration grows.

    :param int order: filter order (target iteration)
    :param int rank: how many ranks in decomposition (8 or less is usually enough).
        Only used when in reduction mode.
    :return:
    """

    dim = com.SolverDimension.D2
    decomp_method = com.DecompMethod.SVD_2D if dim == com.SolverDimension.D2 else com.DecompMethod.SYM_CP_3D

    # =============== Options ===============

    # solver
    solver_type = com.PoissonSolverType.INVERSE
    # solver_type = PoissonSolverType.FORWARD

    # kernel/solver parameters
    kernel_type = com.PoissonKernelType.STANDARD
    zero_init = True
    print(f'kernel type: {kernel_type}')
    print(f'solver type: {solver_type}')
    print(f'zero init: {zero_init}')

    opts_solver = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init)

    # =============== functions Parameters ===============
    # both for diffusion and pressure

    itr = order
    # reduction
    reduce = True
    use_separable_filters = True
    truncation_method = com.TruncationMode.FIXED_THRESHOLD # FIXED_THRESHOLD: Adaptive truncation
    truncation_value = 5e-2 # does not matter, safe default, will be overwritten by the truncation list down below

    if dim == com.SolverDimension.D2:
        # ----------------- Interesting truncation thresholds---------------
        # should be in ascending order for the plot to work
        trunc_list = [1e-3, 1e-2, 1e-1]  # used to overwrite the options when looping over trunc vals

    elif dim == com.SolverDimension.D3:
        # ----------------- Interesting truncation thresholds---------------
        trunc_list = [0, 1e-3, 1e-2]  # used to overwrite the options when looping over trunc vals

    else:
        assert False, "Unknown Solver Dimension"

    # ----------------- The following does not really matter for filter analysis
    rand_range = (-1, 1)
    domain_size = (11, 11)  # use ODD numbers
    assert domain_size[0] % 2 != 0 and domain_size[1] % 2 != 0, "Use odd domain shape. Data needs ot be symmetrical"
    obj_collide = False  # if False only walls will be taken into account

    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 1.  # diffusivity - only for diffusion
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)
    opts_kernel = com.OptionsKernel(kernel_type=kernel_type, itr=itr, dx=dx, dt=dt, kappa=kappa,
                                    alpha=alpha, beta=beta, clear_memo=True)

    # reduction
    opts_reduction = com.OptionsReduction(decomp_method=decomp_method,
                                          reduce=reduce,
                                          use_separable_filters=use_separable_filters,
                                          rank=rank,
                                          truncation_method=truncation_method,
                                          truncation_value=truncation_value,
                                          preserve_shape=True if truncation_method == com.TruncationMode.FIXED_THRESHOLD
                                          else False)
    # example data matrix
    opts_input = com.OptionsDataMatrix(shape=domain_size,
                                       mode=com.DataMatrixMode.RANDOMIZE_FLOAT_UNIFORM,
                                       rand_range=rand_range,  # integer, excluding the high
                                       const_input=0.)
    # boundary
    opts_boundary = com.OptionsBoundary(enforce=False,
                                        condition=com.BoundaryType.NEUMANN_EDGE,
                                        obj_collide=obj_collide,
                                        post_solve_enforcement=False,
                                        val=0,
                                        dynamic_padding=False,
                                        padding_size=com.get_default_padding(),
                                        left_wall=False,
                                        right_wall=False,
                                        up_wall=False,
                                        down_wall=False,
                                        front_wall=False,  # does not matter in 2D
                                        back_wall=False,  # does not matter in 2D
                                    )

    # packing all..
    opts_general = com.OptionsGeneral(solver=opts_solver, kernel=opts_kernel, reduction=opts_reduction,
                                      boundary=opts_boundary, input=opts_input)

    # ============ Solve ==========
    print(f'kernel type: {kernel_type}')
    print(f'solver type: {solver_type}')
    print(f'zero init: {zero_init}')

    rank_filter_info_list = []
    for i in range(len(trunc_list)):
        print(f'Current Truncation: {trunc_list[i]}')
        opts_general.reduction.truncation_value = trunc_list[i]
        # Compute effective filter sizes for fixed truncation threshold
        rank_and_filter_info = an.compute_adaptive_truncation_factors(opts=opts_general)
        rank_filter_info_list.append(rank_and_filter_info)

    # ============ Plot ==========
    plot_adaptive_truncation_info(trunc_list=trunc_list, rank_filter_info_list=rank_filter_info_list, color_list=None)

    plt.show()


def plot_adaptive_truncation_info(trunc_list, rank_filter_info_list, color_list=None):
    """
    :param List[float] trunc_list:
    :param List[tuple] rank_filter_info_list:
    :param List[str] color_list:
    """

    if color_list is None:
        color_list = ['dodgerblue', 'red', 'orange', 'pink', 'green', 'purple']

    # ============= Max required rank plot ========
    num_itr_rank_trunc = np.arange(0, rank_filter_info_list[0].shape[0], 1)
    num_itr_rank_trunc += 1  # to make it start from 1

    # format (max rank, max effective filter size, actual size)
    fig_rank, ax_rank = plt.subplots()
    for i in range(len(trunc_list)):
        ax_rank.scatter(num_itr_rank_trunc, rank_filter_info_list[i][:, 0], color=color_list[i],
                        label=f'{trunc_list[i]}')
        ax_rank.plot(num_itr_rank_trunc, rank_filter_info_list[i][:, 0], color=color_list[i])
    ax_rank.set_title(f'Max rank needed per filter order (target iteration) for each truncation threshold')
    ax_rank.set_xlabel('Filter Order (Target Iteration)')
    ax_rank.set_ylabel('Max Rank')
    plt.grid()
    plt.legend()

    # ============= Max required HALF filter size plot: hence division by 2 ========
    want_half_size = True # or full size
    include_max_full = True

    # format (max rank, max effective filter size, actual size)
    fig_filter, ax_filter = plt.subplots()
    # ------ include full size bar? -----
    if include_max_full:
        max_filter_size = (rank_filter_info_list[0][:, 2] / (2 if want_half_size else 1)).astype(np.int32)
        ax_filter.bar(num_itr_rank_trunc, max_filter_size,
                      color=color_list[0],
                      label=r'$\delta=$' + '0')  # actual full size (no truncation)

    # ---------------- HACK FORMATTING FOR THE PAPER -------------
    hack = True # hard-coded, only works for a known trunc_list
    # trunc_list= [1e-3, 1e-2, 1e-1] # paper
    labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$']

    for i in range(len(trunc_list)):
        max_filter_size = (rank_filter_info_list[i][:, 1] / (2 if want_half_size else 1)).astype(np.int32)
        ax_filter.bar(num_itr_rank_trunc, max_filter_size,
                      color=color_list[(i + 1) if include_max_full else i],
                      label=r'$\delta=$' + (labels[i] if hack else '{:.0e}'.format(trunc_list[i])))  # effective size

    plt.grid()
    plt.title('Memory Footprint: half filter values stored for each truncation threshold')
    plt.xlabel('Filter Order (Target Iteration)')
    plt.ylabel('Half Filter Size')
    plt.legend()

def demo_residual_comparison_jacobi_poisson_filters_infinite_domain(order, rank=8, truncation_value=0.0,
                                                                    use_full_poisson_kernel=True):
    """Compare Jacobi solution and its residual in the *matrix-form* to that of the Poisson kernel \
    convolution.

    Option to use full Poisson kernel (machine precision error is expected compared to Jacobi), or
    truncated Poisson filters (:code:`use_full_poisson_kernel=False`)

    This is without boundary enforcement. This effectively means an infinite domain setup.
    Infinite domain is achieved by dynamically padding the input data based on the desired
    Jacobi iteration. In other words, we pad the data with sufficiently large blocks so the
    effects of the wall boundaries never reach inside the domain, where the original much
    smaller input data exists.

    In case of using the full kernel, there is no reduction, so we expect an exact match
    between Jacobi *matrix-form* and the Poisson filters solution.

    .. warning::
        The residual comparison is only valid for *inverse* Poisson setup. If you ever decide  \
        to use the *forward* Poisson setup keep in mind that while plotting residuals is not reliable \
        (because of the way we compute residuals in our implementation),\
        the actual solutions to Jacobi and Poisson filters perfectly match, as demonstrated in paper results.

        The implementation of the *matrix form* residual for *forward* Poisson is future work.

    :param int order: filter order (target iteration)
    :param int rank: how many ranks in decomposition (8 or less is usually enough).
        Only used when in reduction mode.
    :param float truncation_value: cut-off threshold to trim the filters, given the truncation
        method in this demo is set to :code:`FIXED_THRESHOLD`. A :code:`0` value means no truncation.
    :param bool use_full_poisson_kernel: if :code:`True` there is no reduction, full kernel convolution.
        If :code:`False`, we are either using the reduced square Poisson kernel, or
        just using the separable filters. Look inside the code for comments on :code:`reduce` and
        :code:`use_separable_filters` variables.
    """
    dim = com.SolverDimension.D2

    # =============== Options ===============

    # solver
    solver_type = com.PoissonSolverType.INVERSE

    # kernel/solver parameters
    kernel_type = com.PoissonKernelType.STANDARD
    # kernel_type = com.PoissonKernelType.UNIFIED
    zero_init = True  # See :func:`OptionsPoissonSolver`

    print(f'kernel type: {kernel_type}')
    print(f'solver type: {solver_type}')
    print(f'zero init: {zero_init}')
    opts_solver = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init)

    # =============== functions Parameters ===============
    # both for diffusion and pressure
    itr = order
    domain_base_size = 15  # always odd size to keep everything symmetric with a central cell available
    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 1.  # diffusivity - only for diffusion
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)
    opts_kernel = com.OptionsKernel(kernel_type=kernel_type, itr=itr, dx=dx, dt=dt, kappa=kappa,
                                    alpha=alpha, beta=beta, clear_memo=True)

    truncation_method = com.TruncationMode.FIXED_THRESHOLD  # truncate everything below the given threshold value

    # 'reduce': if True use reduced kernel, else use the full unreduced kernel
    # 'use_separable_filters': if False use the full 'reduced' kernel, if 'reduce=False' this will be ignored
    if use_full_poisson_kernel:
        reduce = False
        use_separable_filters = False
    else:
        reduce = True
        use_separable_filters = True  # if set False, a 'reduced' kernel will be used, else, filters will be used.

    truncation_info = f', $\delta = ${truncation_value:1.1e}'
    poisson_info = 'Poisson ' + ('Full Kernel' if not reduce else
                                 ('Reduced Kernel' if not use_separable_filters else 'Filters') +
                                 f', Rank={rank}' + truncation_info)

    opts_reduction = com.OptionsReduction(decomp_method=com.DecompMethod.SVD_2D,
                                          reduce=reduce,
                                          use_separable_filters=use_separable_filters,
                                          rank=rank,
                                          truncation_method=truncation_method,
                                          truncation_value=truncation_value,
                                          preserve_shape=True if truncation_method == com.TruncationMode.FIXED_THRESHOLD
                                          else False)
    # example data matrix
    opts_input = com.OptionsDataMatrix(shape=(domain_base_size, domain_base_size),
                                       mode=com.DataMatrixMode.RANDOMIZE_FLOAT_UNIFORM,
                                       rand_range=(-1, 1),  # integer, excluding the high
                                       const_input=0.)
    # boundary
    opts_boundary = com.OptionsBoundary(enforce=False, val=0, padding_size=com.get_default_padding())

    # packing all..
    opts_general = com.OptionsGeneral(solver=opts_solver, kernel=opts_kernel, reduction=opts_reduction,
                                      boundary=opts_boundary, input=opts_input)

    # =============== Solve ===============

    error_abs_jm_vs_pk, error_rel_jm_vs_pk, jacobi_solution_matrix_form, poisson_solution, data, \
        jacobi_residuals, poisson_residual = compare_jacobi_poisson_no_bc_2d(opts=opts_general)

    # =============== Plots ===============

    vis.use_style(style_name="dark_background")
    fig_size = (16, 10)

    specs_info = f'Jacobi Matrix Form vs ' + poisson_info + f' (Itr/Order={itr}, Data={opts_input.shape})'

    opts_plots = com.OptionsPlots(show_values=False, no_ticks=True, aspect_ratio=1., cbar=True,
                                  cbar_orientation="horizontal", cbar_only_min_max=True, cbar_shrink=.48,
                                  line_widths=0.0, cmap="rocket", fmt=".2f", plt_show=False)

    opts_subplots = com.OptionsSubPlots(layout=(2, 2))
    # using divergent colormap for the error plots
    manual_cmaps = ["rocket", "rocket", "rocket", "twilight", "twilight", "twilight"]

    sub_plots = []
    vis.init_subplots(sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Jacobi solution matrix form
    vis.add_subplot(M=jacobi_solution_matrix_form, title="Jacobi Solution Matrix Form", sub_plots=sub_plots,
                    opts_subplots=opts_subplots)

    # Full Poisson kernel solution
    vis.add_subplot(M=poisson_solution, title=poisson_info, sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Rel. Err. Jacobi matrix form vs poisson kernel
    vis.add_subplot(M=error_abs_jm_vs_pk, title='Absolute Error', sub_plots=sub_plots, opts_subplots=opts_subplots)

    # plot all
    axes = vis.plot_matrix_grid(G=sub_plots, title=specs_info, projection='2d', fig_size=fig_size,
                                opts_general=opts_plots, opts_subplots=opts_subplots,
                                cmaps=manual_cmaps,
                                clear_unused=False)

    # make sure the remaining subplots are not cleared: clear_unused=False in plot_matrix_grid (above)
    ax = axes.reshape((-1,))[-1]  # get last subplot
    color_pf = 'greenyellow'
    color_jac = 'cyan'
    ax.plot(jacobi_residuals, color=color_jac)
    ax.plot(itr, poisson_residual, 'o', color='red')
    ax.set_title('Convergence: Iterative Jacobi vs. ' + poisson_info)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_yscale('log')
    # highlight the poisson kernel/filter residual: it is a single scalar matching
    # the max Jacobi iteration
    from_x = 0
    from_y = poisson_residual
    to_x = itr
    to_y = poisson_residual
    # highlighting the Poisson kernel/filter residual
    vis.draw_line(ax=ax, from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y, color=color_pf, linewidth=2.5)
    vis.add_text(ax=ax, coord_x=from_x + 1, coord_y=1.1 * to_y, text=poisson_info, fontsize=12, color=color_pf,
             fontweight='bold', alpha=0.75, horizontal_alignment='left', vertical_alignment='center')
    # highlighting the Jacobi residuals
    vis.add_text(ax=ax, coord_x=from_x + 1, coord_y=np.max(jacobi_residuals), text='Jacobi', fontsize=12,
                 color=color_jac, fontweight='bold', alpha=0.75,
                 horizontal_alignment='left', vertical_alignment='center')

    # This line will act on behalf of all functions with plots.
    vis.plt.show()


def compare_jacobi_poisson_no_bc_2d(opts):
    """Compare Jacobi solution and its residual in the *matrix-form* to that of the Poisson kernel convolution for an \
    infinite domain (See :func:`demo_3methods_comparison_no_bc_2d` for an explanation of the infinite domain).

    The Jacobi convergence curve provides a lower bound on the Poisson filter possible solutions. This means \
    if there is no numerical loss due to filter reductions or truncations, Poisson filters must exactly match \
    the convergence behaviour of Jacobi within machine precision, \
    which will be the best case for numerical quality, and worst case performance-wise.

    Any reduction or truncation results in degradation of the convergence behaviour in exchange for performance speed up.

    .. warning::
        **IMPORTANT!**

        If going with :code:`UNIFIED` kernel instead of :code:`STANDARD` you need the following sign flip in the rhs
        data to make it work. This is the only downside of benefiting from a :code:`UNIFIED` kernel.

        .. code-block:: python

            data_domain *= -1. if opts.kernel.kernel_type == com.PoissonKernelType.UNIFIED else 1.

    :param OptionsGeneral opts:
    :return: Jacobi solution and residuals, Poisson filter solution and residual, errors and generated data
    """
    assert com.is_solver_2d(dim=opts.solver.dim), "This function is supposed to work in 2D"

    # Making an example data matrix as the input. data is padded, but data_domain is smaller original one
    data_padded, data_domain, padding_size = an.prepare_padded_input_matrix(opts=opts)
    opts.boundary.padding_size = padding_size  # updating the options

    # Solve..

    # Full Poisson kernel solve
    poisson_kernel = gen.poisson_kernel_2d(opts=opts)
    poisson_solution, safe_rank = an.solve_poisson_2d(M=data_padded, poisson_kernel=poisson_kernel, opts=opts)

    # Compute residual
    # IMPORTANT:#
    # since the residual function is in the tensor form, it automatically ignores 1 cell on each
    # side because the Laplacian operator has to avoid walls. This will result in a slightly smaller block
    # of the solution to be used in residual computation.
    #
    # Here we are interested in the overall residual behavior and the final residual, so working with a
    # slightly smaller domain is fine. But this might result in a little bump (deterioration) in the beginning
    # of the convergence plot. If you are not happy with that you need to expand the solution and the input rhs
    # to exactly match the desired data domain size to avoid it.

    # Jacobi matrix form
    jacobi_solution_matrix_form, jacobi_residuals = mlib.solve_jacobi_matrix_form_no_boundary_2d(
        M=data_padded,
        opts=opts,
        do_residuals=True,
        is_subdomain_residual=True,
        subdomain_shape=data_domain.shape)

    # IMPORTANT!
    # If going with UNIFIED kernel instead of STANDARD you need the following sign flip in the rhs data
    # to make it work. This is the only downside of benefiting from a UNIFIED kernel.
    data_domain *= -1. if opts.kernel.kernel_type == com.PoissonKernelType.UNIFIED else 1.

    poisson_residual = mlib.compute_residual_poisson_operator(X=poisson_solution,
                                                              B=data_domain,
                                                              solver_dimension=com.SolverDimension.D2)

    # =============== Errors ===============

    # ============ extra padding +1  ==========
    # Note: we need one extra cell trimming from sides because of using pure Poisson filters
    # in comparison. We could have easily added the incomplete Laplacian kernels for the corners
    # and the cells next to the wall, but this will make it inconsistent with the Jacobi function
    # used to generate the kernels.

    #  Removing the effect of BC
    jacobi_solution_matrix_form = mlib.trim(M=jacobi_solution_matrix_form, size=padding_size + 1)
    poisson_solution = mlib.trim(M=poisson_solution, size=1)

    # =============== comparing jacobi matrix form and poisson kernel ============== "
    epsilon = 1e-20  # regularization to avoid division by zero
    err_abs_jm_vs_pk = np.abs(poisson_solution - jacobi_solution_matrix_form).astype(np.double)
    epsilon_m = epsilon * np.ones_like(jacobi_solution_matrix_form).astype(np.double)
    err_rel_jm_vs_pk = 100. * np.abs(err_abs_jm_vs_pk / (jacobi_solution_matrix_form + epsilon_m))  # relative error %

    return err_abs_jm_vs_pk, err_rel_jm_vs_pk, jacobi_solution_matrix_form, poisson_solution, data_padded, \
        jacobi_residuals, poisson_residual


if __name__ == "__main__":
    # UNCOMMENT TO RUN THE DEMOS...

    # 1. Sanity check: compare jacobi solutions; no boundary condition: infinite domain
    # UNCOMMENT------------------------------------------------
    # demo_3methods_comparison_no_bc_2d(order=50, rank=8, use_full_poisson_kernel=True)

    # 2. Sanity check: testing 3 methods with Neumann bc
    # UNCOMMENT------------------------------------------------
    # demo_3methods_comparison_with_wall_neumann_bc_2d(order=20, rank=8)

    # 3. Plot adaptive truncation info
    # UNCOMMENT------------------------------------------------
    # demo_adaptive_truncation_analysis_2d(order=100, rank=8)

    # 4. Compare Jacobi solution and its residual in the *matrix-form* to that of the Poisson kernel \
    #     convolution.
    #  Option to use full kernel (machine precision error is expected compared to Jacobi), or
    #  truncated Poisson filters (use_full_poisson_kernel=False)
    # UNCOMMENT------------------------------------------------
    demo_residual_comparison_jacobi_poisson_filters_infinite_domain(order=60, rank=8, truncation_value=1e-3,
                                                                    use_full_poisson_kernel=True)
