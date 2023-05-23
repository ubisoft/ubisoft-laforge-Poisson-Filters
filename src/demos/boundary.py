""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`
"""
import numpy as np
import src.functions.analytics as an
import src.helper.commons as com
import src.helper.visualizer as vis
import matplotlib.pyplot as plt
import src.functions.generator as gen
import src.functions.mathlib as mlib
import src.functions.decompositions as dec
import src.helper.iohandler as io


def demo_wall_neumann_tiled_mirrored_2d(order=15, rank=8):
    """
    Enforcing Neumann boundary using data reflection technique. Here the input data
    is extended by tiled mirrors on both horizontal and vertical axes. Solving with :math:`n` iteration
    Jacobi for the input data with explicit Neumann boundary enforcement must be equivalent to
    convolving the much larger tiled mirrored data with Poisson filters without an explicit boundary
    condition enforcement. Boundary condition seeks to achieve zero gradient on the edge of the cell

    This only works for Neumann, and is the basis for our *mirror marching* algorithm (see paper).
    This method is similar to what spectral methods use for enforcing Neumann boundary condition.

    **How it works**: based on the target iteration we compute how much padding the input data needs
    so that shrinkage of the extended data due to convolution still gives the same matrix size as
    performing Jacobi after :math:`n` iteration.

    :param int order: filter order (target iteration)
    :param int rank: how many ranks in decomposition (8 or less is usually enough)
    """
    dim = com.SolverDimension.D2

    # =============== Options ===============

    solver_type = com.PoissonSolverType.INVERSE
    kernel_type = com.PoissonKernelType.STANDARD
    zero_init = True

    opts_solver = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init)

    print(f'solver type: {solver_type}')
    print(f'kernel type: {kernel_type}')
    print(f'zero_init: {zero_init}')

    # =============== functions Parameters ===============
    # both for diffusion and pressure
    itr = order
    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 100.  # diffusivity - only for diffusion
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)
    opts_kernel = com.OptionsKernel(kernel_type=kernel_type, itr=itr, dx=dx, dt=dt, kappa=kappa,
                                    alpha=alpha, beta=beta, clear_memo=True)
    # reduction
    truncation_method = com.TruncationMode.PERCENTAGE
    truncation_value = 0.
    opts_reduction = com.OptionsReduction(decomp_method=com.DecompMethod.SVD_2D,
                                          reduce=True,
                                          use_separable_filters=True,
                                          rank=rank,
                                          truncation_method=truncation_method,
                                          truncation_value=truncation_value,
                                          preserve_shape=True if truncation_method == com.TruncationMode.FIXED_THRESHOLD
                                          else False)
    # example data matrix
    opts_input = com.OptionsDataMatrix(shape=(9, 9),
                                       mode=com.DataMatrixMode.RANDOMIZE_INT,
                                       special_pattern=com.DataMatrixPattern.CENTRAL_SQUARE,
                                       rand_range=(1, 5),  # integer, excluding the high
                                       const_input=2)
    # boundary
    opts_boundary = com.OptionsBoundary(enforce=True,
                                        condition=com.BoundaryType.NEUMANN_EDGE,
                                        val=0,
                                        padding_size=com.get_default_padding(),
                                        left_wall=True,
                                        right_wall=True,
                                        up_wall=True,
                                        down_wall=True)

    # packing all..
    opts_general = com.OptionsGeneral(solver=opts_solver, kernel=opts_kernel, reduction=opts_reduction,
                                      boundary=opts_boundary, input=opts_input)

    # =============== Solve ===============

    jacobi_solution, poisson_solution, data_raw, data_mirrored, safe_rank, err_abs, err_rel =\
        an.compare_jacobi_poisson_neumann_edge_mirrored_correction_2d(opts=opts_general)

    # poisson gradient
    poisson_grad = mlib.compute_gradient_2d(M=poisson_solution, grad_scale=1., half_dx=0.5)
    poisson_grad_x = poisson_grad[:, :, 0]
    poisson_grad_y = poisson_grad[:, :, 1]

    # =============== Plots ===============

    # compute and show gradients for the pressure test
    _specs_info = f'{solver_type.name} Poisson Solve, Order={itr}, Rank={safe_rank},' \
        f' {opts_reduction.truncation_method}, TruncationValue={opts_reduction.truncation_value}, Data={opts_input.shape}'

    opts_plots = com.OptionsPlots(show_values=False, no_ticks=True, aspect_ratio=1., cbar=True,
                                  cbar_orientation="horizontal",
                                  line_widths=0.0, fmt=".0f", plt_show=False,
                                  cmap="viridis",
                                  # interpolate=True, # if interpolated there wont be a cbar
                                  interpolate=False, # able to add cbar
                                  alpha=.9)

    opts_subplots = com.OptionsSubPlots(layout=(2, 3))

    # Solutions
    sub_plots = []
    vis.init_subplots(sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Jacobi solution
    vis.add_subplot(M=jacobi_solution, title="Jacobi Solution (Explicit Boundary Enforcement)" + f' iteration={itr}',
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # Poisson kernel solution tiled mirrored
    vis.add_subplot(M=poisson_solution, title="Poisson Convolution (Tiled Mirrored)" + f' Order={itr}',
                    sub_plots=sub_plots, opts_subplots=opts_subplots)

    # relative error
    vis.add_subplot(M=err_rel, title="Relative Error %", sub_plots=sub_plots, opts_subplots=opts_subplots)

    # data
    vis.add_subplot(M=data_raw, title="Raw Data", sub_plots=sub_plots, opts_subplots=opts_subplots)

    # data mirrored
    vis.add_subplot(M=data_mirrored, title="Tiled Mirrored Data", sub_plots=sub_plots, opts_subplots=opts_subplots)

    # gradient x
    vis.add_subplot(M=poisson_grad_x, title="Gradient X", sub_plots=sub_plots, opts_subplots=opts_subplots)

    # plot all
    vis.plot_vector_field_2d(Mx=poisson_grad_x, My=poisson_grad_y, title='Solution Gradient', interp=True, cmap=None)
    vis.plot_matrix_grid(G=sub_plots, title=_specs_info, projection='2d',
                         opts_general=opts_plots, opts_subplots=opts_subplots)

    # This line will act on behalf of all functions with plots.
    plt.show()


def demo_multi_modal_solve_neumann_complex_object_2d(order=60, rank=4, verbose=True):
    """Plot the break down of the Poisson filters solve visualizing individual rank contributions and their cumulative sum.
    With wall and complex object Neumann boundary treatment using *mirror marching* algorithm.

    :param int order: filter order (target iteration)
    :param int rank: how many ranks in decomposition (8 or less is usually enough).
    :param bool verbose: show logs
    """
    # Dimension #
    dim = com.SolverDimension.D2

    # Solver / Kernel #
    solver_type = com.PoissonSolverType.INVERSE
    kernel_type = com.PoissonKernelType.STANDARD
    zero_init = True  # See :func:`OptionsPoissonSolver`

    # Parameters
    itr = order
    # Getting a list of ordered ranks.
    # if this line does not work simply make a list of ranks, e.g. [1, 2, 3, 4] for rank=4
    individual_rank_list = [(lambda m:m + 1)(m) for m in range(rank)]

    truncation_value = 0.
    truncation_method = com.TruncationMode.PERCENTAGE
    domain_size = (81, 81)
    rand_seed = 50
    obj_collide = True  # if False only walls will be taken into account
    collision_mask_fname = 'boundaries_circle.png'

    # Plots ...
    # SSIM or normalized abs error?
    show_ssim_diff_instead_of_abs_err = False
    clean_obj_mask_solution_values = False  # for the sake of correct representative color bar, manually
    # set the cells values inside the mask to be the average of the domain value, instead of zero

    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 1.  # only for diffusion
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)  # alpha = dx^2
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)  # beta = 4

    opts_general = com.generic_options(dim=dim, solver_type=solver_type, zero_init=zero_init, kernel_type=kernel_type,
                                       itr=itr, rank=rank, domain_size=domain_size, rand_seed=rand_seed,
                                       obj_collide=obj_collide, dx=dx, dt=dt, kappa=kappa, alpha=alpha, beta=beta,
                                       truncation_method=truncation_method, truncation_value=truncation_value,
                                       force_symmetric_unit_range=True)

    # =============== Solve ===============
    jacobi_solution, poisson_solution_all_ranks, poisson_solution_individual_ranks, data_domain, data_wall_padding, \
        collision_mask, contour_mask, safe_rank, \
        err_abs_all_ranks, err_rel_all_ranks, err_abs_individual_ranks, err_rel_individual_ranks, \
        ssim_image_all_ranks, diff_image_all_ranks, ssim_mean_all_ranks, \
        ssim_image_individual_ranks, diff_image_individual_ranks, \
        ssim_grad_individual_ranks, ssim_mean_individual_ranks = \
        compare_jacobi_poisson_filters_neumann_individual_ranks(opts=opts_general,
                                                                collision_mask_fname=collision_mask_fname,
                                                                individual_ranks=individual_rank_list,
                                                                verbose=verbose)

    err_matrix_individual_ranks = []
    if show_ssim_diff_instead_of_abs_err:
        err_matrix_all_ranks = ssim_image_all_ranks

        for ee in range(len(ssim_image_individual_ranks)):
            err_matrix_individual_ranks.append(ssim_image_individual_ranks[ee])
    else:
        err_matrix_all_ranks = err_abs_all_ranks
        # normalizing the absolute error
        err_matrix_all_ranks /= (np.max(jacobi_solution) - np.min(jacobi_solution))
        # err_matrix_all_ranks = mlib.normalize_range(np.abs(err_matrix_all_ranks ))

        for ee in range(len(err_abs_individual_ranks)):
            err_matrix_this_rank = err_abs_individual_ranks[ee]
            # normalizing the absolute error
            err_matrix_this_rank /= (
                    np.max(poisson_solution_individual_ranks[ee]) - np.min(poisson_solution_individual_ranks[ee]))
            err_matrix_individual_ranks.append(err_matrix_this_rank)

    # =============== Plots ===============
    print("Plotting...")
    # fig_size = None
    fig_size = (18, 12)
    vis.sns.set(font_scale=1.)

    truncation_info = f'Truncation $\delta = ${truncation_value:1.1e}'
    poisson_info = 'Poisson Filters' + f', Itr/Order={itr}, Rank={opts_general.reduction.rank}, ' + truncation_info
    specs_info = poisson_info + f', Domain Size={opts_general.input.shape}'
    title = specs_info + '\nTOP: Modal Solutions - Contributions of each rank to the solution. \nBOTTOM: Absolute ' \
                         'normalized error of each component when compared to ground truth Jacobi.'
    title += '\n' + f'Solution Range=({np.min(jacobi_solution):1.1e},{np.max(jacobi_solution):1.1e})'
    title += '\n' + 'Color bars indicate min/max, and 95th, 99th percentiles.'

    # General options for plots
    opts_plots = com.OptionsPlots(show_values=False,
                                  no_ticks=True,
                                  aspect_ratio=1.,
                                  cbar=True,
                                  cbar_orientation="horizontal",
                                  cbar_shrink=.9,
                                  cbar_scientific=False,
                                  cbar_only_min_max=True,
                                  cbar_add_percentile=True,
                                  line_widths=0.0,  # 0.01,
                                  fmt=".0f",
                                  plt_show=False,
                                  cmap="rocket",  # cmap="viridis",
                                  # cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True),
                                  # interpolate=True, # if interpolated there won't be a cbar
                                  interpolate=False,  # no cbar with interpolate
                                  alpha=.9)

    # Sub plot options
    opts_subplots = com.OptionsSubPlots(layout=(2, 1 + len(individual_rank_list)))
    sub_plots = []
    vis.init_subplots(sub_plots=sub_plots, opts_subplots=opts_subplots)

    interior_face_color = (65. / 255., 21. / 255., 92. / 255.)  # dark indigo
    contour_face_color = 'cornflowerblue'
    edge_color = 'azure'
    interior_line_width = 0
    contour_line_width = 1

    # finding the bad cells, which are the interior of the solid (excluding the contour)
    interior_mask = an.subtract_contour_from_solid(solid=collision_mask, contour=contour_mask)

    # ============  Poisson solutions subplots ============
    # Poisson solution all ranks
    if clean_obj_mask_solution_values:
        poisson_solution_all_ranks = vis.correct_color_bar_2d(M=poisson_solution_all_ranks, interior_mask=interior_mask)
    vis.add_subplot(M=poisson_solution_all_ranks, title='Sum of ' + str(rank) + ' Ranks',
                    sub_plots=sub_plots, opts_subplots=opts_subplots, cmap="viridis")

    # Poisson solution individual ranks
    for rr in range(len(poisson_solution_individual_ranks)):
        poisson_solution_this_rank = poisson_solution_individual_ranks[rr]
        if clean_obj_mask_solution_values:
            poisson_solution_this_rank = vis.correct_color_bar_2d(M=poisson_solution_this_rank, interior_mask=interior_mask)
        vis.add_subplot(M=poisson_solution_this_rank, title='Contribution of Component ' + str(individual_rank_list[rr]),
                        sub_plots=sub_plots, opts_subplots=opts_subplots, cmap="viridis")

    # ============ Error subplots ============
    cbar_percentile_list = [95, 99]
    # Error all ranks
    if clean_obj_mask_solution_values:
        err_matrix_all_ranks = vis.correct_color_bar_2d(M=err_matrix_all_ranks, interior_mask=interior_mask)

    vis.add_subplot(M=err_matrix_all_ranks, title='Sum of ' + str(rank) + ' Ranks Abs. Err.', sub_plots=sub_plots,
                    opts_subplots=opts_subplots, cbar_percentile_list=cbar_percentile_list, cmap="rocket")

    # Error matrix individual ranks
    for rr in range(len(poisson_solution_individual_ranks)):
        err_matrix_this_rank = err_matrix_individual_ranks[rr]
        if clean_obj_mask_solution_values:
            err_matrix_this_rank = vis.correct_color_bar_2d(M=err_matrix_this_rank, interior_mask=interior_mask)

        vis.add_subplot(M=err_matrix_this_rank,
                        title='Abs. Err. of Rank ' + str(individual_rank_list[rr]) + ' Contribution',
                        sub_plots=sub_plots, opts_subplots=opts_subplots,
                        cbar_percentile_list=cbar_percentile_list, cmap="rocket")

    # plot all
    additional_info = ' - NOTE: object mask values are manipulated for the sake of' \
                      ' a correct color bar instead of zero, look at clean_obj_mask_solution_values' if \
        clean_obj_mask_solution_values and opts_plots.show_values else ''

    additional_info += ' for a correct color bar enable clean_obj_mask_solution_values' \
        if not clean_obj_mask_solution_values else ''

    axes = vis.plot_matrix_grid(G=sub_plots, title=title,
                                projection='2d', opts_general=opts_plots, opts_subplots=opts_subplots,
                                fig_size=fig_size, cmaps=opts_subplots.cmaps)

    # add collision mask to the plots
    ax_index = 0  # just put the following functions in the same order as the subplots

    # ============ Poisson solution + collision mask ============
    vis.add_object_mask(interior_mask=interior_mask, contour_mask=contour_mask, opts_plots=opts_plots,
                        show_interior_mask=True, fill_interior_mask=True,
                        show_contour_mask=True, fill_contour_mask=False, ax=axes[ax_index],
                        interior_facecolor=interior_face_color, contour_facecolor=contour_face_color,
                        edge_color=edge_color,
                        interior_linewidth=interior_line_width, contour_linewidth=contour_line_width)
    ax_index += 1

    for rr in range(len(poisson_solution_individual_ranks)):
        # Poisson solution + collision mask
        vis.add_object_mask(interior_mask=interior_mask, contour_mask=contour_mask, opts_plots=opts_plots,
                            show_interior_mask=True, fill_interior_mask=True,
                            show_contour_mask=True, fill_contour_mask=False, ax=axes[ax_index],
                            interior_facecolor=interior_face_color, contour_facecolor=contour_face_color,
                            edge_color=edge_color,
                            interior_linewidth=interior_line_width, contour_linewidth=contour_line_width)
        ax_index += 1

    # ============ Error + collision mask ============
    vis.add_object_mask(interior_mask=interior_mask, contour_mask=contour_mask, opts_plots=opts_plots,
                        show_interior_mask=True, fill_interior_mask=False,
                        show_contour_mask=True, fill_contour_mask=False, ax=axes[ax_index],
                        interior_facecolor='black', contour_facecolor=contour_face_color,
                        edge_color=edge_color,
                        interior_linewidth=interior_line_width, contour_linewidth=contour_line_width)
    ax_index += 1

    for rr in range(len(poisson_solution_individual_ranks)):
        # Error + collision mask
        vis.add_object_mask(interior_mask=interior_mask, contour_mask=contour_mask, opts_plots=opts_plots,
                            show_interior_mask=True, fill_interior_mask=False,
                            show_contour_mask=True, fill_contour_mask=False, ax=axes[ax_index],
                            interior_facecolor='black', contour_facecolor=contour_face_color,
                            edge_color=edge_color,
                            interior_linewidth=interior_line_width, contour_linewidth=contour_line_width)
        ax_index += 1

    # plt.subplots_adjust(top=1, bottom=0.03, right=0.97, left=0.03)
    vis.plt.show()


def compare_jacobi_poisson_filters_neumann_individual_ranks(opts, collision_mask_fname, individual_ranks,
                                                            verbose=False):
    """Break down of the Poisson filters solve with individual rank contributions and their cumulative sum.
    With wall and complex object Neumann boundary treatment using *mirror marching* algorithm.

    :param OptionsGeneral opts:
    :param str collision_mask_fname: collision mask file name. See :func:`functions.analytics.generate_collision_mask`
    :param List[int] individual_ranks: list of ranks to be included. Ranks do not have to be consecutive.
    :param bool verbose: show logs
    :return: Jacobi solution, Poisso filters solution and its individual rank contributions, absolute and relative
        errors, SSIM components
    """
    # data generation
    data_padded, data_domain, padding_size = an.prepare_padded_input_matrix(opts=opts)

    # =============== Solve using Jacobi ===============

    # jacobi data: single padding Jacobi on raw data to add walls
    data_wall_padding = mlib.expand_with_padding_2d(M=data_domain,
                                                    pad_size=1,
                                                    pad_value=opts.boundary.val,
                                                    opts_boundary_detailed_wall=opts.boundary)

    # Prepare boundary collision mask
    collision_mask, contour_mask = an.generate_collision_mask(M=data_wall_padding,
                                                              path=an.image_folder_path + collision_mask_fname,
                                                              load_shape_boundary=opts.boundary.obj_collide)

    jacobi_solution = mlib.solve_jacobi_single_padding_obj_collision_2d(M=data_wall_padding,
                                                                        opts=opts,
                                                                        collision_mask=collision_mask)

    # =============== Solve using PF ===============
    # no boundary solution for the forward equation yet
    if opts.solver.solver_type == com.PoissonSolverType.FORWARD:  # use good old dynamic padding
        assert False, "Not working on Forward yet...."

    # Get filters
    v_hor, v_ver, safe_rank = dec.poisson_filters_2d(opts=opts)

    if verbose:
        io.print_filters(v_hor, same_line=True, additional_info=True)

    # Test using separable filters

    # solving for cumulative full rank (individual_rank=None)
    poisson_solution_all_ranks = mlib.solve_poisson_separable_filters_obj_aware_2d(M=data_wall_padding,
                                                                                   filter_hor=v_hor, filter_ver=v_ver,
                                                                                   safe_rank=safe_rank,
                                                                                   collision_mask=collision_mask,
                                                                                   opts=opts,
                                                                                   individual_rank=None)
    small_value_threshold = 1e-14

    # =============== Compute errors/SSIM ===============
    # zero out small values to avoid unnecessary dissimilarities
    jacobi_solution = mlib.zero_out_small_values(M=jacobi_solution, threshold=small_value_threshold)
    poisson_solution_all_ranks = mlib.zero_out_small_values(M=poisson_solution_all_ranks, threshold=small_value_threshold)

    # abs and rel errors
    err_abs_all_ranks, err_rel_all_ranks = mlib.compute_abs_rel_error(M1=jacobi_solution, M2=poisson_solution_all_ranks)
    # ssim
    ssim_image_all_ranks, diff_image_all_ranks, ssim_grad, ssim_mean_all_ranks = mlib.compute_ssim(
        test_image=poisson_solution_all_ranks, ref_image=jacobi_solution)

    # =============== solving for individual ranks ===============
    poisson_solution_individual_ranks = []
    err_abs_individual_ranks = []
    err_rel_individual_ranks = []
    ssim_image_individual_ranks = []
    diff_image_individual_ranks = []
    ssim_grad_individual_ranks = []
    ssim_mean_individual_ranks = []
    for rr in range(len(individual_ranks)):
        individual_rank_solution = mlib.solve_poisson_separable_filters_obj_aware_2d(
            M=data_wall_padding,
            filter_hor=v_hor, filter_ver=v_ver,
            safe_rank=safe_rank,
            collision_mask=collision_mask,
            opts=opts,
            individual_rank=individual_ranks[rr])

        # zero out small values to avoid unnecessary dissimilarities
        # individual_rank_solution = zero_out_small_values(M=individual_rank_solution, threshold=small_value_threshold)
        # abs and rel errors
        err_abs_this_rank, err_rel_this_rank = mlib.compute_abs_rel_error(M1=jacobi_solution, M2=individual_rank_solution)
        # ssim
        ssim_image_this_rank, diff_image_this_rank, ssim_grad_this_rank, ssim_mean_this_rank = \
            mlib.compute_ssim(test_image=poisson_solution_all_ranks, ref_image=jacobi_solution)

        poisson_solution_individual_ranks.append(individual_rank_solution)
        err_abs_individual_ranks.append(err_abs_this_rank)
        err_rel_individual_ranks.append(err_rel_this_rank)
        ssim_image_individual_ranks.append(ssim_image_this_rank)
        diff_image_individual_ranks.append(diff_image_this_rank)
        ssim_grad_individual_ranks.append(ssim_grad_this_rank)
        ssim_mean_individual_ranks.append(ssim_mean_this_rank)

    return jacobi_solution, poisson_solution_all_ranks, poisson_solution_individual_ranks, data_domain, \
        data_wall_padding, collision_mask, contour_mask, safe_rank, \
        err_abs_all_ranks, err_rel_all_ranks, err_abs_individual_ranks, err_rel_individual_ranks, \
        ssim_image_all_ranks, diff_image_all_ranks, ssim_mean_all_ranks, \
        ssim_image_individual_ranks, diff_image_individual_ranks, ssim_grad_individual_ranks, ssim_mean_individual_ranks


def compare_jacobi_poisson_filters_neumann_simple(opts, collision_mask_fname, verbose=False):
    """Run a specific experiment to compare ground truth Jacobi solution and Poisson filters solution with Neumann \
    boundary treatment.

    :param OptionsGeneral opts:
    :param str collision_mask_fname: collision mask file name. See :func:`functions.analytics.generate_collision_mask`
    :param bool verbose: show logs
    :return: ground truth Jacobi solution, Poisson filters solution, errors, generated domain data, collision and \
        contour masks, SSIM components, safe rank
    """
    # data generation
    data_padded, data_domain, padding_size = an.prepare_padded_input_matrix(opts=opts)

    # =============== Solve using Jacobi ===============

    # jacobi data: single padding Jacobi on raw data to add walls
    data_wall_padding = mlib.expand_with_padding_2d(M=data_domain,
                                                    pad_size=1,
                                                    pad_value=opts.boundary.val,
                                                    opts_boundary_detailed_wall=opts.boundary)

    # Prepare boundary collision mask
    collision_mask, contour_mask = an.generate_collision_mask(M=data_wall_padding,
                                                              path=an.image_folder_path + collision_mask_fname,
                                                              load_shape_boundary=opts.boundary.obj_collide)

    jacobi_solution = mlib.solve_jacobi_single_padding_obj_collision_2d(M=data_wall_padding,
                                                                   opts=opts,
                                                                   collision_mask=collision_mask)

    # =============== Solve using PF ===============
    # no boundary solution for the forward equation yet
    if opts.solver.solver_type == com.PoissonSolverType.FORWARD:  # use good old dynamic padding
        assert False, "Not working on Forward yet...."

    # Get filters
    v_hor, v_ver, safe_rank = dec.poisson_filters_2d(opts=opts)

    if verbose:
        io.print_filters(v_hor, same_line=True, additional_info=True)

    # Test using separable filters

    # solving for cumulative full rank (individual_rank=None)
    poisson_solution = mlib.solve_poisson_separable_filters_obj_aware_2d(M=data_wall_padding,
                                                                         filter_hor=v_hor, filter_ver=v_ver,
                                                                         safe_rank=safe_rank,
                                                                         collision_mask=collision_mask,
                                                                         opts=opts,
                                                                         individual_rank=None)
    small_value_threshold = 1e-14

    # =============== Compute errors/SSIM ===============
    # zero out small values to avoid unnecessary dissimilarities
    jacobi_solution = mlib.zero_out_small_values(M=jacobi_solution, threshold=small_value_threshold)
    poisson_solution = mlib.zero_out_small_values(M=poisson_solution, threshold=small_value_threshold)

    # abs and rel errors
    err_abs, err_rel = mlib.compute_abs_rel_error(M1=jacobi_solution, M2=poisson_solution)
    # ssim
    ssim_image, diff_image, ssim_grad, ssim_mean = mlib.compute_ssim(
        test_image=poisson_solution, ref_image=jacobi_solution)

    return jacobi_solution, poisson_solution, data_domain, data_wall_padding, collision_mask, contour_mask, safe_rank, \
        err_abs, err_rel, ssim_image, diff_image, ssim_mean


def run_targeted_experiment(order, rank, truncation_method, truncation_value, opts_general, collision_mask_fname,
                            verbose, show_ssim_diff_instead_of_abs_err):
    """Run a specific experiment to compare ground truth Jacobi solution and Poisson filters solution with Neumann \
    boundary treatment.

    :param int order: filter order (target iteration)
    :param int rank: how many ranks in decomposition (8 or less is usually enough).
    :param TruncationMode truncation_method: See :func:`TruncationMode`
    :param float truncation_value: :func:`TruncationMode`
    :param OptionsGeneral opts_general:
    :param str collision_mask_fname: collision mask file name. See :func:`functions.analytics.generate_collision_mask`
    :param bool verbose: show logs
    :param bool show_ssim_diff_instead_of_abs_err: use SSIM instead of absolute error for the difference
    :return: ground truth Jacobi, error of Poisson filters solution, collision mask, contour mask
    """
    opts_general.kernel.itr = order
    opts_general.reduction.rank = rank
    opts_general.reduction.truncation_method = truncation_method
    opts_general.reduction.truncation_value = truncation_value

    jacobi_solution, poisson_solution, data_domain, data_wall_padding, collision_mask, contour_mask, safe_rank, \
        err_abs, err_rel, ssim_image, diff_image, ssim_mean = \
        compare_jacobi_poisson_filters_neumann_simple(opts=opts_general, collision_mask_fname=collision_mask_fname,
                                                      verbose=verbose)

    if show_ssim_diff_instead_of_abs_err:
        err_matrix = ssim_image

    else:
        err_matrix = err_abs
        # normalizing the absolute error
        err_matrix /= (np.max(jacobi_solution) - np.min(jacobi_solution))

    return err_matrix, collision_mask, contour_mask, jacobi_solution


def demo_versatile_error_scenarios_neumann_2d(rank=4, verbose=True):
    """Experimenting with various scenarios with different truncation values and Poisson filters order \
    (target Jacobi iteration).

    .. warning::
        Solving with current settings might take a while, be patient!

    :param int rank: how many ranks in decomposition (8 or less is usually enough).
    :param bool verbose: show logs
    """
    # Dimension #
    dim = com.SolverDimension.D2

    # Solver / Kernel #
    solver_type = com.PoissonSolverType.INVERSE
    kernel_type = com.PoissonKernelType.STANDARD
    zero_init = True  # See :func:`OptionsPoissonSolver`

    # Parameters #
    base_itr = 100
    truncation_method = com.TruncationMode.PERCENTAGE
    truncation_value = 0.
    domain_size = (81, 81)
    rand_seed = 50  # for resolution 81^2 and itr=30: 55 gives the best result, 50 gives the best structural error
    obj_collide = True  # if False only walls will be taken into account
    collision_mask_fname = 'boundaries_circle.png'

    # Plots ...
    # SSIM or normalized abs error?
    show_ssim_diff_instead_of_abs_err = False
    clean_obj_mask_solution_values = False  # for the sake of correct representative color bar, manually
    # set the cells values inside the mask to be the average of the domain value, instead of zero

    dx = 1.0
    dt = 1.0  # only for diffusion
    kappa = 1.  # only for diffusion
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)  # alpha = dx^2
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)  # beta = 4

    opts_general = com.generic_options(dim=dim, solver_type=solver_type, zero_init=zero_init, kernel_type=kernel_type,
                                       itr=base_itr, rank=rank, domain_size=domain_size, rand_seed=rand_seed,
                                       obj_collide=obj_collide, dx=dx, dt=dt, kappa=kappa, alpha=alpha, beta=beta,
                                       truncation_method=truncation_method, truncation_value=truncation_value,
                                       force_symmetric_unit_range=True)

    # =============== Solve: Each solution is shown in one subplot===============

    def get_specs_str(mu, trunc_val):
        specs = ''
        specs += r'$\delta$=' + io.format_scientific(x=trunc_val)
        specs += r', $\mu$=' + io.format_scientific(mu, dynamic_format=False)
        return specs

    subplot_matrix_list = []  # collects solutions to be displayed in each subplot
    subplot_titles = []
    # paramteres in the experiments will overwrite the options before solving for anything

    # =========== 1. no truncation ===========
    truncation_method = com.TruncationMode.PERCENTAGE
    truncation_value = 0.

    print("Solving experiment 1: no truncation...")
    err_matrix, collision_mask, contour_mask, jacobi_solution = run_targeted_experiment(
        order=base_itr, rank=rank,
        truncation_method=truncation_method,
        truncation_value=truncation_value,
        opts_general=opts_general,
        collision_mask_fname=collision_mask_fname,
        verbose=verbose,
        show_ssim_diff_instead_of_abs_err=show_ssim_diff_instead_of_abs_err)

    subplot_matrix_list.append(err_matrix)
    # setting the title
    mean = np.mean(err_matrix)
    experiment_title = 'No Truncation'
    experiment_title += '\n' + f'Solution Range=({np.min(jacobi_solution):1.1e},{np.max(jacobi_solution):1.1e})'
    experiment_title += '\n' + 'Filter Order ' + str(base_itr)
    experiment_title += '\n' + get_specs_str(mu=mean, trunc_val=truncation_value)
    subplot_titles.append(experiment_title)

    # =========== 2. Least aggressive adaptive truncation ===========
    truncation_method = com.TruncationMode.FIXED_THRESHOLD
    truncation_value = 1e-2

    print("Solving experiment 2: Least aggressive adaptive truncation...")
    err_matrix, collision_mask, contour_mask, jacobi_solution = run_targeted_experiment(
        order=base_itr, rank=rank,
        truncation_method=truncation_method,
        truncation_value=truncation_value,
        opts_general=opts_general,
        collision_mask_fname=collision_mask_fname,
        verbose=verbose,
        show_ssim_diff_instead_of_abs_err=show_ssim_diff_instead_of_abs_err)

    subplot_matrix_list.append(err_matrix)

    # setting the title
    mean = np.mean(err_matrix)
    experiment_title = 'Least Aggressive Truncation'
    experiment_title += '\n' + 'Filter Order ' + str(base_itr)
    experiment_title += '\n' + get_specs_str(mu=mean, trunc_val=truncation_value)
    experiment_title += '\n' + f'Solution Range=({np.min(jacobi_solution):1.1e},{np.max(jacobi_solution):1.1e})'
    subplot_titles.append(experiment_title)

    # =========== 3. low adaptive truncation ===========
    truncation_method = com.TruncationMode.FIXED_THRESHOLD
    truncation_value = 2.5e-2

    print("Solving experiment 3: low adaptive truncation...")
    err_matrix, collision_mask, contour_mask, jacobi_solution = run_targeted_experiment(
        order=base_itr, rank=rank,
        truncation_method=truncation_method,
        truncation_value=truncation_value,
        opts_general=opts_general,
        collision_mask_fname=collision_mask_fname,
        verbose=verbose,
        show_ssim_diff_instead_of_abs_err=show_ssim_diff_instead_of_abs_err)

    subplot_matrix_list.append(err_matrix)
    # setting the title
    mean = np.mean(err_matrix)
    experiment_title = 'Low Truncation'
    experiment_title += '\n' + 'Filter Order ' + str(base_itr)
    experiment_title += '\n' + get_specs_str(mu=mean, trunc_val=truncation_value)
    experiment_title += '\n' + f'Solution Range=({np.min(jacobi_solution):1.1e},{np.max(jacobi_solution):1.1e})'
    subplot_titles.append(experiment_title)

    # =========== 4. medium truncation ===========
    truncation_method = com.TruncationMode.FIXED_THRESHOLD
    truncation_value = 5e-2

    print("Solving experiment 4: medium truncation...")
    err_matrix, collision_mask, contour_mask, jacobi_solution = run_targeted_experiment(
        order=base_itr, rank=rank,
        truncation_method=truncation_method,
        truncation_value=truncation_value,
        opts_general=opts_general,
        collision_mask_fname=collision_mask_fname,
        verbose=verbose,
        show_ssim_diff_instead_of_abs_err=show_ssim_diff_instead_of_abs_err)

    subplot_matrix_list.append(err_matrix)
    # setting the title
    mean = np.mean(err_matrix)
    experiment_title = 'Medium Truncation'
    experiment_title += '\n' + 'Filter Order ' + str(base_itr)
    experiment_title += '\n' + get_specs_str(mu=mean, trunc_val=truncation_value)
    experiment_title += '\n' + f'Solution Range=({np.min(jacobi_solution):1.1e},{np.max(jacobi_solution):1.1e})'
    subplot_titles.append(experiment_title)

    # # =========== 5. high iteration ===========
    base_itr = 200
    truncation_method = com.TruncationMode.FIXED_THRESHOLD
    truncation_value = 1e-3

    print("Solving experiment 5: high iteration...")
    err_matrix, collision_mask, contour_mask, jacobi_solution = run_targeted_experiment(
        order=base_itr, rank=rank,
        truncation_method=truncation_method,
        truncation_value=truncation_value,
        opts_general=opts_general,
        collision_mask_fname=collision_mask_fname,
        verbose=verbose,
        show_ssim_diff_instead_of_abs_err=show_ssim_diff_instead_of_abs_err)

    subplot_matrix_list.append(err_matrix)
    # setting the title
    mean = np.mean(err_matrix)
    experiment_title = 'Higher Target Iteration' + '\n' + 'Very Low Truncation'
    experiment_title += '\n' + 'Filter Order ' + str(base_itr)
    experiment_title += '\n' + get_specs_str(mu=mean, trunc_val=truncation_value)
    experiment_title += '\n' + f'Solution Range=({np.min(jacobi_solution):1.1e},{np.max(jacobi_solution):1.1e})'
    subplot_titles.append(experiment_title)

    # =============== Plots ===============
    print("Plotting...")
    # fig_size = None
    fig_size = (16, 7)
    title = 'Experimenting with various adaptive truncation thresholds and Poisson filters order (target iterations).'
    title += '\n' + 'Color bars indicate min/max, and 95th, 99th percentiles.'
    title += '\n' + f'input data range={opts_general.input.rand_range}'
    title += '\n' + r'$\delta$: adaptive truncation threshold'
    title += '\n' + r'$\mu$: mean absolute error (normalized)'
    # adjust subplot titles font size
    vis.sns.set(font_scale=1.)

    # plot options
    opts_plots = com.OptionsPlots(show_values=False, no_ticks=True, aspect_ratio=1., cbar=True,
                                  cbar_orientation="horizontal", cbar_shrink=.9, cbar_scientific=True,
                                  cbar_only_min_max=True, cbar_add_percentile=True, line_widths=0.0,  # 0.01, fmt=".0f",
                                  plt_show=False, cmap="rocket",  # cmap="viridis",
                                  # cmap = vis.sns.diverging_palette(0, 230, 90, 60, as_cmap=True),
                                  # interpolate=True, # if interpolated there won't be a cbar
                                  interpolate=False,  # no cbar with interpolate
                                  alpha=.9)

    # Sub plots
    opts_subplots = com.OptionsSubPlots(layout=(1, len(subplot_matrix_list)))
    sub_plots = []
    vis.init_subplots(sub_plots=sub_plots, opts_subplots=opts_subplots)

    for ee in range(len(subplot_matrix_list)):
        subplot_matrix = subplot_matrix_list[ee]
        add_subplot_matrix(subplot_matrix=subplot_matrix, opts_subplots=opts_subplots, sub_plots=sub_plots,
                           collision_mask=collision_mask, contour_mask=contour_mask,
                           clean_obj_mask_solution_values=clean_obj_mask_solution_values,
                           title=subplot_titles[ee])

    # flush out all subplots
    axes = vis.plot_matrix_grid(G=sub_plots, title=title,
                            projection='2d', opts_general=opts_plots, opts_subplots=opts_subplots, fig_size=fig_size,
                            cmaps=opts_subplots.cmaps)

    # finding the bad cells, which are the interior of the solid (excluding the contour)
    interior_mask = an.subtract_contour_from_solid(solid=collision_mask, contour=contour_mask)

    vis.add_collision_masks(interior_mask=interior_mask, contour_mask=contour_mask,
                            opts_plots=opts_plots, opts_subplots=opts_subplots, axes=axes)

    vis.plt.subplots_adjust(top=1, bottom=0.03, right=0.97, left=0.03)
    vis.plt.show()


def add_subplot_matrix(subplot_matrix, opts_subplots, sub_plots, collision_mask, contour_mask,
                       clean_obj_mask_solution_values, title):
    # finding the bad cells, which are the interior of the solid (excluding the contour)
    interior_mask = an.subtract_contour_from_solid(solid=collision_mask, contour=contour_mask)

    # ============ Error subplots ============
    cbar_percentile_list = [95, 99]
    # Error all ranks
    if clean_obj_mask_solution_values:
        subplot_matrix = vis.correct_color_bar_2d(M=subplot_matrix, interior_mask=interior_mask)

    vis.add_subplot(M=subplot_matrix, title=title, sub_plots=sub_plots, opts_subplots=opts_subplots,
                    cbar_percentile_list=cbar_percentile_list, cmap="rocket")


if __name__ == '__main__':
    # UNCOMMENT TO RUN THE DEMOS...

    # 1. Enforcing Neumann boundary using reflection tiles for the Poisson solve suing Poisson filters convolution
    # UNCOMMENT------------------------------------------------
    # demo_wall_neumann_tiled_mirrored_2d(order=15, rank=8)

    # 2. Break down of a Poisson filters solve with Neumann boundary for individual rank contributions.
    #     With wall and complex object Neumann boundary treatment.
    # UNCOMMENT------------------------------------------------
    # multi_modal_solve_neumann_complex_object_2d(order=60, rank=4)

    # 3. Experimenting with various scenarios with different truncation values and Poisson filters order
    #     (target Jacobi iteration).
    # Solving with current settings might take a while, be patient!
    # UNCOMMENT------------------------------------------------
    demo_versatile_error_scenarios_neumann_2d()

