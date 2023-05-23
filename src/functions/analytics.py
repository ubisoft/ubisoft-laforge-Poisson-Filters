""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`
"""

import sys
import cv2
import scipy
import scipy.ndimage
import numpy as np
import src.helper.commons as com
import src.functions.decompositions as dec
import src.functions.mathlib as mlib
import src.functions.generator as gen

sys.path.insert(0, './')
# sys.path.insert(0, '../../')
image_folder_path = '../../images/'


def generate_collision_mask(M, path, load_shape_boundary):
    """ Create a 2D collision masks with the same size as the input data matrix. Walls are marked as both contours and \
    solids. Do cell marking for solid/contour/data for complex collider shapes, if loaded.

    It is optional to load a complex shape, in which case we do proper sampling to best match the resolutions \
    of the loaded image and the domain matrix.

    The mask marking convention:
        - data domain = 0
        - solid = 1
        - contour = 2 (still part of solid, but with a special flag)

    :param ndarray M: 2D data domain
    :param str path: path to load the image of the complex object
    :param bool load_shape_boundary: load a complex object shape from file
    :return: a collision mask with +1 flags for solid, and a contour mask for the contours around the
        solid object with +2 flags.
    """

    size_x = M.shape[0]
    size_y = M.shape[1]

    image = cv2.imread(path,  cv2.IMREAD_GRAYSCALE)
    image_size_x = image.shape[0]
    image_size_y = image.shape[1]

    ratio_x = image_size_x/size_x
    ratio_y = image_size_y/size_y

    solid_mask = np.zeros(shape=(size_x, size_y))
    contour_mask = np.zeros(shape=(size_x, size_y))

    # Set flags[i,j] if obstacle in cell [i,j]
    for i in range(size_x):
        for j in range(size_y):

            # Walls
            if i == 0 or i == size_x-1 or j == 0 or j == size_y-1:
                mlib.mark_solid_2d(mask=solid_mask, i=i, j=j)
                mlib.mark_contour_2d(mask=contour_mask, i=i, j=j)

            # Complex object
            if load_shape_boundary:
                im_index_i = round((i * 1.0 + 0.5) * ratio_x)
                im_index_j = round((j * 1.0 + 0.5) * ratio_y)
                if image[im_index_i, im_index_j] != 0:
                    mlib.mark_solid_2d(mask=solid_mask, i=i, j=j)

    # marking the contours:
    # it is a contour if the current cell is solid and has at least one
    # active domain neighbour.
    # only done for a solid object. The walls are already added above. Exclude them.
    for i in range(1, size_x - 1):
        for j in range(1, size_y - 1):
            if mlib.is_solid_2d(mask=solid_mask, i=i, j=j) and (mlib.is_active_domain(mask=solid_mask, i=i - 1, j=j)
                                                                or mlib.is_active_domain(mask=solid_mask, i=i + 1, j=j)
                                                                or mlib.is_active_domain(mask=solid_mask, i=i, j=j - 1)
                                                                or mlib.is_active_domain(mask=solid_mask, i=i, j=j + 1)):
                mlib.mark_contour_2d(mask=contour_mask, i=i, j=j)

    return solid_mask, contour_mask


def subtract_contour_from_solid(solid, contour):
    """Given a 2D solid object mask and its corresponding contour, extract the interior
    part of the solid maks by subtracting the two masks.

    :param ndarray solid: 2d input matrix
    :param ndarray contour: 2d input matrix
    :return: new 2d matrix with marked interior cells by :code:`1`, else :code:`0`
    """
    assert solid.shape == contour.shape
    m, n = solid.shape

    Out = np.zeros_like(solid)

    for i in range(0, m):
        for j in range(0, n):
            if mlib.is_solid_2d(solid, i=i, j=j) and not mlib.is_contour_2d(mask=contour, i=i, j=j):
                Out[i, j] = 1

    return Out


def make_special_data_central_square(size, radius, value):
    """Make a 2D data matrix with a non-zero box at the center and zero everywhere else.

    :param tuple size: base size of the square matrix
    :param float radius: not a circle radius, but the span length of the non-zeros in the central box. Take it as base \
        square length.
    :param float value: non-zero value to be assigned to the pattern
    :return: 2d data matrix with a non-zero box pattern at the center
    """
    data = np.zeros(shape=size)

    rows = size[0]
    cols = size[1]
    cx = int(rows / 2.)
    cy = int(cols / 2.)

    radius = min(0.5, radius) # make sure it does not exceed the matrix size
    offset = int(radius * (min(rows, cols)))
    data[cx - offset: cx + offset + 1, cy - offset: cy + offset + 1] = value

    return data


def prepare_padded_input_matrix(opts):
    """
    Cook up a data matrix (2D or 3D), either with randomized or constant values, with proper dynamic wall padding.

    We expand the matrix to account for convolution shrinkage. This helps with \
    comparing the Jacobi and the Poisson kernel solutions as the two might end up \
    having different sizes.

    .. note::
        Check out :func:`helper.commons.DataMatrixMode` for choices of making the data matrix.

    :param OptionsGeneral opts:
    :return:
        - :code:`data_padded`: padded (expanded) data matrix
        - :code:`data_domain`: original un-padded data matrix
        - :code:`padding_size`: computed padding size due to convolution shrinkage
    """

    # Dynamic padding
    # final data matrix size = base size + padding based on itr value
    kernel_size = mlib.get_kernel_effective_size(opts=opts)
    print(f'Kernel size {kernel_size}')

    padding_size = 0
    if opts.boundary.dynamic_padding:
        # dynamic padding, securing one cell padding for convolution
        padding_size = max(mlib.compute_conv_padding(kernel_size), com.get_default_padding())

    data_rows = opts.input.shape[0] + 2 * padding_size
    data_cols = opts.input.shape[1] + 2 * padding_size
    data_depth = None # 3D
    if com.is_solver_3d(dim=opts.solver.dim):
        data_depth = opts.input.shape[2] + 2 * padding_size

    # default: zero
    if com.is_solver_2d(dim=opts.solver.dim):
        data = np.zeros(shape=(data_rows, data_cols))
    elif com.is_solver_3d(dim=opts.solver.dim):
        data = np.zeros(shape=(data_rows, data_cols, data_depth))
    else:
        assert False, "Unknown Solver Dimension"

    # Constructing an example matrix
    # ------ 2D -------
    if com.is_solver_2d(dim=opts.solver.dim):
        if opts.input.mode == com.DataMatrixMode.RANDOMIZE_INT:
            np.random.seed(opts.input.rand_seed)
            data = np.random.randint(low=opts.input.rand_range[0], high=opts.input.rand_range[1],
                                     size=(data_rows, data_cols)).astype(float)

        elif opts.input.mode == com.DataMatrixMode.RANDOMIZE_FLOAT_UNIFORM:
            np.random.seed(opts.input.rand_seed)
            data = np.random.uniform(low=opts.input.rand_range[0], high=opts.input.rand_range[1],
                                     size=(data_rows, data_cols)).astype(float)

        elif opts.input.mode == com.DataMatrixMode.RANDOMIZE_FLOAT_GAUSSIAN:
            np.random.seed(opts.input.rand_seed)
            data = np.random.uniform(low=opts.input.rand_range[0], high=opts.input.rand_range[1],
                                     size=(data_rows, data_cols)).astype(float)
            data = scipy.ndimage.gaussian_filter(data, 4)

        elif opts.input.mode == com.DataMatrixMode.CONST_VALUE:
            data = opts.input.const_input * np.ones((data_rows, data_cols))

        elif opts.input.mode == com.DataMatrixMode.SPECIAL_PATTERN:

            if opts.input.special_pattern == com.DataMatrixPattern.CENTRAL_SQUARE:
                data = make_special_data_central_square(size=(data_rows, data_cols),
                                                        radius=opts.input.pattern_params.radius,
                                                        value=opts.input.const_input)

        elif opts.input.mode == com.DataMatrixMode.LINEAR_RANGE_FULL:
            data = np.arange(start=0, stop=data_rows * data_cols, dtype=float).reshape(data_rows, data_cols)

        elif opts.input.mode == com.DataMatrixMode.LINEAR_RANGE_4BLOCK_SYMMETRIC_MIRRORED:
            odd_rows = False if data_rows % 2 == 0 else True
            odd_cols = False if data_cols % 2 == 0 else True

            rows_quad = int(data_rows / 2)
            cols_quad = int(data_cols / 2)

            if odd_rows:
                rows_quad += 1
            if odd_cols:
                cols_quad += 1

            data_quad = np.arange(start=0, stop=rows_quad * cols_quad, dtype=float).reshape(rows_quad, cols_quad)
            # forcing symmetric along diagonal
            data_quad = np.maximum(np.tril(data_quad), np.triu(data_quad.T))

            data = mlib.make_4_tiles_mirrored_2d(M=data_quad, is_odd=True if odd_rows or odd_cols else False)

        else:
            assert False, "Unknown Data Matrix Mode"

    # ------ 3D -------
    elif com.is_solver_3d(dim=opts.solver.dim):
        if opts.input.mode == com.DataMatrixMode.RANDOMIZE_INT or opts.input.mode == \
                com.DataMatrixMode.RANDOMIZE_FLOAT_UNIFORM:
            np.random.seed(opts.input.rand_seed)
            data = np.random.randint(low=opts.input.rand_range[0], high=opts.input.rand_range[1],
                                     size=(data_rows, data_cols, data_depth)).astype(float)

        elif opts.input.mode == com.DataMatrixMode.CONST_VALUE:
            data = opts.input.const_input * np.ones((data_rows, data_cols, data_depth))
        else:
            assert False, "Other modes are not implemented for 3D"

    # ------ UNKNOWN DIM -------
    else:
        assert False, "Unknown Solver Dimension"

    if opts.input.force_symmetric_unit_range:
        data = data - np.mean(data)
        data = data / (np.max(data) - np.min(data)) * 2

    # apply padding
    # ------ 2D -------
    if com.is_solver_2d(dim=opts.solver.dim):
        data_padded = mlib.set_padding_2d(M=data, padding_size=padding_size, padding_value=opts.boundary.val)
        # trimmed input domain
        data_unpadded = data_padded[padding_size: data_rows - padding_size, padding_size: data_cols - padding_size]

    # ------ 3D -------
    elif com.is_solver_3d(dim=opts.solver.dim):
        data_padded = mlib.set_padding_3d(M=data, padding_size=padding_size, padding_value=opts.boundary.val)
        # trimmed input domain
        data_unpadded = data_padded[padding_size: data_rows - padding_size, padding_size: data_cols - padding_size,
                                    padding_size: data_depth - padding_size]

    # ------ UNKNOWN DIM -------
    else:
        assert False, "Unknown Solver Dimension"

    return data_padded, data_unpadded, padding_size


def solve_poisson_2d(M, poisson_kernel, opts, skip_margin=0):
    """General solver for 2D Poisson equation, for either *inverse* or *forward* Poisson.

    The function supports solving the Poisson equation with the full kernel (input), reduced kernel (computed here), \
    or separable Poisson filters (computed here), depending on how parameters are set in the input options. Truncation \
    parameters will be retrieved from options as well.

    .. note::
        To set rank, order, and other relevant parameters you need to pack :code:`OptionsKernel` and
        :code:`OptionsReduction` in :code:`OptionsGeneral` and send it to this function. \
        These dataclasses are in :code:`helper.commons.py`.
        To see how to pack options, look at main demos, or see :func:`helper.commons.generic_options`.

    :param ndarray M: input 2D domain matrix
    :param ndarray poisson_kernel: precomputed 2D Poisson kernel (*inverse* or *forward*)
    :param OptionsGeneral opts:
    :param int skip_margin: number of lateral elements to skip in the convolution. This helps with saving computation
        time when having redundant padding (**Default=** :code:`0`)
    :return: solution to the 2D Poisson equation, computed safe rank
    """

    safe_rank = None

    # Low Rank
    if opts.reduction.reduce is True and opts.reduction.rank is not None:
        safe_rank = dec.rank_safety_clamp_2d(itr=opts.kernel.itr, rank=opts.reduction.rank, zero_init=opts.solver.zero_init)
        # reduce
        U, S, VT, low_rank = dec.poisson_svd_2d(P=poisson_kernel, rank=safe_rank)

        # solve
        if opts.reduction.use_separable_filters:  # solve using Poisson filters
            poisson_solution = mlib.solve_poisson_separable_filters_from_components_2d(M=M, U=U, S=S, VT=VT,
                                                                                       rank=safe_rank,
                                                                                       trunc_method=opts.reduction.truncation_method,
                                                                                       trunc_factor=opts.reduction.truncation_value,
                                                                                       skip_margin=skip_margin)
        else:  # solve using full low rank kernel, no truncation
            poisson_solution = mlib.solve_poisson_full_kernel_2d(M=M, kernel=low_rank, skip_margin=skip_margin)

    else:  # solve using full unreduced kernel
        poisson_solution = mlib.solve_poisson_full_kernel_2d(M=M, kernel=poisson_kernel, skip_margin=skip_margin)

    return poisson_solution, safe_rank


def compare_jacobi_poisson_neumann_edge_mirrored_correction_2d(opts):
    """Compare ground truth Jacobi solution to Poisson filters solution (*only for inverse*) using tiled mirror
    method to treat the Neumann boundary. This is the core principle of the *mirror marching* algorithm
    proposed in the paper to do the boundary treatment.

    The mirrored data tiles are automatically computed based on the target Jacobi iteration to make sure there is
    enough padding for the base data matrix when doing the convolution. \
    See :func:`demos.boundary.demo_wall_neumann_tiled_mirrored_2d` for an example demo.

    :param OptionsGeneral opts:
    :return:
        - :code:`jacobi_solution` : ground truth solution
        - :code:`poisson_solution` : Poisson filter solution
        - :code:`data_domain` : created data matrix
        - :code:`data_mirrored` : expanded data matrix used in the Poisson filters method
        - :code:`safe_rank` : computed safe rank after kernel reduction
        - :code:`err_abs` : absolute error
        - :code:`err_rel` : relative error
    """

    data_padded, data_domain, padding_size = prepare_padded_input_matrix(opts=opts)

    boundary_padding = 1
    padding_size += boundary_padding

    data_mirrored = data_domain
    num_of_lateral_neighbours = 0
    data_unpadded_size = data_domain.shape[0]

    # keep adding mirrored padding until we have enough to cover for the convolution shrinkage.
    # this will be powers of 3 so often we end up having more padding than needed. An exact padding
    # will need to match the possion kernel target iteration.
    num_of_recurseve_tiling = 0
    while num_of_lateral_neighbours * data_unpadded_size < padding_size:
        data_mirrored = mlib.make_9_tiles_mirrored_2d(M=data_mirrored)
        num_of_lateral_neighbours += 3**num_of_recurseve_tiling
        num_of_recurseve_tiling += 1

    opts.boundary.padding_size = num_of_lateral_neighbours * data_unpadded_size  # updating the options

    print(f'Input data size {data_unpadded_size}')
    print(f'# Lateral neighbour blocks {num_of_lateral_neighbours}')
    print(f'Total mirrored padding {opts.boundary.padding_size}')

    # ------ Solve -----

    # Jacobi solve

    # jacobi data: single padding Jacobi on raw data
    data_jacobi_single_padding = mlib.expand_with_padding_2d(M=data_domain, pad_size=1, pad_value=opts.boundary.val,
                                                        opts_boundary_detailed_wall=opts.boundary)

    jacobi_solution = mlib.solve_jacobi_single_padding_obj_collision_2d(M=data_jacobi_single_padding, opts=opts)

    # Full Poisson kernel solve
    poisson_kernel = gen.poisson_kernel_2d(opts=opts)
    # no correction
    if opts.solver.solver_type == com.PoissonSolverType.FORWARD:  # use good old dynamic padding
        assert False, "Not working on Forward yet...."

    # raw solution, convolve inside the domain
    # make a convolution pass excluding the boundary, reading it but not updating it.
    redundant_padding = opts.boundary.padding_size - padding_size

    poisson_solution, safe_rank = solve_poisson_2d(M=data_mirrored, poisson_kernel=poisson_kernel,
                                                   opts=opts, skip_margin=redundant_padding)

    print(f'Redundant padding to skip in convolution {redundant_padding}')

    err_abs, err_rel = mlib.compute_abs_rel_error(M1=jacobi_solution, M2=poisson_solution)

    return jacobi_solution, poisson_solution, data_domain, data_mirrored, \
           safe_rank, err_abs, err_rel


def compute_adaptive_truncation_factors(opts, filters_iterations=None):
    """Given a list of target iterations, collect info about the effective maximum number of \
    rank and filter size required for each iteration, for either 2D or 3D case.

    In case of :code:`filters_iterations=None` force generate the filters, for a range of target iterations \
    :code:`[1, max_itr]`, where :code:`max_itr` is the same as target iteration set in the options.

    See :func:`demos.convergence.demo_adaptive_truncation_info_2d` for the example demo.

    .. warning::
        Filters are assumed to be already truncated upon load or generation.

    :param OptionsGeneral opts:
    :param List[int] filters_iterations: list of target iterations. \
        If filters are already available use this parameter, else (:code:`None`) force compute them.
    :return: output will be a :code:`(max_itr, 3)` shape, \
        with format :code:`(max_effective_rank, max_filter_size, actual_filter_size)`.
        If in dynamic truncation mode return the analysis, else a zero vector with the size of :code:`max_itr`
    """
    max_itr = opts.kernel.itr
    rank_and_filter_info = np.zeros(shape=(max_itr, 3), dtype=np.int64)

    # dynamic truncation only works with a fixed threshold mode
    assert opts.reduction.truncation_method == com.TruncationMode.FIXED_THRESHOLD
    # only makes sense in the reduced mode
    assert opts.reduction.reduce
    # only makes sense with separable filters
    assert opts.reduction.use_separable_filters

    preserve_shape = True  # make sure you preserve the shape for the truncation to work

    print("Computing rank truncation info for dynamic truncation...")

    if filters_iterations is None:

        print('Dynamic truncation analysis for generated filters on the fly...')

        for k in range(0, max_itr):  # including the last iteration
            # overwriting options with current iteration
            opts.kernel.itr = k + 1

            if com.is_solver_2d(dim=opts.solver.dim):
                # dynamic truncation with a fixed threshold is already done inside the filter generators
                v_hor, v_ver, safe_rank = dec.poisson_filters_2d(opts)
                # horizontal and vertical filters are symmetrical, so pick one
                filters = v_hor

            elif com.is_solver_3d(dim=opts.solver.dim):
                filters = dec.poisson_filters_3d(opts, rank_filter_reorder=True)
                safe_rank = opts.reduction.rank # always safe for 3D
                assert False, "Filters need proper reshaping to work with the rest of the code."

            else:
                assert False, "Unknown Solver Dimension"

            max_effective_rank, max_filter_size, actual_filter_size = mlib.find_max_rank_and_filter_size(ranked_filters=filters,
                                                                                                         safe_rank=safe_rank)

            rank_and_filter_info[k][0] = max_effective_rank
            rank_and_filter_info[k][1] = max_filter_size
            rank_and_filter_info[k][2] = actual_filter_size # assuming preserving shape, all ranked filters have
            # the same size for the current iteration

    else:
        print('Dynamic truncation analysis on existing filters...')

        # format (iterations, ranks, filters)
        num_itr = len(filters_iterations)

        for k in range(num_itr):  # including the last iteration
            # overwriting options with current iteration
            filters = filters_iterations[k]
            # shape(rank, filter)
            safe_rank = filters.shape[0]

            # setting small values to zero before doing the analysis (this is the dynamic truncation part)
            for i in range(safe_rank):
                # do it for every rank separately
                filters[i] = mlib.truncate_fixed_threshold_1darray(arr=filters[i], cut_off=opts.reduction.truncation_value,
                                                                   preserve_shape=preserve_shape)

            max_effective_rank, max_filter_size, actual_filter_size = mlib.find_max_rank_and_filter_size(
                ranked_filters=filters, safe_rank=safe_rank)

            rank_and_filter_info[k][0] = max_effective_rank
            rank_and_filter_info[k][1] = max_filter_size
            rank_and_filter_info[k][2] = actual_filter_size  # assuming preserving shape, all ranked filters have
            # the same size for the current iteration

    return rank_and_filter_info


def compare_jacobi_3methods_2d(opts):
    """Compare solutions to the 2D Poisson equation for 3 cases: Jacobi in the *matrix form* vs *vector form* \
    vs *Poisson Filters* method (convolution, reduced or full kernel). \
    No boundary treatment is done, which means data domain is treated as infinite domain.

    See :func:`demos.convergence.demo_3methods_comparison_no_bc_2d` for the example demo.

    :param OptionsGeneral opts:
    :return:
        - :code:`err_abs_jm_vs_jv` : absolute error Jacobi matrix form vs Jacobi vector form
        - :code:`err_rel_jm_vs_jv` : relative error Jacobi matrix form vs Jacobi vector form in percent
        - :code:`err_abs_jv_vs_pk` : absolute error Jacobi vector form vs Poisson kernel
        - :code:`err_rel_jv_vs_pk` : relative error Jacobi vector form vs Poisson kernel in percent
        - :code:`err_abs_jm_vs_pk` : absolute error Jacobi matrix form vs Poisson kernel
        - :code:`err_rel_jm_vs_pk` : relative error Jacobi matrix form vs Poisson kernel in percent
        - :code:`jacobi_solution_matrix_form` : Jacobi solution in matrix form
        - :code:`jacobi_solution_vector_form` : Jacobi solution in vector form
        - :code:`poisson_solution` : Poisson kernel solution
        - :code:`data_padded` : generate sample data
    """
    dim = com.SolverDimension.D2

    # Making an example data matrix as the input. data is padded, but data_domain is smaller original one
    data_padded, data_domain, padding_size = prepare_padded_input_matrix(opts=opts)
    opts.boundary.padding_size = padding_size  # updating the options

    # Solve..

    # Full Poisson kernel solve
    poisson_kernel = gen.poisson_kernel_2d(opts=opts)
    poisson_solution, safe_rank = solve_poisson_2d(M=data_padded, poisson_kernel=poisson_kernel, opts=opts)

    # Jacobi matrix form

    # Jacobi solve with dynamic padding
    jacobi_solution_matrix_form, residuals_jacobi_matrix_form = \
        mlib.solve_jacobi_matrix_form_no_boundary_2d(M=data_padded, opts=opts)

    # Jacobi vector form
    base_size = data_padded.shape[0]  # assuming a square 2D matrix
    laplacian, flat_size = mlib.construct_laplacian_nd_vector_friendly(dim=dim, base_size=base_size)
    jacobi_solution_vector_form, residuals_jacobi_vector_form = mlib.solve_jacobi_vector_form(
        A=laplacian,
        b=data_padded.reshape(-1, 1),
        max_iterations=opts.kernel.itr,
        is_subdomain_residual=False,
        sub_domain_shape=(0, 0),
        dim=dim)

    # reshaping solution vector to get it in the matrix form
    jacobi_solution_vector_form = jacobi_solution_vector_form.reshape(jacobi_solution_matrix_form.shape)

    # =============== Errors ===============

    # ============ extra padding +1  ==========
    # Note: we need one extra cell trimming from sides because of using pure Poisson filters
    # in comparison. We could have easily added the incomplete Laplacian kernels for the corners
    # and the cells next to the wall, but this will make it inconsistent with the Jacobi function
    # used to generate the kernels.

    #  Removing the effect of BC (vector form has BC but the matrix form does not)
    jacobi_solution_matrix_form = mlib.trim(M=jacobi_solution_matrix_form, size=padding_size + 1)
    jacobi_solution_vector_form = mlib.trim(M=jacobi_solution_vector_form, size=padding_size + 1)
    poisson_solution = mlib.trim(M=poisson_solution, size=1)

    # =============== comparing jacobis ==============
    err_abs_jm_vs_jv = np.abs(jacobi_solution_matrix_form - jacobi_solution_vector_form).astype(np.double)
    epsilon = 1e-20  # regularization to avoid division by zero
    epsilon_m = epsilon * np.ones_like(jacobi_solution_matrix_form).astype(np.double)
    err_rel_jm_vs_jv = 100. * np.abs(err_abs_jm_vs_jv / (jacobi_solution_matrix_form + epsilon_m))  # relative error %

    # =============== comparing jacobi vector form and poisson kernel ============== "
    err_abs_jv_vs_pk = np.abs(poisson_solution - jacobi_solution_vector_form).astype(np.double)
    epsilon_m = epsilon * np.ones_like(jacobi_solution_vector_form).astype(np.double)
    err_rel_jv_vs_pk = 100. * np.abs(err_abs_jv_vs_pk / (jacobi_solution_vector_form + epsilon_m))  # relative error %

    # =============== comparing jacobi matrix form and poisson kernel ============== "
    err_abs_jm_vs_pk = np.abs(poisson_solution - jacobi_solution_matrix_form).astype(np.double)
    epsilon_m = epsilon * np.ones_like(jacobi_solution_matrix_form).astype(np.double)
    err_rel_jm_vs_pk = 100. * np.abs(err_abs_jm_vs_pk / (jacobi_solution_matrix_form + epsilon_m))  # relative error %

    return err_abs_jm_vs_jv, err_rel_jm_vs_jv, err_abs_jv_vs_pk, err_rel_jv_vs_pk, err_abs_jm_vs_pk, err_rel_jm_vs_pk, \
        jacobi_solution_matrix_form, jacobi_solution_vector_form, poisson_solution, data_padded


def compare_jacobi_3methods_neumann_2d(opts):
    """Compare solutions to the 2D Poisson equation for 3 cases: Jacobi in the *matrix form* vs *vector form* \
    vs *Poisson Filters* method (convolution, reduced or full kernel).

    This is with Neumann boundary treatment.

    See :func:`demos.convergence.demo_3methods_comparison_no_bc_2d` for the example demo.

    :param OptionsGeneral opts:
    :return:
        - :code:`err_abs_jm_vs_jv` : absolute error Jacobi matrix form vs Jacobi vector form
        - :code:`err_rel_jm_vs_jv` : relative error Jacobi matrix form vs Jacobi vector form in percent
        - :code:`err_abs_jv_vs_pk` : absolute error Jacobi vector form vs Poisson kernel
        - :code:`err_rel_jv_vs_pk` : relative error Jacobi vector form vs Poisson kernel in percent
        - :code:`err_abs_jm_vs_pk` : absolute error Jacobi matrix form vs Poisson kernel
        - :code:`err_rel_jm_vs_pk` : relative error Jacobi matrix form vs Poisson kernel in percent
        - :code:`jacobi_solution_matrix_form` : Jacobi solution in matrix form
        - :code:`jacobi_matrix_form_residuals` : Jacobi solution in matrix form residuals
        - :code:`jacobi_solution_vector_form` : Jacobi solution in vector form
        - :code:`jacobi_vector_form_residuals` : Jacobi solution in vector form residuals
        - :code:`poisson_solution` : Poisson kernel solution
        - :code:`poisson_residual` : Poisson kernel solution residuals
        - :code:`data_padded` : generate sample data
    """
    assert opts.solver.dim == com.SolverDimension.D2
    dim = com.SolverDimension.D2

    use_neumann = True
    assert use_neumann, "using wall aware PF solver with neumann bc. This has to be to true for consistent results."
    is_sub_domain_residual = True

    # Making an example data matrix as the input. data is padded, but data_domain is smaller original one
    data_padded, data_domain, padding_size = prepare_padded_input_matrix(opts=opts)

    # single padding Jacobi on raw data to add walls
    wall_padding = 1
    data_wall_padding = mlib.expand_with_padding_2d(M=data_domain,
                                                    pad_size=wall_padding,
                                                    pad_value=opts.boundary.val,
                                                    opts_boundary_detailed_wall=opts.boundary)

    # getting the residula before solve
    # safe domain for computing residual: cut one from each side to exclude the wall padding
    num_exclude_cell = 1
    sub_domain_residual = (data_wall_padding.shape[0] - 2 * num_exclude_cell,
                           data_wall_padding.shape[1] - 2 * num_exclude_cell)

    # =============== Solve =============== ..

    # ----Poisson filters with marching bc
    v_hor, v_ver, safe_rank = dec.poisson_filters_2d(opts=opts)
    poisson_solution = mlib.solve_poisson_separable_filters_wall_aware_2d(M=data_wall_padding,
                                                                          filter_hor=v_hor, filter_ver=v_ver,
                                                                          safe_rank=safe_rank,
                                                                          opts=opts)
    # getting the poisson residual
    poisson_residual = mlib.compute_residual_subdomain_2d(X=poisson_solution,
                                                          B=data_wall_padding,
                                                          sub_shape=sub_domain_residual)

    # Remove padding - the padding is there only because of matching with the vector form. You can remove it if
    # just comparing jacboi matrix form and PF
    poisson_solution = mlib.trim(M=poisson_solution, size=wall_padding)

    # ----Jacobi matrix form

    # Jacobi solve with wall padding - suppprts neumann bc
    jacobi_solution_matrix_form, jacobi_matrix_form_residuals = mlib.solve_jacobi_single_padding_only_wall_2d(
        M=data_wall_padding,
        sub_shape_residual=sub_domain_residual,
        opts=opts)

    # Remove padding - the padding is there only because of matching with the vector form. You can remove it if
    # just comparing jacboi matrix form and PF
    jacobi_solution_matrix_form = mlib.trim(M=jacobi_solution_matrix_form, size=wall_padding)

    # ----Jacobi vector form with neumann bc

    base_size = data_domain.shape[0]  # assuming a square 2D matrix
    laplacian, flat_size = mlib.construct_laplacian_nd_vector_friendly(dim=dim, base_size=base_size,
                                                                       positive_definite=False,
                                                                       neumann=True)
    jacobi_solution_vector_form, jacobi_vector_form_residuals = mlib.solve_jacobi_vector_form(
        A=laplacian,
        b=data_domain.reshape(-1, 1),
        max_iterations=opts.kernel.itr,
        is_subdomain_residual=is_sub_domain_residual,
        sub_domain_shape=sub_domain_residual,
        dim=dim,
        warm_start=False)

    # reshaping solution vector to get it in the matrix form
    jacobi_solution_vector_form = jacobi_solution_vector_form.reshape(jacobi_solution_matrix_form.shape)

    # =============== Errors ===============

    # ============ extra padding +1  ==========
    # Note: we need one extra cell trimming from sides because of using pure Poisson filters
    # in comparison. We could have easily added the incomplete Laplacian kernels for the corners
    # and the cells next to the wall, but this will make it inconsistent with the Jacobi function
    # used to generate the kernels.

    # # Removing the effect of BC (vector form has BC but the matrix form does not)
    # jacobi_solution_vector_form = trim(M=jacobi_solution_vector_form, size=padding_size + 1)

    # =============== comparing jacobis ==============
    err_abs_jm_vs_jv = np.abs(jacobi_solution_matrix_form - jacobi_solution_vector_form).astype(np.double)
    epsilon = 1e-20  # regularization to avoid division by zero
    epsilon_m = epsilon * np.ones_like(jacobi_solution_matrix_form).astype(np.double)
    err_rel_jm_vs_jv = 100. * np.abs(err_abs_jm_vs_jv / (jacobi_solution_matrix_form + epsilon_m))  # relative error %

    # =============== comparing jacobi vector form and poisson kernel ============== "
    err_abs_jv_vs_pk = np.abs(poisson_solution - jacobi_solution_vector_form).astype(np.double)
    epsilon_m = epsilon * np.ones_like(jacobi_solution_vector_form).astype(np.double)
    err_rel_jv_vs_pk = 100. * np.abs(err_abs_jv_vs_pk / (jacobi_solution_vector_form + epsilon_m))  # relative error %

    # =============== comparing jacobi matrix form and poisson kernel ============== "
    err_abs_jm_vs_pk = np.abs(poisson_solution - jacobi_solution_matrix_form).astype(np.double)
    epsilon_m = epsilon * np.ones_like(jacobi_solution_matrix_form).astype(np.double)
    err_rel_jm_vs_pk = 100. * np.abs(err_abs_jm_vs_pk / (jacobi_solution_matrix_form + epsilon_m))  # relative error %

    return err_abs_jm_vs_jv, err_rel_jm_vs_jv, err_abs_jv_vs_pk, err_rel_jv_vs_pk, err_abs_jm_vs_pk, err_rel_jm_vs_pk, \
        jacobi_solution_matrix_form, jacobi_matrix_form_residuals, \
        jacobi_solution_vector_form, jacobi_vector_form_residuals, \
        poisson_solution, poisson_residual, data_padded

