""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`

    .. _mathlib_technicals:

    Technicals
    ===============

    The Poisson solver has a linear formulation like in the typical :math:`Ax=b` but with a difference of replacing
    vectors :math:`x` and :math:`b` with matrices, and replacing matrix-vector multiplication with a convolution
    operator. We use a proper notation with a difference naming convention:

    :math:`Ax = b \\leftrightarrow L*X = B`

    where:
        - :math:`L`: Laplacian operator
        - :math:`*`: convolution operator
        - :math:`X`: given or unknown matrix, depending on the Poisson equation we are solving for.
        - :math:`B`: given or unknown *rhs* matrix, depending on the Poisson equation we are solving for.

    LHS and RHS are both in *matrix-forms*, i.e. all matrices/tensors have the same dimension.

    .. note::
        We call :math:`Ax=b` the *vector-form* and :math:`L*X=B` the *matrix-form*.

    .. note::
        The Poisson kernel (and subsequently its separable filters) are computed based on the *matrix-form*, \
        so for an infinite domain without any boundary treatment the solution using the Poisson kernel and the \
        one from the *matrix-form* must match to the machine precision.

    The linear solver works with two versions of the Poisson equation, namely *inverse* and *forward* :

        - **Inverse** : An example is Poisson-pressure where :math:`B` is the input divergence and :math:`X` is \
        the output pressure.
        - **Forward** : An example is diffusion where :math:`X` is the input density and the output :math:`B` is \
        the diffused quantity.

    Depending on what we are solving for, we have different setups for input/output:

        1. **Inverse Poisson equation** : Given :math:`M` as input in :math:`L*X = M`, obtain solution :math:`X` \
            by implicitly approximating :math:`L^{-1}` in :math:`X = L^{-1}*M`.
        2. **Forward Poisson equation** : Given :math:`M` as input, obtain the solution :math:`B` to the diffusion \
            equation, i.e. perform :math:`L*M = B` , where :math:`B` is the output.

    The solution is computed using an 'implicit' finite difference method, Just like Jacobi.

    .. note::
        Note how the input matrix :math:`M` changes role in :math:`L*X=B` based on the type of the Poisson \
        equation setup.

    We use multi-rank Poisson filters for a given rank. Given the Poisson equation in the *matrix-form*
    :math:`L*X=B`, the Poisson kernel :math:`L` (in forward setup) and its inverse :math:`L^{-1}` (in inverse setup)
    are already baked into the Poisson filters. Just provide the input data matrix and the corresponding
    filters matching the formulation setup you are interested in.

    In general, we can replace :math:`L` and :math:`L^{-1}` with a unified kernel :math:`F` that operates on a
    data domain matrix (or tensor), and perform Eigen decomposition on :math:`F` to get the Poisson filters.

    .. _mathlib_convolution_order:

    Convolution Order of Poisson Filters
    -------------------------------------
    Poisson filters are used in cascaded convolutions to get the solution.

    In 3D, we have

    :math:`F * M \\approx \\displaystyle\\sum_{r=1}^n f_{v_r} * (f_{h_r} * (f_{d_r} * M))`

    where
        - :math:`F` - Full Poisson kernel (either :math:`L` or :math:`L^{-1}`)
        - :math:`M` - Input data field
        - :math:`f_v` - Vertical filter
        - :math:`f_h` - Horizontal filter
        - :math:`f_d` - Depth (fiber) filter
        - double subscript :math:`_r` means the filter corresponding the current rank
        - :math:`\\displaystyle\\sum_{r=1}^n` is multi-rank summation (i.e. modal solutions)

    The convolution order goes from the inner
    brackets to outer brackets, meaning first we need to convolve :math:`M` with the fiber filter, then
    convolve the results with the horizontal and vertical filters.

    For multi-rank convolution we have separate and independent convolutions passes on :math:`M`, then sum up
    the results. The summation comes from the Canonical Polyadic Decomposition (*CPD*) view in our matrix/tensor
    decomposition setup (different in 2D and 3D),
    which makes it possible to have rank-1 kernel convolutions to get modal solutions taking care of different
    frequencies in the data domain.

    .. warning::
        **DO NOT** feed input data matrix :math:`M` in outer bracket convolution.

        **ALWAYS** use the results of the previous convolution pass to do the next one.

    Also see :func:`solve_poisson_separable_filters_2d` and
    :func:`solve_poisson_separable_filters_wall_aware_3d` for the convolution order in 2D and 3D.


    .. note::
        The ground truth Jacobi solver we use in our code uses the same *matrix-form* setup as the Poisson \
        Filters/Kernel setup. It is a straightforward practice to establish the connection between Jacobi \
        *matrix-form* and the more widely known and used version, Jacobi *vector-form* .
"""

import numpy as np
import src.helper.commons as com
import src.functions.generator as gen
import sys

sys.path.insert(0, '../../')

# not currently used, but feel free to use them for matrix data generation
kernels_functions = {
    'multiquadric': lambda x: np.sqrt(x ** 2 + 1),
    'inverse': lambda x: 1.0 / np.sqrt(x ** 2 + 1),
    'gaussian': lambda x: np.exp(-x ** 2),
    'linear': lambda x: x,
    'quadric': lambda x: x ** 2,
    'cubic': lambda x: x ** 3,
    'quartic': lambda x: x ** 4,
    'quintic': lambda x: x ** 5,
    'thin_plate': lambda x: x ** 2 * np.log(x + 1e-10),
    'logistic': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -5, 5))),
    'smoothstep': lambda x: ((np.clip(1.0 - x, 0.0, 1.0)) ** 2.0) * (3 - 2 * (np.clip(1.0 - x, 0.0, 1.0)))
}


def get_1d_array_half_size(length, trunc_factor_percent, enforce_min_size=True):
    """ Given 1d array length and a truncation factor (%), compute the new half array size.
    Half size excludes the element in the middle. Minimum filter/kernel size should be 3, a half size of 0 means
    a convolution of a 1x1 element.

    :param int length: size of the 1d array
    :param float trunc_factor_percent: truncation factor in [0, 1]
    :param bool enforce_min_size: this is specially used for convolutional stuff. If True, minimum filter/kernel
        size should be 3, which is achieved by avoiding a half size of 0. This is necessary to remain consistent
        with the Jacobi 3x3 base kernel.
    :return:
        - **half_size** - new half size
        - **middle_index** - the index to the middle element

    """
    middle_index = int(length / 2.0)
    half_size = int((1.0 - trunc_factor_percent) * middle_index)

    if enforce_min_size:
        half_size = max(half_size, 1)  # minimum filter/kernel size should be 3, a half size of 0 means
    # a convolution of a 1x1 element

    return half_size, middle_index


def is_solid_2d(mask, i, j):
    """ Check if a certain position of a mask is solid.

    The marking convention:
        - data domain = 0
        - solid = 1
        - contour = 2 (still part of solid, but with a special flag)

    :param ndarray mask: mask matrix
    :param int i: row index
    :param int j: column index
    :return: True if it is solid or contour
    :rtype: bool
    """
    return mask[i, j] == 1 or mask[i, j] == 2


def is_wall_2d(M, i, j):
    """ Check if a certain position of a 2d domain matrix falls in walls.

    Walls are 1-cell wide paddings around the domain.

    :param ndarray M: domain matrix
    :param int i: row index
    :param int j: column index
    :return: True if it is inside the wall
    :rtype: bool
    """
    rows, cols = M.shape

    assert i < rows and j < cols, "Index larger than matrix size"

    in_wall = False
    in_wall = in_wall or (i == 0 or i == rows - 1)
    in_wall = in_wall or (j == 0 or j == cols - 1)

    return in_wall


def is_wall_3d(M, i, j, k):
    """ Check if a certain position of a 3d domain matrix falls in walls.

    Walls are 1-cell wide paddings around the domain.

    :param 3darray M: domain matrix
    :param int i: row index
    :param int j: column index
    :param int k: depth index
    :return: True if it is inside the wall
    :rtype: bool
    """
    rows, cols, depth = M.shape

    assert i < rows and j < cols and k < depth, "Index larger than matrix size"

    in_wall = False
    in_wall = in_wall or (i == 0 or i == rows - 1)
    in_wall = in_wall or (j == 0 or j == cols - 1)
    in_wall = in_wall or (k == 0 or k == depth - 1)

    return in_wall


def mark_solid_2d(mask, i, j):
    """ Mark a certain position of the mask to be solid.

    The marking convention:
        - data domain = 0
        - solid = 1
        - contour = 2 (still part of solid, but with a special flag)

    :param ndarray mask: mask matrix
    :param int i: row index
    :param int j: column index
    """
    mask[i, j] = 1


def mark_contour_2d(mask, i, j):
    """ Mark a certain position of the mask to be contour.

    The marking convention:
        - data domain = 0
        - solid = 1
        - contour = 2 (still part of solid, but with a special flag)

    :param ndarray mask: mask matrix
    :param int i: row index
    :param int j: column index
    """
    mask[i, j] = 2


def is_contour_2d(mask, i, j):
    """ Check if a certain position of the mask is contour.

    The marking convention:
        - data domain = 0
        - solid = 1
        - contour = 2 (still part of solid, but with a special flag)

    :param ndarray mask: mask matrix
    :param int i: row index
    :param int j: column index
    :return: True if the position is part of the contour
    :rtype: bool
    """
    return mask[i, j] == 2


def is_active_domain(mask, i, j):
    """ Check if a certain position of the mask is active (data) domain.

    The marking convention:
       - data domain = 0
       - solid = 1
       - contour = 2 (still part of solid, but with a special flag)

    :param ndarray mask: mask matrix
    :param int i: row index
    :param int j: column index
    :return: True if the position is part of the active (data) domain
    :rtype: bool
    """
    return mask[i, j] == 0


def compute_conv_padding(convolution_kernel_size):
    """ Compute the padding required to compensate for shrinkage due to convolution.

    :param int convolution_kernel_size: size of a square convolution kernel
    :return: required padding size
    :rtype: int
    """
    return int(convolution_kernel_size / 2.0)


def set_padding_2d(M, padding_size, padding_value):
    """ Set the padding region of the input 2d matrix with the padding value.

    :param ndarray M: input 2d matrix
    :param int padding_size: thickness of the padding region
    :param int padding_value: const value to be put in the padding region
    :return: padded matrix
    :rtype: ndarray
    """
    if padding_size > 0:
        M[0:padding_size, :] = padding_value
        M[-padding_size:, :] = padding_value
        M[:, 0:padding_size] = padding_value
        M[:, -padding_size:] = padding_value

    return M


def set_padding_3d(M, padding_size, padding_value):
    """ Set the padding region of the input 3d matrix with the padding value.

    :param ndarray M: input 3d matrix
    :param int padding_size: thickness of the padding region
    :param int padding_value: const value to be put in the padding region
    :return: padded matrix
    :rtype: ndarray
    """
    if padding_size > 0:
        M[0:padding_size, :, :] = padding_value
        M[-padding_size:, :, :] = padding_value
        M[:, 0:padding_size, :] = padding_value
        M[:, -padding_size:, :] = padding_value
        M[:, :, 0:padding_size] = padding_value
        M[:, :, -padding_size:] = padding_value

    return M


def get_sub_domain_shape(data_original, num_exclude_cell, dim):
    """ Compute the size of the subdomain data given the numbers of cells to exclude from each side.

    :param ndarray data_original: original data matrix, either 2d or 3d
    :param int num_exclude_cell: number of cells to exclude from each side
    :param SolverDimension dim: 2D or 3D
    :return: 2d or 3d shape
    :rtype: tuple
    """
    if com.is_solver_2d(dim):
        sub_domain_shape = (data_original.shape[0] - 2 * num_exclude_cell,
                            data_original.shape[1] - 2 * num_exclude_cell)

    elif com.is_solver_3d(dim):
        sub_domain_shape = (data_original.shape[0] - 2 * num_exclude_cell,
                            data_original.shape[1] - 2 * num_exclude_cell,
                            data_original.shape[2] - 2 * num_exclude_cell)
    else:
        assert False, "Unknown Solver Dimension"

    return sub_domain_shape


def expand_with_padding(M, pad_size, pad_value, dim, opts_boundary=None):
    """ Add equal padding to each side of the matrix with a fixed value (wall padding).

    :param ndarray M: 2d or 3d matrix
    :param int pad_size:
    :param int pad_value:
    :param SolverDimension dim: 2D or 3D
    :param OptionsBoundary opts_boundary: Optional. If available, enforce padding on specific walls only.
    :return: new padded matrix if pad_size != 0, else return the original matrix
    """
    if com.is_solver_2d(dim):
        return expand_with_padding_2d(M=M, pad_size=pad_size, pad_value=pad_value,
                                      opts_boundary_detailed_wall=opts_boundary)

    elif com.is_solver_3d(dim):
        return expand_with_padding_3d(M=M, pad_size=pad_size, pad_value=pad_value,
                                      opts_boundary_detailed_wall=opts_boundary)
    else:
        assert False, "Unknown Solver Dimension"


def expand_with_padding_2d(M, pad_size, pad_value, opts_boundary_detailed_wall=None):
    """ Add equal padding to each side of the matrix with a fixed value (wall padding).

    :param ndarray M: 2d matrix
    :param int pad_size:
    :param int pad_value:
    :param OptionsBoundary opts_boundary_detailed_wall: Optional. If available, enforce padding on specific walls only.
    :return: new padded matrix if :code:`pad_size != 0`, else return the original matrix
    """
    if pad_size == 0:
        return M

    rows, cols = M.shape
    rows_padded = rows + 2 * pad_size
    cols_padded = cols + 2 * pad_size
    P = np.zeros(shape=(rows_padded, cols_padded))

    # copying the interior
    P[pad_size: -pad_size, pad_size: -pad_size] = M[:, :]

    # copying the exterior values..

    # setting boundary values only for selected walls
    if opts_boundary_detailed_wall is not None:

        if opts_boundary_detailed_wall.up_wall:
            P[0:pad_size, :] = pad_value
        if opts_boundary_detailed_wall.down_wall:
            P[-pad_size:, :] = pad_value
        if opts_boundary_detailed_wall.left_wall:
            P[:, 0:pad_size] = pad_value
        if opts_boundary_detailed_wall.right_wall:
            P[:, -pad_size:] = pad_value

    else:  # treat all walls the same...

        P[0:pad_size, :] = pad_value
        P[-pad_size:, :] = pad_value
        P[:, 0:pad_size] = pad_value
        P[:, -pad_size:] = pad_value

    return P


def expand_with_padding_3d(M, pad_size, pad_value, opts_boundary_detailed_wall=None):
    """ Add equal padding to each side of the matrix with a fixed value (wall padding).

     :param ndarray M: 3d matrix
     :param int pad_size:
     :param int pad_value:
     :param OptionsBoundary opts_boundary_detailed_wall: Optional. If available, enforce padding on specific walls only.
     :return: new padded matrix if :code:`pad_size != 0`, else return the original matrix
     """
    if pad_size == 0:
        return M

    rows, cols, depth = M.shape
    rows_padded = rows + 2 * pad_size
    cols_padded = cols + 2 * pad_size
    depth_padded = depth + 2 * pad_size
    P = np.zeros(shape=(rows_padded, cols_padded, depth_padded))

    # copying the interior
    P[pad_size: -pad_size, pad_size: -pad_size, pad_size: -pad_size] = M[:, :, :]

    # copying the exterior values
    if opts_boundary_detailed_wall is not None:

        if opts_boundary_detailed_wall.up_wall:
            P[0:pad_size, :, :] = pad_value
        if opts_boundary_detailed_wall.down_wall:
            P[-pad_size:, :, :] = pad_value
        if opts_boundary_detailed_wall.left_wall:
            P[:, 0:pad_size, :] = pad_value
        if opts_boundary_detailed_wall.right_wall:
            P[:, -pad_size:, :] = pad_value
        if opts_boundary_detailed_wall.front_wall:
            P[:, :, 0:pad_size] = pad_value
        if opts_boundary_detailed_wall.back_wall:
            P[:, :, -pad_size:] = pad_value

    else:  # None opts_boundary

        P[0:pad_size, :, :] = pad_value
        P[-pad_size:, :, :] = pad_value
        P[:, 0:pad_size, :] = pad_value
        P[:, -pad_size:, :] = pad_value
        P[:, :, 0:pad_size] = pad_value
        P[:, :, -pad_size:] = pad_value

    return P


def get_kernel_trimmed_size(kernel_size, trunc_factor_percent):
    """ Compute a new kernel size based on a truncation factor % and the original kernel size (assume a square kernel).

    :param int kernel_size: original kernel size
    :param float trunc_factor_percent: truncation factor in [0, 1]
    :return: new kernel size
    """
    half_size, middle_index = get_1d_array_half_size(length=kernel_size, trunc_factor_percent=trunc_factor_percent)
    return gen.get_kernel_size_from_half_size(half_size=half_size)


def get_kernel_effective_size(opts):
    """ Get the actual convolution kernel size, whether it is full or reduced.

    :param OptionsGeneral opts:
    :return: effective kernel size
    """
    kernel_size = gen.get_kernel_size(itr=opts.kernel.itr, zero_init=opts.solver.zero_init)

    # in case of reduction AND filter truncation we have smaller kernel.
    if opts.reduction.reduce and opts.reduction.truncation_method == com.TruncationMode.PERCENTAGE:
        kernel_size = get_kernel_trimmed_size(kernel_size=kernel_size,
                                              trunc_factor_percent=opts.reduction.truncation_value)

    return kernel_size


def trim(M, size):
    """ Trim the input 2d matrix, which will shrink equally on each side and dimension.

    :param ndarray M: input 2d matrix
    :param int size: trimming size
    :return: trimmed matrix
    """
    if size > 0:
        M = M[size: -size, size: -size]

    return M


def extract_from_center_2d(M, sub_shape):
    """ Extract a sub-matrix from the center.

    .. warning::
        Both the input matrix and the extraction shape must indicate a symmetrical shape with \
        odd number for rows and columns.

    :param ndarray M: input 2d matrix
    :param tuple sub_shape: tuple (rows, cols) of the sub matrix
    :return: sub-matrix
    """
    rows, cols = M.shape

    assert rows % 2 != 0, "This only works with symmetrical matrix: must have odd # rows"
    assert cols % 2 != 0, "This only works with symmetrical matrix: must have odd # cols"

    # finding the center of the input matrix
    c_row = int(rows / 2)
    c_col = int(cols / 2)

    # unpack
    sub_rows, sub_cols = sub_shape
    assert sub_rows % 2 != 0, "This only works with symmetrical sub-matrix: must have odd # rows"
    assert sub_cols % 2 != 0, "This only works with symmetrical sub-matrix: must have odd # cols"

    # sub matrix must be at least the same size as the input matrix
    assert rows >= sub_rows and cols >= sub_cols, "sub matrix is too large"

    half_sub_rows = int(sub_rows / 2)
    half_sub_cols = int(sub_cols / 2)
    # including the high end of the range by +1
    return M[
           c_row - half_sub_rows: c_row + half_sub_rows + 1,
           c_col - half_sub_cols: c_col + half_sub_cols + 1
           ]


def extract_from_center_3d(M, sub_shape):
    """ Extract a sub-matrix from the center.

    .. warning::
       Both the input matrix and the extraction shape must indicate a symmetrical shape with \
       odd number for rows and columns.

    :param ndarray M: input 3d matrix
    :param tuple sub_shape: tuple (rows, cols, depth) of the sub matrix
    :return: sub-matrix
    """
    rows, cols, depth = M.shape

    assert rows % 2 != 0, "This only works with symmetrical matrix: must have odd # rows"
    assert cols % 2 != 0, "This only works with symmetrical matrix: must have odd # cols"
    assert depth % 2 != 0, "This only works with symmetrical matrix: must have odd # depths"

    # finding the center of the input matrix
    c_row = int(rows / 2)
    c_col = int(cols / 2)
    c_depth = int(depth / 2)

    # unpack
    sub_rows, sub_cols, sub_depth = sub_shape
    assert sub_rows % 2 != 0, "This only works with symmetrical sub-matrix: must have odd # rows"
    assert sub_cols % 2 != 0, "This only works with symmetrical sub-matrix: must have odd # cols"
    assert sub_depth % 2 != 0, "This only works with symmetrical sub-matrix: must have odd # depth"

    # sub matrix must be at least the same size as the input matrix
    assert rows >= sub_rows and cols >= sub_cols and depth >= sub_depth, "sub matrix is too large"

    half_sub_rows = int(sub_rows / 2)
    half_sub_cols = int(sub_cols / 2)
    half_sub_depth = int(sub_depth / 2)
    # including the high end of the range by +1
    return M[
           c_row - half_sub_rows: c_row + half_sub_rows + 1,
           c_col - half_sub_cols: c_col + half_sub_cols + 1,
           c_depth - half_sub_depth: c_depth + half_sub_depth + 1
           ]


def set_submatrix_zero_2d(M, skip_margin):
    """ Set the sub-matrix to zero.

    The values inside the skip margin will remain intact. If skip_margin is zero, this means no skip margin, which \
    means set everything to zero.

    :param ndarray M: 2d input matrix
    :param int skip_margin: how many elements from each side should keep their values
    :return: modified matrix
    """
    # no skip margin.. set everything to zero
    if skip_margin == 0:
        return np.zeros_like(M)

    # set the inside of the matrix to zero skipping the margin
    M[skip_margin:-skip_margin, skip_margin:-skip_margin] *= 0

    return M


def set_submatrix_zero_3d(M, skip_margin):
    """ Set the sub-matrix to zero.

    The values inside the skip margin will remain intact. If skip_margin is zero, this means no skip margin, which \
    means set everything to zero.

    :param ndarray M: 3d input matrix
    :param int skip_margin: how many elements from each side should keep their values
    :return: modified matrix
    """
    # no skip margin.. set everything to zero
    if skip_margin == 0:
        return np.zeros_like(M)

    # set the inside of the matrix to zero skipping the margin
    M[skip_margin:-skip_margin, skip_margin:-skip_margin, skip_margin:-skip_margin] *= 0

    return M


def set_frame_fixed_value(M, skip_margin, thickness, value, left=True, right=True, top=True, bottom=True):
    """ Set the elements of the outer frame of the 2d matrix to the given value. 
    
    The frame starts from the skip_margin. 
    The values in the skip_margin and inside the matrix (excluding frame) will remain intact.    

    :param ndarray M: input 2d matrix
    :param int skip_margin: offset from sides
    :param int thickness: thickness of the frame in terms of number of elements
    :param float value: the values inside the frame (fixed for all)
    :param bool left: apply the left side of the frame
    :param bool right: apply the right side of the frame
    :param bool top: apply the top side of the frame
    :param bool bottom: apply the bottom side of the frame
    :return: overwritten input matrix with the frame values
    """

    thickness = abs(thickness)
    skip_margin = abs(skip_margin)

    m, n = M.shape
    assert thickness <= m and thickness <= n, "thickness is too large"
    assert skip_margin <= m and skip_margin <= n, "skip_margin is too large"

    # no skip margin... the outmost boundary
    if skip_margin == 0 and thickness != 0:
        if top:
            M[0: thickness, :] = value
        if bottom:
            M[-thickness:, :] = value
        if left:
            M[:, 0: thickness] = value
        if right:
            M[:, -thickness:] = value

        return M

    if top:
        M[skip_margin: skip_margin + thickness, skip_margin:-skip_margin] = value
    if bottom:
        M[-(skip_margin + thickness): -skip_margin, skip_margin:-skip_margin] = value
    if left:
        M[skip_margin:-skip_margin, skip_margin: skip_margin + thickness] = value
    if right:
        M[skip_margin:-skip_margin, -(skip_margin + thickness):-skip_margin] = value

    return M


def set_frame_boundary_from_matrix(M, thickness, source):
    """ Set the boundary values of the given matrix from another matrix.

    The extracted boundary elements can have larger than one-cell thickness.

    :param ndarray M: 2d target matrix to edit
    :param int thickness: thickness of the boundary
    :param ndarray source: 2d source matrix to extract the boundary values from
    :return: modified matrix
    """

    assert M.shape == source.shape, "Inconsistent matrix sizes"

    b_frame = extract_frame_values(M=source, skip_margin=0, thickness=thickness)

    M[0: thickness, :] = b_frame[0: thickness, :]
    M[-thickness:, :] = b_frame[-thickness:, :]
    M[:, 0: thickness] = b_frame[:, 0: thickness]
    M[:, -thickness:] = b_frame[:, -thickness:]

    return M


def extract_frame_values(M, skip_margin, thickness):
    """ Extract the elements of the outer frame of the 2d matrix.

    The rest of the matrix will be zero. The frame starts from the skip_margin.

    :param ndarray M: 2d input matrix to extract from
    :param int skip_margin: offset from sides
    :param int thickness: thickness of the frame in terms of number of elements
    :return: new 2d matrix with the extracted frame values.
    """
    thickness = abs(thickness)
    skip_margin = abs(skip_margin)

    m, n = M.shape
    assert thickness <= m and thickness <= n, "thickness is too large"
    assert skip_margin <= m and skip_margin <= n, "skip_margin is too large"

    E = np.zeros_like(M)

    # no skip margin... the outmost boundary
    if skip_margin == 0 and thickness != 0:
        E[0: thickness, :] = M[0: thickness, :]
        E[-thickness:, :] = M[-thickness:, :]
        E[:, 0: thickness] = M[:, 0: thickness]
        E[:, -thickness:] = M[:, -thickness:]

        return E

    E[skip_margin: skip_margin + thickness, skip_margin:-skip_margin] = \
        M[skip_margin: skip_margin + thickness, skip_margin:-skip_margin]
    E[-(skip_margin + thickness): -skip_margin, skip_margin:-skip_margin] = \
        M[-(skip_margin + thickness): -skip_margin, skip_margin:-skip_margin]
    E[skip_margin:-skip_margin, skip_margin: skip_margin + thickness] = \
        M[skip_margin:-skip_margin, skip_margin: skip_margin + thickness]
    E[skip_margin:-skip_margin, -(skip_margin + thickness):-skip_margin] = \
        M[skip_margin:-skip_margin, -(skip_margin + thickness):-skip_margin]

    return E


def truncate_filters(truncation_method, truncation_value, safe_rank, filters_1d, preserve_shape=True):
    """ Truncate filters either by `PERCENTAGE` or `FIXED_THRESHOLD` (*adaptive truncation*).

    Poisson filters are symmetrical, hence truncation automatically implies symmetrically removing values from sides.

    :param TruncationMode truncation_method: :code:`PERCENTAGE` or :code:`FIXED_THRESHOLD` (*adaptive truncation*)
    :param float truncation_value:  if the truncation method is :code:`PERCENTAGE` then a value in [0, 1], else \
        fixed floating point cut off threshold (:code:`FIXED_THRESHOLD`)
    :param int safe_rank: desired input rank. 'safe' means a rank that does not exceed the actual rank of the kernel
    :param ndarray filters_1d:
    :param bool preserve_shape: if True keep the original shape and fill them with zeros (Default= :code:`True`)
    :return: a new smaller array with truncated elements
    """

    if truncation_method == com.TruncationMode.PERCENTAGE:
        filters_1d = truncate_percent_filter_1d(arr=filters_1d, trunc_percent=truncation_value)

    elif truncation_method == com.TruncationMode.FIXED_THRESHOLD:
        filters_1d = apply_adaptive_truncation_1d(array_1d=filters_1d, safe_rank=safe_rank,
                                                  cut_off=truncation_value, preserve_shape=preserve_shape)
    else:
        assert False, "Unknown truncation method"

    return filters_1d


def truncate_percent_filter_1d(arr, trunc_percent):
    """ Truncate a percentage of a 1d array symmetrically.

    It finds the middle, then throws out the truncation % from both ends. The results will be exactly symmetrical for
    odd sized arrays. For arrays with an even size the middle will be the ceiling (the larger index of the pair in the
    middle). For a 100% truncation the output is the middle element.

    If truncation results in fractional elimination of an element, we assume ceiling and still eliminate that element.

    :param ndarray arr: input 1d array
    :param float trunc_percent: truncation factor in [0, 1]
    :return: a new smaller array with truncated elements
    """
    assert 0.0 <= trunc_percent <= 1.0, "Invalid truncation. Must be in [0.0, 1.0]"

    rows, cols = arr.shape
    half_size, f_index_middle = get_1d_array_half_size(length=cols, trunc_factor_percent=trunc_percent)

    return np.copy(arr)[:, f_index_middle - half_size: f_index_middle + half_size + 1]


def apply_adaptive_truncation_1d(array_1d, safe_rank, cut_off, preserve_shape):
    """ Adaptive truncate all ranked filters using a fixed threshold as the cut-off value, assuming symmetrical \
    filters and values are sorted sideways from largest (center) to smallest (tales).

    :param ndarray array_1d: input filter
    :param int safe_rank: desired input rank. 'safe' means a rank that does not exceed the actual rank of the kernel
    :param float cut_off: truncation threshold (absolute value)
    :param bool preserve_shape: if True keep the original shape and fill them with zeros
    :return: truncated (smaller) array. If preserving the shape, keep the shape and insert zeros in the truncated parts.
    """
    for i in range(safe_rank):
        # do it for every rank separately
        array_1d[i] = truncate_fixed_threshold_1darray(arr=array_1d[i], cut_off=cut_off, preserve_shape=preserve_shape)

    return array_1d


def truncate_fixed_threshold_1darray(arr, cut_off, preserve_shape=True):
    """ Adaptive truncation. Cut any elements smaller than a fixed absolute value, assuming a symmetrical filter \
    and values are sorted sideways from largest (center) to smallest (tales).

    Only works with symmetrical filters, and only cuts the outer parts of the filters.

    [....outer part.... cut || .......inner part....... || cut ....outer part....]

    Cut elements < cut_off

    Keep elements >= cut_off

    Checking the absolute values against the absolute threshold value.

    :param ndarray arr: input 1d array
    :param float cut_off: truncation threshold (absolute value)
    :param bool preserve_shape: if True keep the original shape and fill them with zeros (Default= :code:`True`)
    :return: truncated (smaller) array. If preserving the shape, keep the shape and insert zeros in the truncated parts.
    """
    # make sure threshold is positive
    cut_off = abs(cut_off)

    # there is nothing smaller than abs(0) so return the original
    if cut_off == 0:
        return arr

    # if all array elements are smaller than threshold return zero or empty array
    if np.max(np.abs(arr)) < cut_off:
        return np.zeros_like(arr) if preserve_shape else np.array([])

    cut, cut_indices = get_truncate_indices(arr=arr, cut_off=cut_off)
    assert cut is not None and cut_indices is not None, 'Every thing is truncated. This case must have been caught ' \
                                                        'before this line.'

    # if there is nothing to truncate return the array
    if cut == 0:
        return arr

    if preserve_shape:
        # preserve shape by just zeroing out the left and right truncated parts.
        new_arr = np.copy(arr)
        new_arr[0:cut] = 0
        new_arr[-cut:] = 0
    else:
        # return only the non-truncated part (does not necessarily have the same shape as input)
        new_arr = arr[cut:-cut]

    # make sure the minimum nonzero is equal to or greater than the cut-off threshold
    # assert np.min(np.abs(new_arr[np.where(np.abs(new_arr) > 0)])) >= cut_off

    return new_arr


def get_truncate_indices(arr, cut_off):
    """ Find indices of array values larger than or equal to the cut_off value.

    We get the indices that *should be kept*. For a symmetrical array with larger values around the center,
    this means an index list of the sub-array spanning a range around the center.

    :param ndarray arr: input 1darray
    :param float cut_off: truncation threshold.
    :return:
            - **cut** - the exact truncation index, cut_indices: all indices that should be kept
            - **cut_indices** - the index list of all cut values

    .. warning::
        Returns :code:`None` for both if the whole array is subject to truncation (nothing will be left).
    """

    # if all array elements are smaller than threshold return zero or empty array
    if np.max(np.abs(arr)) < cut_off:
        return None, None

    # find elements larger than or equal to the threshold
    indices = np.where(np.abs(arr) >= cut_off)
    indices = indices[0]  # for an array of shape (n, 1) we are only interested the indices for first dim
    #  find the leftmost element that is >= threshold.
    #  assuming the 1d array is symmetrical, cut equally from both sides
    cut_indices = np.asarray(indices, dtype=np.int64).flatten()
    # the exact cut is the first index. We could likewise use the last index for a symmetrical array.
    cut = cut_indices[0]

    return cut, cut_indices


def find_max_rank_and_filter_size(ranked_filters, safe_rank):
    """ Return maximum rank and filter size required based on excluding the zero-out elements
    of their values set during adaptive truncation.

    .. note::
            Assuming filters are already adaptively truncated, meaning their small values are
            set to zero given a fixed truncation threshold. All filters have the same size, preserving
            their original size.

    :param ndarray ranked_filters: 2d array of the shape (ranks, filters)
    :param int safe_rank: desired input rank. 'safe' means a rank that does not exceed the actual rank of the kernel.
    :return:
        - **max_effective_rank** : maximum rank required
        - **max_filter_size**: maximum filter size required (max of all ranked filters)
        - **actual_filter_size**: filter size if there was no truncation
    """

    max_filter_size = 0
    max_effective_rank = 0
    for i in range(safe_rank):
        effective_size = len(get_effective_nonzero_1darray(arr=ranked_filters[i]))
        # if
        if effective_size == 0:
            continue

        else:  # effective_size > 0 -- large enough to have it in computation
            # involve current rank
            max_effective_rank += 1
            # keep score of the maximum filter size we need in convolution
            if max_filter_size < effective_size:
                max_filter_size = effective_size

    actual_filter_size = len(ranked_filters[0])
    return max_effective_rank, max_filter_size, actual_filter_size


def get_effective_nonzero_1darray(arr):
    """ Find where non-zero elements are.

    :param arr: input array
    :return: array with only nonzero elements
    """
    return arr[np.where(np.abs(arr) > 0)]


def construct_laplacian_nd_vector_friendly(dim, base_size, positive_definite=False, neumann=False, singularity_value=1):
    """ Construct the Laplacian matrix for the *vector from* of :math:`Ax=b` where :math:`A` is the Laplacian.

    .. note::
        The system size in :math:`Ax=b` is the size of the vectors :math:`x` and :math:`b` in the *vector form*, and \
        can be achieved from the size of the Laplacian matrix:

        - *2D* : :math:`\\sqrt{dim(A)}`
        - *3D* : :math:`\\sqrt[3]{dim(A)}`

    :param SolverDimension dim: D2 or D3
    :param int base_size: for a square matrix, number of rows or columns (or depth in 3D).
        This will be used to compute the Laplacian size.
        e.g. base_size = 5 : 2D Laplacian :math:`5^2`, 3D Laplacian :math:`5^3`
    :param bool positive_definite: just a flag to flip the sign of the matrix to have positive values on
        the diagonals. Default is negative diagonals (:code:`False`), to make it consistent with the Poisson kernel generator.
        In our case where we are just interested in convergence properties this sign flip does not change anything
    :param bool neumann: with Neumann boundary condition (Default= :code:`False`)
    :param float singularity_value: regularization factor for treating the ill-conditioned matrix (Default= :code:`1`)
    :return: square/cubic Laplacian matrix/tensor in 2D/3D
    """
    assert com.is_solver_2d(dim) or com.is_solver_3d(dim), "Unknown solver dimension"

    # Building matrix
    nx = base_size
    x_size = nx ** (2 if com.is_solver_2d(dim) else 3)

    A = np.zeros((x_size, x_size), dtype=np.double)  # Laplacian initialization

    if com.is_solver_2d(dim):  # 2D Laplacian
        neighbor = -1
        center = 4

        for i in range(nx):
            for j in range(nx):
                if i - 1 >= 0:
                    A[(i - 1) * nx + j, i * nx + j] = neighbor
                    A[i * nx + j, (i - 1) * nx + j] = neighbor
                if i + 1 <= nx - 1:
                    A[(i + 1) * nx + j, i * nx + j] = neighbor
                    A[i * nx + j, (i + 1) * nx + j] = neighbor
                if j - 1 >= 0:
                    A[i * nx + (j - 1), i * nx + j] = neighbor
                    A[i * nx + j, i * nx + (j - 1)] = neighbor
                if j + 1 <= nx - 1:
                    A[i * nx + (j + 1), i * nx + j] = neighbor
                    A[i * nx + j, i * nx + (j + 1)] = neighbor
                A[i * nx + j, i * nx + j] = center
        if neumann:
            for i in range(nx):
                for j in range(nx):
                    # border: -1
                    if i == 0:
                        A[i * nx + j, i * nx + j] -= 1
                    if i == nx - 1:
                        A[i * nx + j, i * nx + j] -= 1
                    if j == 0:
                        A[i * nx + j, i * nx + j] -= 1
                    if j == nx - 1:
                        A[i * nx + j, i * nx + j] -= 1

            # for the ill conditioned matrix
            A[x_size - 1, x_size - 1] += singularity_value

    elif com.is_solver_3d(dim):  # 3D Laplacian
        neighbor = -1
        center = 6

        for i in range(nx):
            for j in range(nx):
                for k in range(nx):
                    if i - 1 >= 0:
                        A[(i - 1) * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + k] = neighbor
                        A[i * nx ** 2 + j * nx + k, (i - 1) * nx ** 2 + j * nx + k] = neighbor
                    if i + 1 <= nx - 1:
                        A[(i + 1) * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + k] = neighbor
                        A[i * nx ** 2 + j * nx + k, (i + 1) * nx ** 2 + j * nx + k] = neighbor
                    if j - 1 >= 0:
                        A[i * nx ** 2 + (j - 1) * nx + k, i * nx ** 2 + j * nx + k] = neighbor
                        A[i * nx ** 2 + j * nx + k, i * nx ** 2 + (j - 1) * nx + k] = neighbor
                    if j + 1 <= nx - 1:
                        A[i * nx ** 2 + (j + 1) * nx + k, i * nx ** 2 + j * nx + k] = neighbor
                        A[i * nx ** 2 + j * nx + k, i * nx ** 2 + (j + 1) * nx + k] = neighbor
                    if k - 1 >= 0:
                        A[i * nx ** 2 + j * nx + (k - 1), i * nx ** 2 + j * nx + k] = neighbor
                        A[i * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + (k - 1)] = neighbor
                    if k + 1 <= nx - 1:
                        A[i * nx ** 2 + j * nx + (k + 1), i * nx ** 2 + j * nx + k] = neighbor
                        A[i * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + (k + 1)] = neighbor
                    A[i * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + k] = center
        if neumann:
            for i in range(nx):
                for j in range(nx):
                    for k in range(nx):
                        # border: -1
                        if i == 0:
                            A[i * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + k] -= 1
                        if i == nx - 1:
                            A[i * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + k] -= 1
                        if j == 0:
                            A[i * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + k] -= 1
                        if j == nx - 1:
                            A[i * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + k] -= 1
                        if k == 0:
                            A[i * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + k] -= 1
                        if k == nx - 1:
                            A[i * nx ** 2 + j * nx + k, i * nx ** 2 + j * nx + k] -= 1

            # for the ill conditioned matrix
            A[x_size - 1, x_size - 1] += singularity_value

    # sanity check
    row_sum = np.zeros_like(A[0, :])
    for i in range(x_size):
        row_sum += A[i, :]
    ideal_row_sum = np.zeros_like(A[0, :])
    ideal_row_sum[-1] = singularity_value

    # assert(np.all(np.isclose(row_sum, ideal_row_sum))) # only last entry should be 1

    if not positive_definite:
        A *= -1

    return A, x_size


def apply_laplacian_2d(X):
    """ Convolve a 2d domain with a 2d Laplacian operator in the *matrix from*. This is based on a 3x3 base \
    Laplacian kernel when finite differencing (the same as the Laplacian base kernel in Jacobi).

    .. note::
        *matrix from* of :math:`Ax=b` is :math:`L*X=B`

        where
            - :math:`L`: Laplacian operator
            - :math:`*`: convolution operator
            - :math:`X`: given or unknown matrix, depending on the Poisson equation we are solving for.
            - :math:`B`: given or unknown rhs matrix, depending on the Poisson equation we are solving for.

        All matrices have the same dimension in the *matrix form*.

    .. warning::
        Each dimension must be at least 3 to allow for fetching the marginal wall cells.

    :param ndarray X: input 2d matrix
    :return: :math:`L*X`
    """
    rows, cols = X.shape
    assert rows >= 3, "Each dimension must be at least 3 to allow for fetching the marginal cells."
    assert cols >= 3, "Each dimension must be at least 3 to allow for fetching the marginal cells."

    center = X[1:-1, 1:-1]
    up = X[0:-2, 1:-1]
    down = X[2:, 1:-1]
    left = X[1:-1, 0:-2]
    right = X[1:-1, 2:]

    lap = up + down + left + right - 4 * center

    return lap


def apply_laplacian_3d(X):
    """ Convolve a 3d domain with a 3d Laplacian operator in the *matrix from*. This is based on a 3x3x3 base \
    Laplacian kernel when finite differencing (the same as the Laplacian base kernel in Jacobi).

    .. note::
        *matrix from* of :math:`Ax=b` is :math:`L*X=B`

        where
            - :math:`L`: Laplacian operator
            - :math:`*`: convolution operator
            - :math:`X`: given or unknown tensor, depending on the Poisson equation we are solving for.
            - :math:`B`: given or unknown rhs tensor, depending on the Poisson equation we are solving for.

        All tensors have the same dimension in the *matrix form*.

    .. warning::
        Each dimension must be at least 3 to allow for fetching the marginal wall cells.

    :param ndarray X: input 3d tensor
    :return: :math:`L*X`
    """
    rows, cols, fibers = X.shape
    # Each dimension must be at least 3 to allow for fetching the marginal cells.
    assert rows >= 3, "Each dimension must be at least 3 to allow for fetching the marginal cells."
    assert cols >= 3, "Each dimension must be at least 3 to allow for fetching the marginal cells."
    assert fibers >= 3, "Each dimension must be at least 3 to allow for fetching the marginal cells."

    center = X[1:-1, 1:-1, 1:-1]
    up = X[0:-2, 1:-1, 1:-1]
    down = X[2:, 1:-1, 1:-1]
    left = X[1:-1, 0:-2, 1:-1]
    right = X[1:-1, 2:, 1:-1]
    back = X[1:-1, 1:-1, 0:-2]
    front = X[1:-1, 1:-1, 2:]

    lap = up + down + left + right + back + front - 6 * center

    return lap


def solve_jacobi_vector_form(A, b, max_iterations, is_subdomain_residual, sub_domain_shape, dim, warm_start=False):
    """ Solving :math:`Ax=b` in *vector form*.

    Works for both 2D and 3D as long as proper :math:`A` and :math:`b` is fed in the vector form.

    :param ndarray A: n x n (flat Laplacian)
    :param ndarray b: n x 1 (rhs)
    :param int max_iterations: stop when exceeding max iteration
    :param bool is_subdomain_residual: if computing the residual only for a subdomain.
    :param tuple sub_domain_shape: the shape of the subdomain if doing subdomain residual. \
        Must be odd size for proper matrix extraction.
    :param SolverDimension dim: 2D or 3D, dimension of the problem
    :param warm_start: this is loosely based on just copying :math:`b`, currently often makes the convergence \
        worse (Default= :code:`False`)
    :return:
        - **x** - solution
        - **residuals** - residual per iteration
    """

    if warm_start:
        x = np.copy(b).astype(np.double)
    else:
        x = np.zeros_like(b, dtype=np.double)

    residuals = np.zeros(max_iterations + 1, dtype=np.double)  # +1 to account for the zeroth residual

    def add_residual(nth_res):
        if is_subdomain_residual:

            # this uses a no boundary Laplacian, excluding the incomplete kernel for the
            # boundaries; more consistent with pure Poisson filters with no boundaries
            if com.is_solver_2d(dim):
                rows = cols = int(np.sqrt(b.reshape(-1, 1).shape[0]))
                x_mat_form = x.reshape(rows, cols)
                b_mat_form = b.reshape(rows, cols)

            elif com.is_solver_3d(dim):
                rows = cols = depth = int(np.cbrt(b.reshape(-1, 1).shape[0]))
                x_mat_form = x.reshape((rows, cols, depth))
                b_mat_form = b.reshape((rows, cols, depth))
            else:
                assert False, "Unknown Solver Dimension"

            residuals[nth_res] = compute_residual_subdomain(X=x_mat_form,
                                                            B=b_mat_form,
                                                            sub_shape=sub_domain_shape,
                                                            solver_dimension=dim)

        else:
            # this works for both 2D and 3D

            # standard method: this includes full Laplacian with boundary incomplete kernels. Would
            # give different residual than the residual obtained by pure Poisson filter convolution
            residuals[nth_res] = np.linalg.norm((b - A @ x).reshape(b.shape[0]))

    add_residual(nth_res=0)

    for k in range(1, max_iterations + 1):
        # print(f'Jacobi vector form itr {k}')

        x_old = x.copy()

        # Loop over rows
        for i in range(A.shape[0]):
            x[i, 0] = (b[i, 0] - A[i, :i] @ x_old[:i, 0] - A[i, (i + 1):] @ x_old[(i + 1):, 0]) / A[i, i]

        add_residual(nth_res=k)

    return x, residuals.reshape(-1, 1)


# This is useful for infinite domain solve with no boundary condition
def solve_jacobi_matrix_form_no_boundary_2d(M, opts, do_residuals=False,
                                            is_subdomain_residual=True, subdomain_shape=None):
    """Solve the Poisson equation with Jacobi in the *matrix form* for *forward* and *inverse* Poisson equations in \
    2D with no boundary treatment.

    This version of Jacobi solver perfectly matches the results of Poisson filters when boundary treatment is ignored, \
    in other words, for infinite domains.

    .. note::
        We use a general Jacobi setup with flexible :math:`\\alpha` and :math:`\\beta` instead of fixed
        values. This allows to adjust the weights based on the type of the Poisson equation.

    .. warning::
        The computed residual is in the *matrix form* and is only valid for *inverse* Poisson setup.

    .. note::
        We need one extra cell trimming from sides because of using pure Poisson filters
        in comparison. We could have easily added the incomplete Laplacian kernels for the corners
        and the cells next to the wall, but this will make it inconsistent with the Jacobi function
        used to generate the Poisson kernels.

    :param ndarray M: input 2d matrix; if *inverse* setup, :math:`M=B` in :math:`L*X=B`, and if *forward* setup, \
         :math:`M=X` in in :math:`L*X=B`, with :math:`B` being the unknown
    :param OptionsGeneral opts: general options
    :param bool do_residuals: collect residuals at each iteration (Default= :code:`False`)
    :param bool is_subdomain_residual: compute the residual only for a subdomain (Default= :code:`True`)
    :param tuple subdomain_shape: subdomain size to be used in computing the residuals (Default= :code:`None`)
    :return:
        - **Out** - solution
        - **residuals** - residual per iteration
    """

    # initialize the solution
    Out = np.zeros_like(M).astype(np.double)  # acts as warm start

    residuals = None
    if do_residuals:
        residuals = np.zeros(opts.kernel.itr + 1, dtype=np.double)  # +1 to account for the zeroth residual

    alpha = np.double(gen.compute_alpha(
        dx=opts.kernel.dx, dt=opts.kernel.dt, kappa=opts.kernel.kappa, solver_type=opts.solver.solver_type))
    beta = np.double(gen.compute_beta(
        alpha=opts.kernel.alpha, solver_type=opts.solver.solver_type, dim=opts.solver.dim))

    alpha *= -1. if opts.solver.solver_type == com.PoissonSolverType.INVERSE else 1.

    # NOTE: The computed residual is in the *matrix form* and is only valid for *inverse* Poisson setup.
    def add_residual(k):
        # this works for both 2D and 3D

        # standard method: this includes full Laplacian with boundary incomplete kernels. Would
        # give different residual than the residual obtained by pure Poisson filter convolution

        if is_subdomain_residual:
            assert subdomain_shape is not None
            residuals[k] = compute_residual_subdomain(X=Out,
                                                      B=M,
                                                      sub_shape=subdomain_shape,
                                                      solver_dimension=com.SolverDimension.D2)
        else:
            residuals[k] = compute_residual_poisson_operator(X=Out, B=M, solver_dimension=com.SolverDimension.D2)

    m, n = M.shape

    # compute initial residual
    if do_residuals:
        add_residual(0)

    for ii in range(opts.kernel.itr):
        print(f'solving Jacobi matrix form itr {ii + 1}..')
        o_last = np.copy(Out)  # update overall collective values

        for cx in range(1, m - 1):
            for cy in range(1, n - 1):
                o_l = o_last[cx - 1, cy]  # up
                o_r = o_last[cx + 1, cy]  # down
                o_u = o_last[cx, cy + 1]  # right
                o_d = o_last[cx, cy - 1]  # left
                input_c = M[cx, cy]
                # update
                Out[cx, cy] = (o_l + o_r + o_u + o_d + alpha * input_c) / beta

        if do_residuals:
            add_residual(ii + 1)

    return Out, residuals


def solve_jacobi_single_padding_only_wall_2d(M, sub_shape_residual, opts):
    """Solve the Poisson equation with Jacobi in the *matrix form* for *forward* and *inverse* Poisson equations in \
    2D with wall Neumann boundary treatment.

    .. note::
        We use a general Jacobi setup with flexible :math:`\\alpha` and :math:`\\beta` instead of fixed
        values. This allows to adjust the weights based on the type of the Poisson equation.

    .. note::
        This function only supports Neumann boundary treatment on the cell edges. The walls are single cell padding.

    :param ndarray M: input 2d matrix; if *inverse* setup, :math:`M=B` in :math:`L*X=B`, and if *forward* setup, \
         :math:`M=X` in in :math:`L*X=B`, with :math:`B` being the unknown
    :param OptionsGeneral opts: general options (contains number of iterations along with many other variables)
    :param 2-tuple sub_shape_residual: subdomain shape used in computing the residual
    :return:
        - **Out** - solution
        - **residuals** - residual per iteration
    """

    assert com.is_solver_2d(opts.solver.dim)

    residuals = np.zeros(opts.kernel.itr + 1, dtype=np.double)  # +1 to account for the zeroth residual

    def add_residual(k):
        residuals[k] = compute_residual_subdomain_2d(X=Out,
                                                     B=M,
                                                     sub_shape=sub_shape_residual)

    # initialize the solution
    Out = np.copy(M)  # acts as warm start
    if opts.solver.zero_init:
        set_submatrix_zero_2d(M=Out, skip_margin=1)  # keeping the single padding boundary intact

    sign = gen.get_alpha_sign(solver_type=opts.solver.solver_type, kernel_type=opts.kernel.kernel_type)

    rows, cols = M.shape

    add_residual(0)

    for ii in range(1, opts.kernel.itr + 1):
        print(f'solving Jacobi matrix form itr {ii}..')
        o_last = np.copy(Out)  # update overall collective values

        # wall boundary
        if opts.boundary.enforce:
            o_last = set_wall_bound_2d(M=o_last, bound_type=opts.boundary.condition)

        # rows: up & down, columns: left & right"
        for cy in range(1, rows - 1):
            for cx in range(1, cols - 1):
                # object boundary
                if opts.boundary.enforce and opts.boundary.obj_collide:
                    # Same solid cell might acquire different values based on which current cell
                    # we are updating for. This is to ensure we get the same values for the neighbour
                    # and the central cell in computing the central cell update.
                    # because of this potentially changing solid value, we need to keep the
                    # boundary enforcement step inside this nested loop... can't really do it outside as a separate
                    # step. It would cause confusion and race condition for the solid cells.
                    o_last = set_single_cell_wall_bound_2d(M=o_last,
                                                           here_index=com.Vector2DInt(v1=cy, v2=cx),
                                                           bound_type=opts.boundary.condition)

                    if is_wall_2d(M=M, i=cy, j=cx):
                        continue

                o_u = o_last[cy - 1, cx]  # up
                o_d = o_last[cy + 1, cx]  # down
                o_r = o_last[cy, cx + 1]  # right
                o_l = o_last[cy, cx - 1]  # left

                input_c = M[cy, cx]
                # update
                Out[cy, cx] = (o_l + o_r + o_u + o_d + sign * opts.kernel.alpha * input_c) / opts.kernel.beta

        add_residual(ii)

    # post-processing [Optional]
    # final BC enforcement to make sure pressure values give the right edge gradient
    if opts.boundary.enforce and opts.boundary.post_solve_enforcement:
        post_process_wall_boundary_enforcement_2d(M=Out, opts=opts)

    return Out, residuals


def post_process_wall_boundary_enforcement_2d(M, opts):
    """Boundary enforcement to make sure domain values give the right edge gradient (Neumann). Usually used as an \
    optional last step clean up after the Poisson equation solve. Walls only.

    :param ndarray M: input 2d matrix
    :param OptionsGeneral opts: general options (contains boundary enforcement type, but only supports Neumann for now)
    :return: **M** treated domain
    """
    rows, cols = M.shape

    M = set_wall_bound_2d(M=M, bound_type=opts.boundary.condition)

    # rows: up & down, columns: left & right, depth: front and back
    for cy in range(1, rows - 1):
        for cx in range(1, cols - 1):
            # wall  boundary
            if opts.boundary.obj_collide:
                M = set_single_cell_wall_bound_2d(M=M,
                                                  here_index=com.Vector2DInt(v1=cy, v2=cx),
                                                  bound_type=opts.boundary.condition)

    return M


def solve_jacobi_single_padding_obj_collision_2d(M, opts, collision_mask=None):
    """Solve the Poisson equation with Jacobi in the *matrix form* for *forward* and *inverse* Poisson equations in \
    2D with wall and in-domain boundary treatment of solid objects with complex shapes.

    .. note::
        We use a general Jacobi setup with flexible :math:`\\alpha` and :math:`\\beta` instead of fixed
        values. This allows to adjust the weights based on the type of the Poisson equation.

    .. note::
        This function only supports Neumann boundary treatment on the cell edges. The walls are single cell padding.

    :param ndarray M: input 2d matrix; if *inverse* setup, :math:`M=B` in :math:`L*X=B`, and if *forward* setup, \
         :math:`M=X` in in :math:`L*X=B`, with :math:`B` being the unknown
    :param OptionsGeneral opts: general options (contains number of iterations along with many other variables)
    :param collision_mask: 2d object solid mask as in-domain collider
    :return: **Out** - solution
    """

    # initialize the solution
    Out = np.copy(M)  # acts as warm start
    if opts.solver.zero_init:
        set_submatrix_zero_2d(M=Out, skip_margin=1)  # keeping the single padding boundary intact

    sign = gen.get_alpha_sign(solver_type=opts.solver.solver_type, kernel_type=opts.kernel.kernel_type)

    rows, cols = M.shape

    for ii in range(opts.kernel.itr):
        print(f'solving Jacobi matrix form itr {ii + 1}..')
        o_last = np.copy(Out)  # update overall collective values

        # wall boundary
        if opts.boundary.enforce:
            o_last = set_wall_bound_2d(M=o_last, bound_type=opts.boundary.condition)

        # rows: up & down, columns: left & right"
        for cy in range(1, rows - 1):
            for cx in range(1, cols - 1):
                # object boundary
                if opts.boundary.enforce and opts.boundary.obj_collide and collision_mask is not None:
                    # Same solid cell might acquire different values based on which current cell
                    # we are updating for. This is to ensure we get the same values for the neighbour
                    # and the central cell in computing the central cell update.
                    # because of this potentially changing solid value, we need to keep the
                    # boundary enforcement step inside this nested loop... cant really do out as a separate
                    # step. It would cause confusion and race condition for the solid cells.
                    o_last = set_obj_bound_2d(M=o_last,
                                              here_index=com.Vector2DInt(v1=cy, v2=cx),
                                              collision_mask=collision_mask,
                                              bound_type=opts.boundary.condition)

                    if is_solid_2d(mask=collision_mask, i=cy, j=cx):
                        continue

                o_u = o_last[cy - 1, cx]  # up
                o_d = o_last[cy + 1, cx]  # down
                o_r = o_last[cy, cx + 1]  # right
                o_l = o_last[cy, cx - 1]  # left

                input_c = M[cy, cx]
                # update
                Out[cy, cx] = (o_l + o_r + o_u + o_d + sign * opts.kernel.alpha * input_c) / opts.kernel.beta

    # post-processing [Optional]
    # final BC enforcement to make sure pressure values give the right edge gradient
    if opts.boundary.enforce and opts.boundary.post_solve_enforcement:
        post_process_obj_boundary_enforcement_2d(M=Out, collision_mask=collision_mask, opts=opts)

    return Out


def post_process_obj_boundary_enforcement_2d(M, collision_mask, opts):
    """ Boundary enforcement to make sure domain values give the right edge gradient (Neumann). Usually used as an \
    optional last step clean up after the Poisson equation solve.

    .. note::
        *Complex Object Boundary*

        This is not perfect. For corner cells on the object there is a race condition as
        what value we should use to copy from. This is because there are more than one
        cell in the active domain to copy from. A remedy would be to take average.
        Nonetheless, because we don't really compute the gradient, and we are just interested
        in the active domain values, we skip improving this step.

        This is for the collocated grid.

        For MACGrid we do not really have to worry about the interior cells inside the
        object when solving, for instance, for the pressure. The pressure gradient on the
        edge cells is already ensured to be zero during the Jacobi iterations, so we do not
        have to explicitly set the object cell values.

    :param ndarray M: input 2d matrix
    :param ndarray collision_mask: 2d object solid mask as in-domain collider
    :param OptionsGeneral opts: general options (contains boundary enforcement type, but only supports Neumann for now)
    :return: **M** treated domain
    """
    rows, cols = M.shape

    M = set_wall_bound_2d(M=M, bound_type=opts.boundary.condition)

    # rows: up & down, columns: left & right"
    for cy in range(1, rows - 1):
        for cx in range(1, cols - 1):
            if opts.boundary.obj_collide and collision_mask is not None:
                M = set_obj_bound_2d(M=M,
                                     here_index=com.Vector2DInt(v1=cy, v2=cx),
                                     collision_mask=collision_mask,
                                     bound_type=opts.boundary.condition)

    return M


def set_wall_bound_2d(M, bound_type):
    """ Boundary condition enforcement on the domain walls.

    :param ndarray M: input 2d matrix
    :param BoundaryType bound_type: boundary enforcement type. Currently only supports Neumann.
    :return: **M** treated domain walls
    """
    if bound_type == com.BoundaryType.DIRICHLET:
        assert False, "Not implemented yet.."

    elif bound_type == com.BoundaryType.NEUMANN_EDGE:
        return set_wall_bound_neumann_edge_2d(M=M)

    elif bound_type == com.BoundaryType.NEUMANN_CENTER:
        return set_wall_bound_neumann_center_2d(M=M)

    elif bound_type == com.BoundaryType.CONST_FRAME:
        return M  # no action; assuming the caller function already reads from the single padding but does not modify it

    else:
        assert False, "Unknown boundary type"


def set_wall_bound_neumann_edge_2d(M):
    """Enforce Pure Neumann boundary condition on the wall cell edges.

    Set the wall values from the adjacent interior domain cell. Zero gradient on the edge, i.e. the boundary is \
    defined on the edge separating the interior and the wall cells.

    e.g. when left wall is between :math:`x_0` and :math:`x_1`, then we impose :math:`x_0 = x_1`, where \
    :math:`x_0` is inside the wall and :math:`x_1` is inside the domain.

    .. note::
        We do not care much about what values corner cells are getting because in the
        fluids context only the gradient between the wall and the fluid (the cell edges)
        matters. The gradient between solid wall cells should not affect the projection
        step as it does not make sense to correct the fluid velocity inside the wall.

    :param ndarray M: input 2d matrix
    :return: **M** treated domain walls
    """
    M[0, :] = M[1, :]
    M[-1:, :] = M[-2, :]
    M[:, 0] = M[:, 1]
    M[:, -1] = M[:, -2]

    return M


def set_wall_bound_neumann_center_2d(M):
    """Enforce Pure Neumann boundary condition on the cell centers.

    Set the wall values from the adjacent interior domain cell. Zero gradient in the center of the interior cell.
    To achieve zero gradient in the center of first interior cell we need to equate the wall cell and the second
    interior cell.

    e.g. when left wall is :math:`x_1`, then we impose :math:`x_0 = x_2`, where :math:`x_1` is the actual boundary \
    and :math:`x_0` is inside the wall.

    .. note::
        We do not care much about what values corner cells are getting because in the
        fluids context only the gradient between the wall and the fluid (the cell edges)
        matters. The gradient between solid wall cells should not affect the projection
        step as it does not make sense to correct the fluid velocity inside the wall.

    :param ndarray M: input 2d matrix
    :return: **M** treated domain walls
    """
    M[0, :] = M[2, :]
    M[-1:, :] = M[-3, :]
    M[:, 0] = M[:, 2]
    M[:, -1] = M[:, -3]

    return M


def set_obj_bound_2d(M, here_index, collision_mask, bound_type):
    """ Boundary condition enforcement on a specific cell of the solid object inside the domain.

    :param ndarray M: input 2d matrix
    :param Vector2DInt here_index: cell index
    :param ndarray collision_mask: 2d object solid mask as in-domain collider
    :param BoundaryType bound_type: boundary enforcement type. Currently only supports Neumann on cell edge
    :return: **M** treated domain
    """
    if bound_type == com.BoundaryType.DIRICHLET:
        assert False, "Not implemented yet..."

    elif bound_type == com.BoundaryType.NEUMANN_EDGE:
        return set_obj_bound_neumann_edge_2d(M=M, here_index=here_index, collision_mask=collision_mask)

    elif bound_type == com.BoundaryType.NEUMANN_CENTER:
        assert False, "No valid use..."

    elif bound_type == com.BoundaryType.CONST_FRAME:
        return M  # no action; assuming the caller function already reads from the single padding but does not modify it

    else:
        assert False, "Unknown boundary type"


def set_obj_bound_neumann_edge_2d(M, collision_mask, here_index):
    """ Enforce Pure Neumann boundary condition on a specific cell of the solid object inside the domain.

    Set the wall values from the adjacent interior domain cell. Zero gradient on the edge, i.e. the boundary is \
    defined on the edge separating the interior and the exterior cells.

    If current cell is obstacle, do nothing.
    If neighbour cell is wall, consider that the lateral solid value has the same value as 'here' to
    enforce Neumann boundary (:math:`dp/dn=0`).

    .. note::
        *Complex Object Boundary*

        This is not perfect. For corner cells on the object there is a race condition as
        what value we should use to copy from. This is because there are more than one
        cell in the active domain to copy from. A remedy would be to take average.
        Nonetheless, because we don't really compute the gradient, and we are just interested
        in the active domain values, we skip improving this step.

        This is for the collocated grid.

        For MACGrid we do not really have to worry about the interior cells inside the
        object when solving, for instance, for the pressure. The pressure gradient on the
        edge cells is already ensured to be zero during the Jacobi iterations, so we do not
        have to explicitly set the object cell values.

    :param ndarray M: input 2d matrix
    :param ndarray collision_mask: 2d object solid mask as in-domain collider
    :param Vector2DInt here_index: cell index
    :return: **M** treated domain
    """
    # if we are in an obstacle, do nothing
    if is_solid_2d(mask=collision_mask, i=here_index.v1(), j=here_index.v2()):
        return M

    # Get here value from current cell
    here_val = M[here_index.v1(), here_index.v2()]

    # 'i' is used for up & down, 'j' is used for left & right

    # up
    if is_solid_2d(mask=collision_mask,
                   i=here_index.v1() - 1,
                   j=here_index.v2()):
        M[here_index.v1() - 1, here_index.v2()] = here_val

    # down
    if is_solid_2d(mask=collision_mask,
                   i=here_index.v1() + 1,
                   j=here_index.v2()):
        M[here_index.v1() + 1, here_index.v2()] = here_val

    # left
    if is_solid_2d(mask=collision_mask,
                   i=here_index.v1(),
                   j=here_index.v2() - 1):
        M[here_index.v1(), here_index.v2() - 1] = here_val

    # right
    if is_solid_2d(mask=collision_mask,
                   i=here_index.v1(),
                   j=here_index.v2() + 1):
        M[here_index.v1(), here_index.v2() + 1] = here_val

    return M


def set_single_cell_wall_bound_2d(M, here_index, bound_type):
    """Boundary condition enforcement on a specific cell of the domain walls.

    :param ndarray M: input 2d matrix
    :param Vector2DInt here_index: cell index
    :param BoundaryType bound_type: boundary enforcement type. Currently only supports Neumann on cell edge
    :return: **M** treated domain
    """

    if bound_type == com.BoundaryType.DIRICHLET:
        assert False, "Not implemented yet.."

    elif bound_type == com.BoundaryType.NEUMANN_EDGE:
        return set_single_cell_wall_neumann_edge_2d(M=M, here_index=here_index)

    elif bound_type == com.BoundaryType.NEUMANN_CENTER:
        assert False, "Not implemented yet.."

    elif bound_type == com.BoundaryType.CONST_FRAME:
        return M  # no action; assuming the caller function already reads from the single padding but does not modify it

    else:
        assert False, "Unknown boundary type"


def set_single_cell_wall_neumann_edge_2d(M, here_index):
    """Enforce Pure Neumann boundary condition on a specific cell of domain walls.

    Set the wall values from the adjacent interior domain cell. Zero gradient on the edge, i.e. the boundary is \
    defined on the edge separating the interior and the wall cells.

    If current cell is wall, do nothing.
    If neighbour cell is obstacle, consider that the lateral solid value has the same value as 'here' to
    enforce Neumann boundary (:math:`dp/dn=0`).

    e.g. when left wall is between :math:`x_0` and :math:`x_1`, then we impose :math:`x_0 = x_1`, where \
    :math:`x_0` is inside the wall and :math:`x_1` is inside the domain.

    .. note::
        We do not care much about what values corner cells are getting because in the
        fluids context only the gradient between the wall and the fluid (the cell edges)
        matters. The gradient between solid wall cells should not affect the projection
        step as it does not make sense to correct the fluid velocity inside the wall.

    :param ndarray M: input 2d matrix
    :param Vector2DInt here_index: cell index
    :return: **M** treated domain walls
    """
    # if we are in an obstacle, do nothing
    if is_wall_2d(M=M, i=here_index.v1(), j=here_index.v2()):
        return M

    # Get here value from current cell
    here_val = M[here_index.v1(), here_index.v2()]

    # 'i' is used for up & down, 'j' is used for left & right

    # up
    if is_wall_2d(M=M,
                  i=here_index.v1() - 1,
                  j=here_index.v2()):
        M[here_index.v1() - 1, here_index.v2()] = here_val

    # down
    if is_wall_2d(M=M,
                  i=here_index.v1() + 1,
                  j=here_index.v2()):
        M[here_index.v1() + 1, here_index.v2()] = here_val

    # left
    if is_wall_2d(M=M,
                  i=here_index.v1(),
                  j=here_index.v2() - 1):
        M[here_index.v1(), here_index.v2() - 1] = here_val

    # right
    if is_wall_2d(M=M,
                  i=here_index.v1(),
                  j=here_index.v2() + 1):
        M[here_index.v1(), here_index.v2() + 1] = here_val

    return M


def set_wall_bound_3d(M, bound_type):
    """ Boundary condition enforcement on the domain walls.

    :param ndarray M: input 3d matrix
    :param BoundaryType bound_type: boundary enforcement type. Currently only supports Neumann.
    :return: **M** treated domain walls
    """
    if bound_type == com.BoundaryType.DIRICHLET:
        assert False, "Not implemented yet.."

    elif bound_type == com.BoundaryType.NEUMANN_EDGE:
        return set_wall_bound_neumann_edge_3d(M=M)

    elif bound_type == com.BoundaryType.NEUMANN_CENTER:
        assert False, "Not implemented yet.."

    elif bound_type == com.BoundaryType.CONST_FRAME:
        return M  # no action; assuming the caller function already reads from the single padding but does not modify it

    else:
        assert False, "Unknown boundary type"


def set_wall_bound_neumann_edge_3d(M):
    """Enforce Pure Neumann boundary condition on the wall cell edges.

    Set the wall values from the adjacent interior domain cell. Zero gradient on the edge, i.e. the boundary is \
    defined on the edge separating the interior and the wall cells.

    e.g. when left wall is between :math:`x_0` and :math:`x_1`, then we impose :math:`x_0 = x_1`, where \
    :math:`x_0` is inside the wall and :math:`x_1` is inside the domain.

    .. note::
        We do not care much about what values corner cells are getting because in the
        fluids context only the gradient between the wall and the fluid (the cell edges)
        matters. The gradient between solid wall cells should not affect the projection
        step as it does not make sense to correct the fluid velocity inside the wall.

    :param ndarray M: input 3d matrix
    :return: **M** treated domain walls
    """
    M[0, :, :] = M[1, :, :]  # up wall
    M[-1:, :, :] = M[-2, :, :]  # down wall
    M[:, 0, :] = M[:, 1, :]  # left wall
    M[:, -1, :] = M[:, -2, :]  # right wall
    M[:, :, 0] = M[:, :, 1]  # front wall
    M[:, :, -1] = M[:, :, -2]  # back wall

    return M


def set_single_cell_wall_bound_3d(M, here_index, bound_type):
    """Boundary condition enforcement on a specific cell of the domain walls.

    :param ndarray M: input 3d matrix
    :param Vector3DInt here_index: cell index
    :param BoundaryType bound_type: boundary enforcement type. Currently only supports Neumann on cell edge
    :return: **M** treated domain
    """

    if bound_type == com.BoundaryType.DIRICHLET:
        assert False, "Not implemented yet.."

    elif bound_type == com.BoundaryType.NEUMANN_EDGE:
        return set_single_cell_wall_neumann_edge_3d(M=M, here_index=here_index)

    elif bound_type == com.BoundaryType.NEUMANN_CENTER:
        assert False, "Not implemented yet.."

    elif bound_type == com.BoundaryType.CONST_FRAME:
        return M  # no action; assuming the caller function already reads from the single padding but does not modify it

    else:
        assert False, "Unknown boundary type"


def set_single_cell_wall_neumann_edge_3d(M, here_index):
    """Enforce Pure Neumann boundary condition on a specific cell of domain walls.

    Set the wall values from the adjacent interior domain cell. Zero gradient on the edge, i.e. the boundary is \
    defined on the edge separating the interior and the wall cells.

    If current cell is wall, do nothing.
    If neighbour cell is obstacle, consider that the lateral solid value has the same value as 'here' to
    enforce Neumann boundary (:math:`dp/dn=0`).

    e.g. when left wall is between :math:`x_0` and :math:`x_1`, then we impose :math:`x_0 = x_1`, where \
    :math:`x_0` is inside the wall and :math:`x_1` is inside the domain.

    .. note::
        We do not care much about what values corner cells are getting because in the
        fluids context only the gradient between the wall and the fluid (the cell edges)
        matters. The gradient between solid wall cells should not affect the projection
        step as it does not make sense to correct the fluid velocity inside the wall.

    :param ndarray M: input 3d matrix
    :param Vector3DInt here_index: cell index
    :return: **M** treated domain walls
    """
    # if we are in an obstacle, do nothing
    if is_wall_3d(M=M, i=here_index.v1(), j=here_index.v2(), k=here_index.v3()):
        return M

    # Get here value from current cell
    here_val = M[here_index.v1(), here_index.v2(), here_index.v3()]

    # 'i is used for up & down, 'j' is used for left & right

    # up
    if is_wall_3d(M=M,
                  i=here_index.v1() - 1,
                  j=here_index.v2(),
                  k=here_index.v3()):
        M[here_index.v1() - 1, here_index.v2(), here_index.v3()] = here_val

    # down
    if is_wall_3d(M=M,
                  i=here_index.v1() + 1,
                  j=here_index.v2(),
                  k=here_index.v3()):
        M[here_index.v1() + 1, here_index.v2(), here_index.v3()] = here_val

    # left
    if is_wall_3d(M=M,
                  i=here_index.v1(),
                  j=here_index.v2() - 1,
                  k=here_index.v3()):
        M[here_index.v1(), here_index.v2() - 1, here_index.v3()] = here_val

    # right
    if is_wall_3d(M=M,
                  i=here_index.v1(),
                  j=here_index.v2() + 1,
                  k=here_index.v3()):
        M[here_index.v1(), here_index.v2() + 1, here_index.v3()] = here_val

    # front
    if is_wall_3d(M=M,
                  i=here_index.v1(),
                  j=here_index.v2(),
                  k=here_index.v3() - 1):
        M[here_index.v1(), here_index.v2(), here_index.v3() - 1] = here_val

    # back
    if is_wall_3d(M=M,
                  i=here_index.v1(),
                  j=here_index.v2(),
                  k=here_index.v3() + 1):
        M[here_index.v1(), here_index.v2(), here_index.v3() + 1] = here_val

    return M


def compute_gradient_2d(M, grad_scale=2., half_dx=.5):
    """ Compute the gradient of a scalar field. The output would be a 2d vector field.

    .. warning::
        This computes the gradient for the central cell. If, for example, we are interested
        in a MACGrid setup we need to be careful how to interpret this.

    .. note::
        The default values are recommended: :code:`grad_scale = 2.0`, :code:`half_dx = 0.5`

    :param ndarray M: input scalar field
    :param float grad_scale: gradient scale (Default= :code:`2.0`)
    :param float half_dx: half the cell size (Default= :code:`0.5`)
    :return: 2d vector field of gradients
    """
    m, n = M.shape

    # initialize the 2d vector field
    Out = np.zeros(shape=(m, n, 2))

    for cx in range(1, m - 1):
        for cy in range(1, n - 1):
            # horizontal (gradient vector 1st dimension): half_dx * (right - left) * grad scale
            left_m = M[cx, cy - 1]
            right_m = M[cx, cy + 1]
            g_h = right_m - left_m
            g_h *= half_dx * grad_scale

            # vertical (gradient vector 2nd dimension): half_dx * (up - down) * grad scale
            up_m = M[cx - 1, cy]
            down_m = M[cx + 1, cy]
            g_v = up_m - down_m
            g_v *= half_dx * grad_scale

            Out[cx, cy, 0] = g_h  # 1st component
            Out[cx, cy, 1] = g_v  # 2nd component

    return Out


def solve_poisson_full_kernel_2d(M, kernel, skip_margin=0):
    """ Solve the Poisson equation using 2d full Poisson kernel.

    Make a convolution pass excluding the boundary, reading it but not updating it.
    The Poisson kernel is always square shape.

    :param ndarray M: input scalar field (matrix)
    :param ndarray kernel: Poisson square convolutional kernel
    :param int skip_margin: number of lateral elements to skip in the convolution. This helps with saving computation
        time when having redundant padding (Default= :code:`0`)
    :return: solution as scalar field (matrix)
    """
    return convolve_kernel_2d(M=M, kernel=kernel, skip_margin=skip_margin)


def convolve_kernel_2d(M, kernel, skip_margin=0):
    """
    Convolve the input 2d matrix with a kernel.

    :param ndarray M: input matrix. It does not have to be square.
    :param ndarray kernel: convolutional kernel
    :param int skip_margin: number of lateral elements to skip in the convolution. This helps with saving computation
        time when having redundant padding (Default= :code:`0`)
    :return: convolved matrix with shrunk size.
    """
    Out = []

    m_k, n_k = kernel.shape
    half_size = int(m_k / 2)  # assuming square kernel

    m, n = M.shape
    min_row = half_size + skip_margin
    max_row = m - (half_size + skip_margin)

    min_col = half_size + skip_margin
    max_col = n - (half_size + skip_margin)

    # Moving convolution kernel window
    for row in range(min_row, max_row):
        for col in range(min_col, max_col):
            conv_window = M[row - half_size: row + half_size + 1, col - half_size: col + half_size + 1]
            Out.append(convolve_kernel_single_pass_2d(M=conv_window, kernel=kernel))

    size_row = max_row - min_row
    size_col = max_col - min_col
    Out = np.array(Out).reshape((size_row, size_col))

    return Out


def convolve_kernel_single_pass_2d(M, kernel):
    """ Convolve the kernel with a matrix of equal size to give one scalar output. No sliding windows is used.

    :param ndarray M: input matrix
    :param ndarray kernel: convolutional kernel matrix
    :return: scalar output
    """
    assert M.shape == kernel.shape, "Inconsistent shapes."

    kernel_flat = kernel.reshape(1, -1)
    M_flat = np.array(M).reshape(-1, 1)
    return np.dot(kernel_flat, M_flat)


def convolve_filter_1d(M, filter_1d, orientation, skip_margin):
    """ Convolve a 2d matrix with a 1d filter.

    :param ndarray M: 2d input matrix. It does not have to be square
    :param ndarray filter_1d: convolutional filter
    :param ConvolutionOrientation orientation: horizontal or vertical filter
    :param int skip_margin: number of lateral elements to skip in the convolution. This helps with saving computation
        time when having redundant padding (Typical value= :code:`0`)
    :return: convolved matrix with shrunk size in the given orientation.
    """
    Out = []

    m, n = M.shape
    half_size = int(filter_1d.size / 2)

    if orientation == com.ConvolutionOrientation.HORIZONTAL:
        # horizontal filter
        min_row = 0
        max_row = m

        min_col = half_size + skip_margin
        max_col = n - (half_size + skip_margin)

        # Moving convolution filter window
        for row in range(min_row, max_row):
            for col in range(min_col, max_col):
                M_row = M[row, col - half_size: col + half_size + 1]
                M_row = np.array(M_row).reshape(-1, 1)
                o = np.dot(filter_1d, M_row)
                Out.append(o)

    elif orientation == com.ConvolutionOrientation.VERTICAL:
        # vertical filter
        min_row = half_size + skip_margin
        max_row = m - (half_size + skip_margin)

        min_col = 0
        max_col = n

        # Moving convolution filter window
        for row in range(min_row, max_row):
            for col in range(min_col, max_col):
                M_col = M[row - half_size: row + half_size + 1, col]
                M_col = np.array(M_col).reshape(-1, 1)
                o = np.dot(filter_1d, M_col)
                Out.append(o)

    else:
        assert False, "Invalid orientation"

    size_row = max_row - min_row
    size_col = max_col - min_col
    Out = np.array(Out).reshape((size_row, size_col))

    return Out


def solve_poisson_separable_filters_from_components_2d(M, U, S, VT, rank, trunc_method, trunc_factor, skip_margin=0):
    """ 2D Poisson solve in a reduced space from SVD components. Filters are computed on the fly and truncated
    before application.

    .. note::
        see :func:`solve_poisson_separable_filters_2d` for details on filter extraction and application.

    :param ndarray M: input 2d matrix to be convolved on
    :param ndarray U: :math:`U` in :math:`USV^T`
    :param ndarray S: :math:`S` in :math:`USV^T`
    :param ndarray VT: :math:`V^T` in :math:`USV^T`
    :param int rank: desired rank. It will be safely clamped if larger than the input matrix actual rank.
    :param float trunc_factor: if the truncation method is :code:`PERCENTAGE` then a value in [0, 1], else fixed floating
        point cut off threshold (:code:`FIXED_THRESHOLD`)
    :param TruncationMode trunc_method: :code:`PERCENTAGE` or :code:`FIXED_THRESHOLD` (*adaptive truncation*)
    :param int skip_margin: number of lateral elements to skip in the convolution. This helps with saving computation
        time when having redundant padding (Default= :code:`0`)
    :return: the solution to Poisson's equation
    """

    import src.functions.decompositions as dec

    # getting horizontal and vertical filters
    if trunc_method == com.TruncationMode.PERCENTAGE:
        v_hor, v_ver, safe_rank = dec.compute_separable_filters_trunc_percent_2d(U=U, S=S, VT=VT, rank=rank,
                                                                                 trunc_factor=trunc_factor)

    elif trunc_method == com.TruncationMode.FIXED_THRESHOLD:
        preserve_shape = True  # prohibit varying size filters, essential for convolution without complications
        v_hor, v_ver, safe_rank = dec.compute_separable_filters_trunc_adaptive_2d(U=U, S=S, VT=VT, rank=rank,
                                                                                  trunc_threshold=trunc_factor,
                                                                                  preserve_shape=preserve_shape)
    else:
        assert False, "Unknown truncation method"

    return solve_poisson_separable_filters_2d(M=M, filter_hor=v_hor, filter_ver=v_ver,
                                              safe_rank=safe_rank, skip_margin=skip_margin)


def solve_poisson_separable_filters_2d(M, filter_hor, filter_ver, safe_rank, skip_margin=0):
    """ Poisson's equation solve in the reduced space using convolution of separable filters with no boundary treatment.

    .. warning::
        No boundary condition treatment.

        See :func:`solve_poisson_separable_filters_obj_aware_2d`
        and :func:`solve_poisson_separable_filters_wall_aware_2d` for the version with boundary treatment.

    This uses multi-rank Poisson filters for a given rank. Given the Poisson equation in the *matrix form*
    :math:`L*X=B`, the Poisson kernel :math:`L` (in forward setup) and its inverse :math:`L^{-1}` (in inverse setup) are
    already baked into the Poisson filters. Just provide the input data matrix and the corresponding filters matching \
    the formulation setup you are interested in.

    .. note::
        *Order of convolution:*

        The convolution order using separable filters in 3D is
        :math:`F * M \\approx \\displaystyle\\sum_{r=1}^n f_{v_r} * (f_{h_r} * M)`

        where
            - :math:`F` - Full Poisson kernel (either :math:`L` or :math:`L^{-1}`)
            - :math:`M` - Input data field
            - :math:`f_h` - Horizontal filter extracted from :math:`V^T` in :math:`SVD(F) = U.S.V^T`, with :math:`S` \
                values absorbed
            - :math:`f_v` - Vertical filter extracted from :math:`U` in :math:`SVD(F) = U.S.V^T`, with :math:`S` \
                values absorbed
            - double subscript :math:`_r` means the filter corresponding the current rank
            - :math:`\\displaystyle\\sum_{r=1}^n` is multi-rank summation (i.e. modal solutions)

        Filters are obtained from Eigen decomposition of :math:`F` using Singular Value Decomposition (*SVD*)
        in the Canonical Polyadic Decomposition (*CPD*) view.
        The convolution order goes from the inner bracket to outer bracket, meaning first we need to
        convolve :math:`M` with the horizontal filter, then convolve the results with the vertical filter.

        For multi-rank convolution we have separate and independent convolutions passes on :math:`M`, then sum up
        the results. The summation comes from the *CPD* view in tensor decomposition,
        which makes it possible to have rank-1 kernel convolutions to get modal solutions taking care of different
        frequencies in the data domain.

    .. warning::
        **DO NOT** feed input data matrix :math:`M` in outer bracket convolution.

        **ALWAYS** use the results of the previous convolution pass to do the next one.

    :param ndarray M: input 2d matrix to be convolved on
    :param ndarray filter_hor: horizontal Poisson filters
    :param ndarray filter_ver: vertical Poisson filters
    :param int safe_rank: desired input rank. 'safe' means a rank that does not exceed the actual rank of the kernel.
        It is the minimum of Poisson kernel :math:`L` rows and columns.
        You can also use :func:`src.functions.decompositions.rank_safety_clamp_2d` to set this
    :param int skip_margin: number of lateral elements to skip in the convolution. This helps with saving computation
        time when having redundant padding (Default= :code:`0`)
    :return: the solution to Poisson's equation
    """
    result = []
    for rank_index in range(safe_rank):
        M_h = convolve_filter_1d(M=M, filter_1d=filter_hor[rank_index - 1],
                                 orientation=com.ConvolutionOrientation.HORIZONTAL,
                                 skip_margin=skip_margin)
        M_v = convolve_filter_1d(M=M_h, filter_1d=filter_ver[rank_index - 1],
                                 orientation=com.ConvolutionOrientation.VERTICAL,
                                 skip_margin=skip_margin)

        # initialize the result
        if rank_index == 0:
            result = M_v

        # summation in the canonical form
        else:
            result += M_v

    return result


def solve_poisson_separable_filters_obj_aware_2d(M, filter_hor, filter_ver, safe_rank, collision_mask,
                                                 opts, individual_rank=None):
    """ Poisson's equation solve in the reduced space using convolution of separable filters with Neumann
    boundary treatment around in-domain solid object.

    .. note::
        See :func:`solve_poisson_separable_filters_2d` for explanation of doing Poisson filter convolution.

    We use *Mirror Marching* algorithm to enforce Neumann boundary condition. See paper.

    :param ndarray M: input 2d matrix to be convolved on
    :param ndarray filter_hor: horizontal Poisson filters
    :param ndarray filter_ver: vertical Poisson filters
    :param int safe_rank: desired input rank. 'safe' means a rank that does not exceed the actual rank of the kernel.
        It is the minimum of Poisson kernel :math:`L` rows and columns.
        You can also use :func:`src.functions.decompositions.rank_safety_clamp_2d` to set this
    :param ndarray collision_mask: 2d object solid mask as in-domain collider
    :param OptionsGeneral opts: general options
    :param int individual_rank: if not None, only return the result of convolving with this single rank
    :return: the solution to Poisson's equation
    """
    result = np.zeros(M.shape)

    if individual_rank is not None:
        assert individual_rank <= safe_rank, "Not enough ranks. Can be because of low target iteration number, " \
                                             "or simply too many ranks in individual_ranks."

        # HORIZONTAL convolution
        M_h = convolve_filter_obj_aware_2d(M=M, filter_1d=filter_hor[individual_rank - 1],
                                           orientation=com.ConvolutionOrientation.HORIZONTAL,
                                           collision_mask=collision_mask)

        # # VERTICAL convolution
        M_v = convolve_filter_obj_aware_2d(M=M_h, filter_1d=filter_ver[individual_rank - 1],
                                           orientation=com.ConvolutionOrientation.VERTICAL,
                                           collision_mask=collision_mask)

        # Sum contribution for rank_index
        result = np.copy(M_v)

    else:  # cumulative rank solution
        # For each rank
        for rank_index in range(safe_rank):
            # HORIZONTAL convolution
            M_h = convolve_filter_obj_aware_2d(M=M, filter_1d=filter_hor[rank_index],
                                               orientation=com.ConvolutionOrientation.HORIZONTAL,
                                               collision_mask=collision_mask)

            # # VERTICAL convolution
            M_v = convolve_filter_obj_aware_2d(M=M_h, filter_1d=filter_ver[rank_index],
                                               orientation=com.ConvolutionOrientation.VERTICAL,
                                               collision_mask=collision_mask)

            # Sum contribution for rank_index
            result += M_v

    # post-processing [Optional]
    # final BC enforcement to make sure pressure values give the right edge gradient
    if opts.boundary.enforce and opts.boundary.post_solve_enforcement:
        post_process_obj_boundary_enforcement_2d(M=result, collision_mask=collision_mask, opts=opts)

    return result


def convolve_filter_obj_aware_2d(M, filter_1d, orientation, collision_mask):
    """Convolve a 2d data domain with a 1d filter of given orientation. Neumann boundary treatment
    around in-domain solid object.

    We use *Mirror Marching* algorithm to enforce Neumann boundary condition. See paper.

    :param ndarray M: input 2d matrix to be convolved on
    :param ndarray filter_1d: convolutional filter
    :param ConvolutionOrientation orientation: filter orientation, horizontal or vertical
    :param ndarray collision_mask: 2d object solid mask as in-domain collider
    :return: convolved matrix with shrunk size in the given orientation
    """
    Out = np.copy(M)

    half_size = int(filter_1d.size / 2)  # also the index of the central element in the filter (always has an odd size)
    rows, cols = M.shape

    # rows: up & down, columns: left & right"
    for ry in range(rows):
        for cx in range(cols):
            if is_solid_2d(mask=collision_mask, i=ry, j=cx):
                Out[ry, cx] = 0.0  # does not matter in this test, but for something like pressure if we want
                # to compute the gradient we should decide what to allocate here. We usually perform a post solve
                # BC enforcement step to assign the best values to the solid cells touching the domain cells.
                continue

            # contribution of current cell
            result = filter_1d[half_size] * M[ry, cx]

            # Preparation for Mirror Marching...

            # offset and step to march in positive / negative directions
            index_pos = com.Vector2DInt(v1=ry, v2=cx)
            index_neg = com.Vector2DInt(v1=ry, v2=cx)

            # init step directions
            step_pos = com.Vector2DInt(v1=0, v2=0)  # positive direction
            step_neg = com.Vector2DInt(v1=0, v2=0)  # negative direction

            # vector used to flip the directions
            flip_vector = com.Vector2DInt(v1=-1, v2=-1)

            # initialize step depending on convolution direction
            # matrix origin top left
            if orientation == com.ConvolutionOrientation.HORIZONTAL:  # walking left & right: fixed row, variable column
                # x axis: columns
                step_pos.set(v1=0, v2=1)  # positive direction:
                step_neg.set(v1=0, v2=-1)  # negative direction

            elif orientation == com.ConvolutionOrientation.VERTICAL:  # walking up & down: variable row, fixed column
                # y axis: rows
                step_pos.set(v1=1, v2=0)  # positive direction
                step_neg.set(v1=-1, v2=0)  # negative direction

            else:
                assert False, "Invalid orientation"

            # marching, start from 1 to half_size
            for f_index in range(1, half_size + 1):

                # Order of marching:
                # 1. Sample the space by taking a step in each direction first
                # 2. If hitting a solid, flip the direction and take a step in the opposite direction

                # take a step
                index_pos.add(step_pos)
                index_neg.add(step_neg)

                # update index of positive march if in solid
                if is_solid_2d(mask=collision_mask, i=index_pos.v1(), j=index_pos.v2()):
                    step_pos.mul(flip_vector)  # flip direction
                    index_pos.add(step_pos)  # update index, take step backward

                # update index of negative march if in solid
                if is_solid_2d(mask=collision_mask, i=index_neg.v1(), j=index_neg.v2()):
                    step_neg.mul(flip_vector)  # flip direction
                    index_neg.add(step_neg)  # update index, take step backward

                # contribution of positive march
                data_pos = M[index_pos.v1(), index_pos.v2()]
                filter_pos = filter_1d[half_size + f_index]
                result += data_pos * filter_pos

                # contribution of negative march
                data_neg = M[index_neg.v1(), index_neg.v2()]
                filter_neg = filter_1d[half_size - f_index]
                result += data_neg * filter_neg

            Out[ry, cx] = result

    return Out


def solve_poisson_separable_filters_wall_aware_2d(M, filter_hor, filter_ver, safe_rank, opts):
    """ Poisson's equation solve in the reduced space using convolution of separable filters with Neumann
    boundary treatment around the domain walls (no in-domain solid object treatment).

    .. note::
        See :func:`solve_poisson_separable_filters_2d` for explanation of doing Poisson filter convolution.

    We use *Mirror Marching* algorithm to enforce Neumann boundary condition. See paper.

    :param ndarray M: input 2d matrix to be convolved on
    :param ndarray filter_hor: horizontal Poisson filters
    :param ndarray filter_ver: vertical Poisson filters
    :param int safe_rank: desired input rank. 'safe' means a rank that does not exceed the actual rank of the kernel.
        It is the minimum of Poisson kernel :math:`L` rows and columns.
        You can also use :func:`src.functions.decompositions.rank_safety_clamp_2d` to set this
    :param OptionsGeneral opts: general options
    :return: the solution to Poisson's equation
    """
    result = np.zeros(M.shape)

    print("Convolving filters over 2D domain")

    # For each rank
    for rank_index in range(safe_rank):
        # HORIZONTAL convolution
        print(f'{com.ConvolutionOrientation.HORIZONTAL.name} pass rank {rank_index + 1}')
        M_h = convolve_filter_wall_aware_2d(M=M, filter_1d=filter_hor[rank_index],
                                            orientation=com.ConvolutionOrientation.HORIZONTAL)

        # # VERTICAL convolution
        print(f'{com.ConvolutionOrientation.VERTICAL.name} pass rank {rank_index + 1}')
        M_v = convolve_filter_wall_aware_2d(M=M_h, filter_1d=filter_ver[rank_index],
                                            orientation=com.ConvolutionOrientation.VERTICAL)

        # Sum contribution for rank_index
        result += M_v

    # post-processing [Optional]
    # final BC enforcement to make sure pressure values give the right edge gradient
    if opts.boundary.enforce and opts.boundary.post_solve_enforcement:
        post_process_wall_boundary_enforcement_2d(M=result, opts=opts)

    return result


def convolve_filter_wall_aware_2d(M, filter_1d, orientation):
    """ Convolve a 2d data domain with a 1d filter of given orientation. Neumann boundary
    treatment around domain walls.

    We use *Mirror Marching* algorithm to enforce Neumann boundary condition. See paper.

    :param ndarray M: input 2d matrix to be convolved on
    :param ndarray filter_1d: convolutional filter
    :param ConvolutionOrientation orientation: filter orientation, horizontal or vertical
    :return: convolved matrix with shrunk size in the given orientation
    """
    Out = np.copy(M)

    half_size = int(filter_1d.size / 2)  # also the index of the central element in the filter (always has an odd size)
    rows, cols = M.shape

    # rows: up & down, columns: left & right"
    for ry in range(rows):
        for cx in range(cols):
            if is_wall_2d(M=M, i=ry, j=cx):
                Out[ry, cx] = 0.0  # does not matter in this test, but for something like pressure if we want
                # to compute the gradient we should decide what to allocate here. We usually perform a post solve
                # BC enforcement # step to assign the best values to the solid cells touching the domain cells.
                continue

            # contribution of current cell
            result = filter_1d[half_size] * M[ry, cx]

            # offset and step to march in positive / negative directions
            index_pos = com.Vector2DInt(v1=ry, v2=cx)
            index_neg = com.Vector2DInt(v1=ry, v2=cx)

            # init step directions
            step_pos = com.Vector2DInt(v1=0, v2=0)  # positive direction
            step_neg = com.Vector2DInt(v1=0, v2=0)  # negative direction

            # vector used to flip the directions
            flip_vector = com.Vector2DInt(v1=-1, v2=-1)

            # initialize step depending on convolution direction
            # matrix origin top left
            if orientation == com.ConvolutionOrientation.HORIZONTAL:  # walking left & right: fixed row, variable column
                # x axis: columns
                step_pos.set(v1=0, v2=1)  # positive direction:
                step_neg.set(v1=0, v2=-1)  # negative direction

            elif orientation == com.ConvolutionOrientation.VERTICAL:  # walking up & down: variable row, fixed column
                # y axis: rows
                step_pos.set(v1=1, v2=0)  # positive direction
                step_neg.set(v1=-1, v2=0)  # negative direction

            else:
                assert False, "Invalid orientation"

            # marching, start from 1 to half_size
            for f_index in range(1, half_size + 1):

                # Order of marching:
                # 1. Sample the space by taking a step in each direction first
                # 2. If hitting a solid, flip the direction and take a step in the opposite direction

                # take a step
                index_pos.add(step_pos)
                index_neg.add(step_neg)

                # update index of positive march if in solid
                if is_wall_2d(M=M, i=index_pos.v1(), j=index_pos.v2()):
                    step_pos.mul(flip_vector)  # flip direction
                    index_pos.add(step_pos)  # update index, take step backward

                # update index of negative march if in solid
                if is_wall_2d(M=M, i=index_neg.v1(), j=index_neg.v2()):
                    step_neg.mul(flip_vector)  # flip direction
                    index_neg.add(step_neg)  # update index, take step backward

                # contribution of positive march
                data_pos = M[index_pos.v1(), index_pos.v2()]
                filter_pos = filter_1d[half_size + f_index]
                result += data_pos * filter_pos

                # contribution of negative march
                data_neg = M[index_neg.v1(), index_neg.v2()]
                filter_neg = filter_1d[half_size - f_index]
                result += data_neg * filter_neg

            Out[ry, cx] = result

    return Out


# ============================================ 3D ============================================

def solve_jacobi_single_padding_only_wall_3d(M, sub_shape_residual, opts):
    """Solve the Poisson equation with Jacobi in the *matrix form* for *forward* and *inverse* Poisson equations in \
    3D with wall Neumann boundary treatment. Walls only.

    .. note::
        We use a general Jacobi setup with flexible :math:`\\alpha` and :math:`\\beta` instead of fixed
        values. This allows to adjust the weights based on the type of the Poisson equation.

    .. note::
        This function only supports Neumann boundary treatment on the cell edges. The walls are single cell padding.

    :param ndarray M: input 3d tensor; if *inverse* setup, :math:`M=B` in :math:`L*X=B`, and if *forward* setup, \
         :math:`M=X` in in :math:`L*X=B`, with :math:`B` being the unknown
    :param OptionsGeneral opts: general options (contains number of iterations along with many other variables)
    :param 3-tuple sub_shape_residual: subdomain shape used in computing the residual
    :return:
        - **Out** - solution
        - **residuals** - residual per iteration
    """

    assert com.is_solver_3d(opts.solver.dim)

    residual = np.zeros(opts.kernel.itr + 1, dtype=np.double)  # +1 to account for the zeroth residual

    def add_residual(k):
        residual[k] = compute_residual_subdomain_3d(X=Out,
                                                    B=M,
                                                    sub_shape=sub_shape_residual)

    # initialize the solution
    Out = np.copy(M)  # acts as warm start
    if opts.solver.zero_init:
        set_submatrix_zero_3d(M=Out, skip_margin=1)  # keeping the single padding boundary intact

    sign = gen.get_alpha_sign(solver_type=opts.solver.solver_type, kernel_type=opts.kernel.kernel_type)

    rows, cols, depth = M.shape

    add_residual(0)

    for ii in range(1, opts.kernel.itr + 1):  # to account for 0 residual
        print(f'solving Jacobi matrix form itr {ii}..')

        o_last = np.copy(Out)  # update overall collective values

        # wall boundary
        if opts.boundary.enforce:
            o_last = set_wall_bound_3d(M=o_last, bound_type=opts.boundary.condition)

        # rows: up & down, columns: left & right"
        for cy in range(1, rows - 1):
            for cx in range(1, cols - 1):
                for cz in range(1, depth - 1):
                    # object boundary
                    if opts.boundary.enforce and opts.boundary.obj_collide:
                        # Same solid cell might acquire different values based on which current cell
                        # we are updating for. This is to ensure we get the same values for the neighbour
                        # and the central cell in computing the central cell update.
                        # because of this potentially changing solid value, we need to keep the
                        # boundary enforcement step inside this nested loop... can't really do it outside as a separate
                        # step. It would cause confusion and race condition for the solid cells.
                        o_last = set_single_cell_wall_bound_3d(M=o_last,
                                                               here_index=com.Vector3DInt(v1=cy, v2=cx, v3=cz),
                                                               bound_type=opts.boundary.condition)

                        if is_wall_3d(M=M, i=cy, j=cx, k=cz):
                            continue

                    o_u = o_last[cy - 1, cx, cz]  # up
                    o_d = o_last[cy + 1, cx, cz]  # down
                    o_r = o_last[cy, cx + 1, cz]  # right
                    o_l = o_last[cy, cx - 1, cz]  # left
                    o_f = o_last[cy, cx, cz - 1]  # front
                    o_b = o_last[cy, cx, cz + 1]  # back

                    input_c = M[cy, cx, cz]
                    # update
                    Out[cy, cx, cz] = \
                        (o_l + o_r + o_u + o_d + o_f + o_b + sign * opts.kernel.alpha * input_c) / opts.kernel.beta

        add_residual(ii)

    # post-processing [Optional]
    # final BC enforcement to make sure pressure values give the right edge gradient
    if opts.boundary.enforce and opts.boundary.post_solve_enforcement:
        post_process_wall_boundary_enforcement_3d(M=Out, opts=opts)

    return Out, residual.reshape(-1, 1)


def post_process_wall_boundary_enforcement_3d(M, opts):
    """Boundary enforcement to make sure domain values give the right edge gradient (Neumann). Usually used as an \
    optional last step clean up after the Poisson equation solve. Walls only.

    :param ndarray M: input 3d matrix
    :param OptionsGeneral opts: general options (contains boundary enforcement type, but only supports Neumann for now)
    :return: **M** treated domain
    """
    rows, cols, depth = M.shape

    M = set_wall_bound_3d(M=M, bound_type=opts.boundary.condition)

    # rows: up & down, columns: left & right, depth: front and back
    for cy in range(1, rows - 1):
        for cx in range(1, cols - 1):
            for cz in range(1, depth - 1):
                if opts.boundary.obj_collide:
                    M = set_single_cell_wall_bound_3d(M=M,
                                                      here_index=com.Vector3DInt(v1=cy, v2=cx, v3=cz),
                                                      bound_type=opts.boundary.condition)

    return M


def solve_poisson_separable_filters_wall_aware_3d(M, filters_1d, safe_rank, opts):
    """ Poisson's equation solve in the reduced space using convolution of separable filters with Neumann
    boundary treatment around the domain walls (no in-domain solid object treatment) - 3D.

    We use *Mirror Marching* algorithm to enforce Neumann boundary condition. See paper.

    We use multi-rank Poisson filters for a given rank. Given the Poisson equation in the *matrix form*
    :math:`L*X=B`, the Poisson kernel :math:`L` (in forward setup) and its inverse :math:`L^{-1}` (in inverse setup) are
    already baked into the Poisson filters. Just provide the input data matrix and the corresponding filters matching
    the formulation setup you are interested in.

    .. note::
        *Order of convolution:*

        The convolution order using separable filters in 3D is
        :math:`F * M \\approx \\displaystyle\\sum_{r=1}^n f_{v_r} * (f_{h_r} * (f_{d_r} * M))`

        where
            - :math:`F` - Full Poisson kernel (either :math:`L` or :math:`L^{-1}`)
            - :math:`M` - Input data field (tensor)
            - :math:`f_v` - Vertical filter
            - :math:`f_h` - Horizontal filter
            - :math:`f_d` - Depth (fiber) filter
            - double subscript :math:`_r` means the filter corresponding the current rank
            - :math:`\\displaystyle\\sum_{r=1}^n` is multi-rank summation (i.e. modal solutions)

        Filters are obtained from Tensor Eigen Decomposition of :math:`F` using
        *Symmetric-CP* (Symmetric Canonical Polyadic Decomposition). The convolution order goes from the inner
        brackets to outer brackets, meaning first we need to convolve :math:`M` with the fiber filter, then
        convolve the results with the horizontal and vertical filters.

        For multi-rank convolution we have separate and independent convolutions passes on :math:`M`, then sum up
        the results. The summation comes from the Canonical Polyadic Decomposition (*CPD*) view in tensor decomposition,
        which makes it possible to have rank-1 kernel convolutions to get modal solutions taking care of different
        frequencies in the data domain.


    .. warning::
        **DO NOT** feed input data matrix :math:`M` in outer bracket convolution.

        **ALWAYS** use the results of the previous convolution pass to do the next one.

    .. warning::
        *Safe Rank*

        While in 2D 'safe' means a rank that does not exceed the actual rank of the kernel, in 3D it is different.
        Due to the lack of a clear definition of rank in tensor decomposition, we can pretty much ask for any rank in
        the CP-vew eigen decomposition and always get something that "works". This vagueness of a proper rank definition
        in 3D is partially linked to the fact that tensor decomposition is NP-hard.

        In this function 'safe' simply means the number of ranked filters we would like to include in our convolutions,
        and use it to help with setting up loops etc.

    :param ndarray M: input 3d matrix (tensor) to be convolved on
    :param ndarray filters_1d: ranked (but not sorted) filters. Same filter is used for all 3 convolutional
        orientations: horizontal, vertical, and fiber (depth)
    :param int safe_rank: the number of ranked filters to be used in convolution
    :param OptionsGeneral opts: general options
    :return: the solution to Poisson's equation
    """
    print("Convolving filters over 3D domain")

    result = np.zeros(M.shape)

    # For each rank
    for rank_index in range(safe_rank):
        print(f'{com.ConvolutionOrientation.FIBER.name} pass rank {rank_index + 1}')
        # FIBER (depth) convolution
        M_f = convolve_filter_wall_aware_3d(M=M, filter_1d=filters_1d[rank_index],
                                            orientation=com.ConvolutionOrientation.FIBER)
        print(f'{com.ConvolutionOrientation.HORIZONTAL.name} pass rank {rank_index + 1}')
        # HORIZONTAL convolution
        M_h = convolve_filter_wall_aware_3d(M=M_f, filter_1d=filters_1d[rank_index],
                                            orientation=com.ConvolutionOrientation.HORIZONTAL)
        print(f'{com.ConvolutionOrientation.VERTICAL.name} pass rank {rank_index + 1}')
        # VERTICAL convolution
        M_v = convolve_filter_wall_aware_3d(M=M_h, filter_1d=filters_1d[rank_index],
                                            orientation=com.ConvolutionOrientation.VERTICAL)

        # Sum contribution for rank_index
        result += M_v

    # post-processing [Optional]
    # final BC enforcement to make sure pressure values give the right edge gradient
    if opts.boundary.enforce and opts.boundary.post_solve_enforcement:
        post_process_wall_boundary_enforcement_3d(M=result, opts=opts)

    return result


def convolve_filter_wall_aware_3d(M, filter_1d, orientation):
    """Convolve a 3d data domain with a 1d filter of given orientation. Neumann boundary treatment around domain walls.

    We use *Mirror Marching* algorithm to enforce Neumann boundary condition. See paper.

    :param ndarray M: input 3d matrix (tensor) to be convolved on
    :param ndarray filter_1d: convolutional filter
    :param ConvolutionOrientation orientation: filter orientation, horizontal, vertical, or fiber (depth)
    :return: convolved tensor with shrunk size in the given orientation
    """
    Out = np.copy(M)

    half_size = int(filter_1d.size / 2)  # also the index of the central element in the filter (always has an odd size)
    rows, cols, depth = M.shape

    # rows: up & down, columns: left & right, depth: front & back
    for ry in range(rows):
        for cx in range(cols):
            for dz in range(depth):
                if is_wall_3d(M=M, i=ry, j=cx, k=dz):
                    Out[ry, cx, dz] = 0.0  # does not matter in this test, but for something like pressure if we want
                    # to compute the gradient we should decide what to allocate here. We usually perform a post solve
                    # BC enforcement # step to assign the best values to the solid cells touching the domain cells.
                    continue

                # contribution of current cell
                result = filter_1d[half_size] * M[ry, cx, dz]

                # offset and step to march in positive / negative directions
                index_pos = com.Vector3DInt(v1=ry, v2=cx, v3=dz)
                index_neg = com.Vector3DInt(v1=ry, v2=cx, v3=dz)

                # init step directions
                step_pos = com.Vector3DInt(v1=0, v2=0, v3=0)  # positive direction
                step_neg = com.Vector3DInt(v1=0, v2=0, v3=0)  # negative direction

                # vector used to flip the directions
                flip_vector = com.Vector3DInt(v1=-1, v2=-1, v3=-1)

                # initialize step depending on convolution direction
                # matrix origin top left
                if orientation == com.ConvolutionOrientation.FIBER:  # walking back & front: fixed row and columns,
                    # variable depth
                    step_pos.set(v1=0, v2=0, v3=1)  # positive direction:
                    step_neg.set(v1=0, v2=0, v3=-1)  # negative direction

                elif orientation == com.ConvolutionOrientation.HORIZONTAL:  # walking left & right: fixed row and depth,
                    # variable column
                    step_pos.set(v1=0, v2=1, v3=0)  # positive direction:
                    step_neg.set(v1=0, v2=-1, v3=0)  # negative direction

                elif orientation == com.ConvolutionOrientation.VERTICAL:  # walking up & down: variable row,
                    # fixed column and depth
                    step_pos.set(v1=1, v2=0, v3=0)  # positive direction
                    step_neg.set(v1=-1, v2=0, v3=0)  # negative direction

                else:
                    assert False, "Invalid orientation"

                # marching, start from 1 to half_size
                for f_index in range(1, half_size + 1):

                    # Order of marching:
                    # 1. Sample the space by taking a step in each direction first
                    # 2. If hitting a solid, flip the direction and take a step in the opposite direction

                    # take a step
                    index_pos.add(step_pos)
                    index_neg.add(step_neg)

                    # update index of positive march if in solid
                    if is_wall_3d(M=M, i=index_pos.v1(), j=index_pos.v2(), k=index_pos.v3()):
                        step_pos.mul(flip_vector)  # flip direction
                        index_pos.add(step_pos)  # update index, take step backward

                    # update index of negative march if in solid
                    if is_wall_3d(M=M, i=index_neg.v1(), j=index_neg.v2(), k=index_neg.v3()):
                        step_neg.mul(flip_vector)  # flip direction
                        index_neg.add(step_neg)  # update index, take step backward

                    # contribution of positive march
                    data_pos = M[index_pos.v1(), index_pos.v2(), index_pos.v3()]
                    filter_pos = filter_1d[half_size + f_index]
                    result += data_pos * filter_pos

                    # contribution of negative march
                    data_neg = M[index_neg.v1(), index_neg.v2(), index_neg.v3()]
                    filter_neg = filter_1d[half_size - f_index]
                    result += data_neg * filter_neg

                Out[ry, cx, dz] = result

    return Out


def outer_product_2d(a, b):
    """ Outer product of two vectors.

    :param a: first vertical vector
    :param b: second horizontal vector
    :return: :math:`a \\times b`
    """
    return np.outer(a, b)


def generate_bivariate_gaussian(size, ef_rad, center=None):
    """ Generate a normalized 2d gaussian with maximum 1 and minimum 0.

    The effective radius (ef_rad) will be automatically computed to get :math:`\\mu_x` and :math:`\\mu_y`.

    :param int size: size of the 2d matrix
    :param float ef_rad: effective radius, full-width-half-maximum
    :param ndarray center: if None, the center of the matrix, else movable center (Default= :code:`None`)
    :return: 2d Gaussian
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return bivariate_gaussian_normalized(x=x, y=y, x0=x0, y0=y0, ef_rad=ef_rad)


def bivariate_gaussian_normalized(x, y, x0, y0, ef_rad):
    """Generate a normalized 2d gaussian.

    :param ndarray x: x values
    :param ndarray y: y values
    :param float x0: mean x
    :param float y0: mean y
    :param float ef_rad: effective radius, full-width-half-maximum
    :return: normalized 2d Gaussian
    """
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / ef_rad ** 2)


def make_9_tiles_mirrored_2d(M):
    r""" Given the input matrix make 9 tiles arrangement, where the central block is the input,
    and each of the remaining 8 tiles are the mirrored of their neighbours. Each edge separating two tiles acts as \
    the axis the mirrored image is flipped. The output therefore has 3x the size of the original matrix.

    For example for a 2x2 matrix

    .. math::
        \begin{bmatrix}
        1 & 2 \\
        a & b
        \end{bmatrix}

    its 9-tiles structure looks like this, with the matrix being the central block:

    .. math::
        \begin{bmatrix}
        b & a & a & b & b & a \\
        2 & 1 & 1 & 2 & 2 & 1 \\
        2 & 1 & \mathbf{1} & \mathbf{2} & 2 & 1 \\
        b & a & \mathbf{a} & \mathbf{b} & b & a \\
        b & a & a & b & b & a \\
        2 & 1 & 1 & 2 & 2 & 1
        \end{bmatrix}

    :param ndarray M: input 2d matrix
    :return: 9-tiles matrix, with 3 times the size of the input matrix
    """
    # middle column
    data_center = M
    data_center_down = np.flip(data_center, axis=0)
    data_center_up = np.flip(data_center, axis=0)
    # stack vertically
    data_middle = np.concatenate((data_center_up, data_center), axis=0)
    data_middle = np.concatenate((data_middle, data_center_down), axis=0)

    # left column
    data_center = np.flip(M, axis=1)
    data_center_down = np.flip(data_center, axis=0)
    data_center_up = np.flip(data_center, axis=0)
    # stack vertically
    data_left = np.concatenate((data_center_up, data_center), axis=0)
    data_left = np.concatenate((data_left, data_center_down), axis=0)

    # right column - same as left column
    data_right = np.copy(data_left)

    data_mirrored = np.concatenate((data_left, data_middle), axis=1)
    data_mirrored = np.concatenate((data_mirrored, data_right), axis=1)

    return data_mirrored


def make_4_tiles_mirrored_2d(M, is_odd):
    """Given the input matrix make *even* or *odd* tile block arrangement where the top left block is the

    See :func:`make_4_tiles_mirrored_odd_2d` and :func:`make_4_tiles_mirrored_even_2d` for more details.

    :param ndarray M: input 2d matrix
    :param bool is_odd: even or odd arrangement
    :return: even or odd mirrored matrix
    """
    if is_odd:
        return make_4_tiles_mirrored_odd_2d(M=M)
    else:
        return make_4_tiles_mirrored_even_2d(M=M)


def make_4_tiles_mirrored_even_2d(M):
    r"""Given the input matrix make 4 tiles block *even* arrangement where the top left block is the input,
    the top right block is the mirror of the first block and the two bottom blocks are the
    mirrors of the top blocks.

    For example, given

    .. math::
        \begin{bmatrix}
        1 & 2 \\
        a & b
        \end{bmatrix}

    its 4-tiles *even* structure looks like this:

    .. math::
        \begin{bmatrix}
        \mathbf{1} & \mathbf{2} & 2 & 1 \\
        \mathbf{a} & \mathbf{b} & b & a \\
        a & b & b & a \\
        1 & 2 & 2 & 1
        \end{bmatrix}

    :param ndarray M: input 2d matrix
    :return: even mirrored matrix
    """
    # top left block
    data_top_left = M
    # top right block
    data_top_right = np.flip(data_top_left, axis=1)
    # top row with 2 blocks
    data_top_blocks = np.concatenate((data_top_left, data_top_right), axis=1)

    # bottom left block
    data_bottom_left = np.flip(data_top_left, axis=0)
    # bottom right block
    data_bottom_right = np.flip(data_top_right, axis=0)
    # bottom row with 2 blocks
    data_bottom_blocks = np.concatenate((data_bottom_left, data_bottom_right), axis=1)

    # concatenate all
    data_mirrored = np.concatenate((data_top_blocks, data_bottom_blocks), axis=0)

    return data_mirrored


def make_4_tiles_mirrored_odd_2d(M):
    r"""Given the input matrix make 4 tiles block *odd* arrangement where the top left block is the input,
    the top right block is the mirror of the first block and the two bottom blocks are the mirrors of the top blocks.

    For example, given

    .. math::
        \begin{bmatrix}
        1 & 2 \\
        a & b
        \end{bmatrix}

    its 4-tiles *even* structure looks like this:

    .. math::
        \begin{bmatrix}
        \mathbf{1} & \mathbf{2} & 1 \\
        \mathbf{a} & \mathbf{b} & a \\
        1 & 2 & 1
        \end{bmatrix}

    :param ndarray M: input 2d matrix
    :return: odd mirrored matrix
    """
    rows, cols = M.shape

    data_mirrored = np.zeros(shape=(2 * rows - 1, 2 * cols - 1))

    # top left + top middle column and left middle row + center
    data_mirrored[0:rows, 0:cols] = M

    # top right + right middle row
    M_tmp = np.flip(M, axis=1)
    data_mirrored[0:rows, cols:] = M_tmp[0:rows, 1:cols]

    # bottom left + bottom middle column
    M_tmp = np.flip(M, axis=0)
    data_mirrored[rows:, 0:cols] = M_tmp[1:, 0:cols]

    # bottom right
    M_tmp = np.flip(M, axis=1)
    M_tmp = np.flip(M_tmp, axis=0)
    data_mirrored[rows:, cols:] = M_tmp[1:, 1:]

    return data_mirrored


def zero_out_small_values(M, threshold=1e-9):
    """Zero out all the elements whose absolute values are below the threshold

    :param ndarray M: input array
    :param float threshold: non-negative
    :return: same as input with zeros in place of small values
    """
    assert threshold >= 0, "Threshold must be non-negative"

    M[np.abs(M) < threshold] = 0
    return M


def compute_ssim(test_image, ref_image):
    """Compute *SSIM* - Structural Similarity Index.

    Measure the *perceptual* similarity and difference between two images, as well as the gradient and the mean SSIM.
    Check out `the formal definition <https://en.wikipedia.org/wiki/Structural_similarity>`_ and these
    `examples <https://ece.uwaterloo.ca/~z70wang/research/ssim/>`_.

    **How to compute the SSIM difference**:

    :math:`S` is the local similarity index. It is usually expected to be in [0, 1] but it can also be [-1, 1] where
    negative values mean the same structure but with inverted values (it is due to a cross product).
    Since we do not really care about inversion we take the absolute value of :math:`S` to make it [0, 1].
    The computed difference, working with everything normalized, gives :code:`0` when the two matrices are exactly the
    same (because :math:`S` will be 1), and :code:`1` when they are completely different (:math:`S` will be 0).

    *Suggested vragne for plotting: [-1, 1].*

    :param ndarray test_image:
    :param ndarray ref_image:
    :return:
        - **ssim_image** - structural similarity indices. The full SSIM image. This is only returned if `full` is set \
        to :code:`True`
        - **ssim_diff** - structural difference
        - **ssim_grad** - structural similarity gradient
        - **ssim_mean** - mean ssim scalar value
    """

    from skimage.metrics import structural_similarity as ssim
    ssim_mean, ssim_grad, ssim_image = ssim(ref_image, test_image, gradient=True, full=True,
                                            data_range=np.max(test_image) - np.min(test_image))

    # SSIM difference (read the notes up in the function documentation)
    # ASSUMING S IS ALWAYS POSITIVE
    ssim_diff = ref_image - np.multiply(ssim_image, test_image)
    ssim_diff = normalize_range(np.abs(ssim_diff))

    return ssim_image, ssim_diff, ssim_grad, ssim_mean


def mse(x, y=None, axis=None):
    """Mean squared error of two matrices. If only one is given, returns the *L2-norm* of x.

    :param ndarray x: input array/matrix
    :param ndarray y: input array/matrix, optional (Default= :code:`None`)
    :param int, 2-tuple of ints axis: specifies the axis along which to compute the vector norms (Default= :code:`None`)
    :return: error scalar
    """
    if axis is not None:
        n = np.linalg.norm(x - y, axis=axis) if y is not None else np.linalg.norm(x, axis=axis)
    else:
        n = np.linalg.norm(x - y) if y is not None else np.linalg.norm(x)
    return n * n * (1.0 / x.size)


def mre(x, y=None, abs_err=False, axis=None):
    """ Mean relative error of two matrices. If only one matrix is given, returns the average of *L1* norm of x.

    :param ndarray x: input array/matrix
    :param ndarray y: input array/matrix, optional (Default= :code:`None`)
    :param bool abs_err: if :code:`True` use the absolute values of the inputs to compute the error
        (Default= :code:`False`)
    :param int, 2-tuple of ints axis: specifies the axis along which to compute the vector norms (Default= :code:`None`)
    :return: error scalar
    """
    n = np.abs(x) if abs_err else x
    m = (np.abs(y) if abs_err else y) if y is not None else np.zeros_like(n)
    if axis is not None:
        return np.average(n - m, axis=axis)
    return np.average(n - m)


def ms_norm(M):
    """Mean squared norm of the matrix. Element-wise.

    :param ndarray M: input matrix
    :return: error scalar
    """
    return (np.square(M)).mean()


def frobenius_norm(M):
    """Frobenius norm of the matrix.

    :param ndarray M: input matrix
    :return: error scalar
    """
    # return np.linalg.norm(M, ord='fro')
    return np.sqrt(np.sum(np.square(M)))  # equivalent to the line above but works with 3D tensor too


def inf_norm(M):
    """Infinite norm of the matrix.

    :param ndarray M: input matrix
    :return: error scalar
    """
    return abs(M).max()


def compute_norm(M, method):
    """ Different norms of the matrix.

    :param ndarray M: input matrix
    :param NormOptions method: frobenius, mse, infinite
    :return: error scalar
    """

    if method == com.NormOptions.FROBENIUS:
        norm = frobenius_norm(M=M)

    elif method == com.NormOptions.MSE:
        norm = ms_norm(M=M)

    elif method == com.NormOptions.INF:
        norm = inf_norm(M=M)

    else:
        assert False, 'Unknown norm function'

    return norm


def compute_l1_norm_error(M1, M2):
    """Compute *L1* norm of the difference between two matrices.

    :param ndarray M1: input matrix
    :param ndarray M2: input matrix
    :return: error scalar
    """
    return np.linalg.norm((M1 - M2), ord=1)


def compute_abs_rel_error(M1, M2):
    """Compute absolute and relative errors, with :code:`M1` being the reference matrix

    :param ndarray M1: reference matrix
    :param ndarray M2: test matrix
    :return: absolute error, relative error in %
    """
    err_abs = np.abs(M1 - M2)
    epsilon = 1e-5  # regularization to avoid division by zero
    epsilon_m = epsilon * np.ones_like(M1)
    err_rel = 100. * err_abs / (np.abs(M1) + epsilon_m)  # relative error in percentage

    return err_abs, err_rel


def compute_residual_poisson_operator(X, B, solver_dimension):
    """Compute *L2-norm* residual :math:`r= L*X - B` using 2d or 3d Laplacian :math:`L` in the *matrix form*
    (equivalent to :math:`r=Ax-b` in the *vector form*).

    .. warning::
        This is based on the *matrix form* setup, excluding wall boundaries by one cell, and \
        is only valid for *inverse* Poisson equation. The *forward* Poisson version needs \
        to be implemented.

    .. warning::
        This function is not safe if **not** excluding wall boundaries. Use this function in subdomains to properly
        exclude the wall boundaries.

    :param ndarray X: input in :math:`L*X=B`
    :param ndarray B: input in :math:`L*X=B`
    :param SolverDimension solver_dimension: 2d or 3d
    :return: residual scalar
    """

    res_mat = compute_residual_poisson_tensor_operator(X=X, B=B, solver_dimension=solver_dimension)

    if com.is_solver_2d(solver_dimension):
        residual = np.linalg.norm(res_mat.reshape((-1, 1)))

    elif com.is_solver_3d(solver_dimension):
        residual = np.linalg.norm(res_mat.reshape((-1, 1, 1)))
    else:
        assert False, "Unknown Solver Dimension"

    return residual


def compute_residual_poisson_tensor_operator(X, B, solver_dimension):
    """Compute residual matrix/tensor :math:`r= L*X - B` using 2d or 3d Laplacian :math:`L` in the *matrix form*
    (equivalent to :math:`r=Ax-b` in the *vector form*).

    .. warning::
        This is based on the *matrix form* setup, excluding wall boundaries by one cell, and \
        is only valid for *inverse* Poisson equation. The *forward* Poisson version needs \
        to be implemented.

    .. warning::
        This function is not safe if **not** excluding wall boundaries. Use this function in subdomains to properly
        exclude the wall boundaries.

    :param ndarray X: input in :math:`L*X=B`
    :param ndarray B: input in :math:`L*X=B`
    :param SolverDimension solver_dimension: 2d or 3d
    :return: residual matrix/tensor
    """
    residual = np.zeros_like(B, dtype=np.float32)

    # enforcing 1 padding to make sure a correct way of handling of the marginal cells in Laplacian
    # This assumes the div values have already been baked with a negative sign, and hence + instead of -

    if com.is_solver_2d(solver_dimension):
        residual[1:-1, 1:-1] = B[1:-1, 1:-1] - apply_laplacian_2d(X=X)

    elif com.is_solver_3d(solver_dimension):
        residual[1:-1, 1:-1, 1:-1] = B[1:-1, 1:-1, 1:-1] - apply_laplacian_3d(X=X)

    else:
        assert False, "Unknown Solver Dimension"

    return residual


def compute_residual_subdomain(X, B, sub_shape, solver_dimension):
    """Compute subdomain *L2-norm* residual :math:`r= L*X - B` using 2d or 3d Laplacian :math:`L` in the *matrix form*
    (equivalent to :math:`r=Ax-b` in the *vector form*).

    .. warning::
        This is based on the *matrix form* setup, excluding wall boundaries by one cell, and \
        is only valid for *inverse* Poisson equation. The *forward* Poisson version needs \
        to be implemented.

    :param ndarray X: input in :math:`L*X=B`
    :param ndarray B: input in :math:`L*X=B`
    :param 2-tuple or 3-tuple sub_shape: subdomain shape, 2d or 3d
    :param SolverDimension solver_dimension: 2d or 3d
    :return: residual (2norm scalar)
    """
    if com.is_solver_2d(solver_dimension):
        return compute_residual_subdomain_2d(X=X, B=B, sub_shape=sub_shape)

    elif com.is_solver_3d(solver_dimension):
        return compute_residual_subdomain_3d(X=X, B=B, sub_shape=sub_shape)

    else:
        assert False, "Unknown Solver Dimension"


def compute_residual_subdomain_2d(X, B, sub_shape):
    """Compute 2d subdomain *L2-norm* residual :math:`r= L*X - B` using 2d Laplacian :math:`L` in the *matrix form*
    (equivalent to :math:`r=Ax-b` in the *vector form*).

     .. warning::
        This is based on the *matrix form* setup, excluding wall boundaries by one cell, and \
        is only valid for *inverse* Poisson equation. The *forward* Poisson version needs \
        to be implemented.

    :param ndarray X: input in :math:`L*X=B`
    :param ndarray B: input in :math:`L*X=B`
    :param 2-tuple sub_shape: subdomain shape
    :return: residual (2norm scalar)
    """
    # extract the solution to match the fixed size data domain
    X = extract_from_center_2d(M=X, sub_shape=sub_shape)
    B = extract_from_center_2d(M=B, sub_shape=sub_shape)

    return compute_residual_poisson_operator(X=X, B=B, solver_dimension=com.SolverDimension.D2)


def compute_residual_subdomain_3d(X, B, sub_shape):
    """Compute 3d subdomain *L2-norm* residual :math:`r= L*X - B` using 2d Laplacian :math:`L` in the *matrix form*
    (equivalent to :math:`r=Ax-b` in the *vector form*).

     .. warning::
        This is based on the *matrix form* setup, excluding wall boundaries by one cell, and \
        is only valid for *inverse* Poisson equation. The *forward* Poisson version needs \
        to be implemented.

    :param ndarray X: input in :math:`L*X=B`
    :param ndarray B: input in :math:`L*X=B`
    :param 3-tuple sub_shape: subdomain shape
    :return: residual (2norm scalar)
    """
    # extract the solution to match the fixed size data domain
    X = extract_from_center_3d(M=X, sub_shape=sub_shape)
    B = extract_from_center_3d(M=B, sub_shape=sub_shape)

    return compute_residual_poisson_operator(X=X, B=B, solver_dimension=com.SolverDimension.D3)


def normalize_range(M, symmetric=False):
    """Normalize the matrix values.

    :param ndarray M: input matrix
    :param bool symmetric: [-1, +1] if :code:`True`, else [0, +1], optional (Default= :code:`True`)
    :return: normalized matrix
    """
    a = np.copy(M)
    if symmetric:
        a -= np.min(a)
        a = a / np.max(a)
        a -= 0.5
        a *= 2
    else:
        a -= np.min(a)
        a = a / np.max(a)
    return a


def interp_1darray(arr, resolution, symmetric_x_axis=True, kind='cubic'):
    """Interpolate a 1d array.

    :param 1darray arr: input
    :param int resolution: :code:`1` no interpolation, :code:`1>` increased resolution
    :param str kind: interpolation function (Default=cubic); see :code:`scipy.interpolate` for options.
    :param bool symmetric_x_axis: if :code:`True` add points equally to each side of the array center
    :return: up-sampled indices and array
    """
    from scipy.interpolate import interp1d

    size = arr.shape[0]

    if symmetric_x_axis:
        max_index = int(size / 2)
        min_index = -max_index
    else:
        max_index = 0
        min_index = size - 1

    # the following reshaping is to make sure interp works for (n,) and (n,1) shapes no matter the shape
    index = np.linspace(min_index, max_index, num=size).reshape(-1, 1)[:, 0]
    intp = interp1d(index, arr.reshape(-1, 1)[:, 0], kind=kind)
    index_new = np.linspace(min_index, max_index, num=resolution * size)

    return index_new, intp(index_new)


def convergence_rate(defects):
    """ Compute the convergence for a multi-grid setup. Inspired by
    `this work <https://julianroth.org/documentation/multigrid/Multigrid.html>`_.

    :param defects:
    :return:
        - average convergence rate
        - convergence rates
    """
    vals = [d ** 2 for d in defects]
    orders = []

    for i in range(len(vals) - 3):
        orders.append(
            np.log(abs((vals[i + 3] - vals[i + 2]) / (vals[i + 2] - vals[i + 1])))
            / np.log(abs((vals[i + 2] - vals[i + 1]) / (vals[i + 1] - vals[i])))
        )

    conv_rates = orders
    avg_conv_rate = sum(orders) / len(orders)

    return avg_conv_rate, conv_rates
