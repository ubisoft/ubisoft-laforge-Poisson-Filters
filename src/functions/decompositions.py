""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`

    .. _decomposition_method:

    Technicals
    ===============

    While in 2D the eigenvalues are sorted, it is not guaranteed to be true in 3D. This is due to the lack of consensus
    on rank definition in tensor decomposition, and the fact that current symmetric CP algorithm used by our method, is
    an iterative method that does its best to deal with a NP-hard problem.

    This difference between 2D and 3D eigenvalues properties has an important implication in practice:

        - In 2D it is safe to generate, say, rank-8 filters and only use, for example, the first 3 to capture a \
        significant portion of the variance. The contribution of the 4th rank is expected to be less than the \
        first 3 ranks.

        - In 3D we might get different filters if we first get, say, rank-8 filters and choose the first 3, compared \
        to when we directly compute rank-3 filters and use all of them. In both cases we are interested in the first \
        3 ranks, but in the latter case results are expected to be numerically more stable. There is no guarantee \
        to get a decreasing variance contribution with higher ranks in 3D, just like what one would expect in 2D.

    *Poisson filter computation from decomposition components*
        Check out :func:`get_filters_from_svd_components_2d` to see how we do filter computation. The same method
        applied to 3D using *Sym-CP* and with an additional set of vectors for the 3rd dimension.

    *Absorbing eigenvalues in 2D and 3D*
        - **2D** : Because the Poisson kernel is always square with non-negative eigenvalues, the singular values \
        obtained from SVD coincide with the eigenvalues. This is not true in general, and is only true in our case \
        given the aforementioned properties of the kernel. To absorb the eigenvalues in filters, we scale each of the \
        two ranked filters by\
         :math:`\\sqrt{\\sigma_{r}}`, the eigenvalue corresponding to rank :math:`r`.

        - **3D** : Symmetric-CP decomposition of 3D Poisson kernel gives a set of factors and cores, where cores can \
        be taken as the best approximation of true tensor eigenvalues (this should always be taken with a grain of  \
        salt for tensor eigen decomposition). To absorb the eigenvalues in filters, we scale each of the three \
        ranked filters by :math:`\\sqrt[3]{\\sigma_{r}}`, the eigenvalue corresponding to rank :math:`r`.

    .. note::
        *Safe Rank*

        - The Poisson kernel is always half-rank, meaning its rank is equal to half of its size along any of its \
        dimensions.

        - We extensively use a *safe rank* to protect the user from asking for invalid desired ranks in 2D. A *safe rank* \
        is the minimum of the Poisson kernel actual maximum possible rank and the desired rank input by the user. The \
        main function to do this is :func:`rank_safety_clamp_2d` in 2D. There is no need for a rank clamp in 3D because \
        of how CP decomposition works.

        - In our functions 'safe' simply means the number of ranked filters we would like to include in our convolutions, \
        and use it to help with setting up loops etc.

        - While in 2D *safe* means a rank that does not exceed the actual rank of the kernel, in 3D it is different. \
        Due to the lack of a clear definition of rank in tensor decomposition, we can pretty much ask for any rank in \
        the CP-vew eigen decomposition and always get something that "*works*". This vagueness of a proper rank definition \
        in 3D is fundamental, which partially contributes to the fact that tensor decomposition is NP-hard.

"""
import numpy as np
import src.helper.commons as com
import src.functions.generator as gen
import src.functions.mathlib as mlib
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.decomposition import SymmetricCP


# ============================== 2D ==============================

def poisson_filters_2d(opts):
    """Generate 2D Poisson filters.

    First generate the kernel then compute the filters using singular value decomposition (*SVD*).
    Check out :func:`get_filters_from_svd_components_2d` to see how we do filter computation.

    .. note::
        To set rank, order, and other relevant parameters you need to pack :code:`OptionsKernel` and
        :code:`OptionsReduction` in :code:`OptionsGeneral` and send it to this function. \
        These dataclasses are in :code:`helper.commons.py`.
        To see how to pack options, look at main demos, or see :func:`helper.commons.generic_options`.

    :param OptionsGeneral opts: parameters bundle
    :return: horizontal and vertical filters
    """
    U, S, VT, low_rank, safe_rank = poisson_components_2d(opts=opts)

    # percentage truncation
    if opts.reduction.truncation_method == com.TruncationMode.PERCENTAGE:
        v_hor, v_ver, safe_rank = compute_separable_filters_trunc_percent_2d(
            U=U, S=S, VT=VT, rank=safe_rank, trunc_factor=opts.reduction.truncation_value)
    # adaptive truncation using a fixed threshold
    elif opts.reduction.truncation_method == com.TruncationMode.FIXED_THRESHOLD:
        v_hor, v_ver, safe_rank = compute_separable_filters_trunc_adaptive_2d(
            U=U, S=S, VT=VT, rank=safe_rank, trunc_threshold=opts.reduction.truncation_value)
    else:
        assert False, "Unknown truncation method"

    return v_hor, v_ver, safe_rank


def poisson_components_2d(opts):
    """ Generate 2d Poisson kernel and decompose it using *SVD* to get :math:`\mathbf{USV^T}`

    .. note::
        To set rank, order, and other relevant parameters you need to pack :code:`OptionsKernel` and
        :code:`OptionsReduction` in :code:`OptionsGeneral` and send it to this function. \
        These dataclasses are in :code:`helper.commons.py`.
        To see how to pack options, look at main demos, or see :func:`helper.commons.generic_options`.

    :param OptionsGeneral opts: parameters bundle
    :return: :math:`\mathbf{U}`, :math:`\mathbf{S}`, and :math:`\mathbf{V^T}` components, \
        as well as the low rank kernel with the same size as the input with the desired rank
    """
    # generate kernel
    poisson_kernel = gen.poisson_kernel_2d(opts=opts)
    safe_rank = rank_safety_clamp_2d(itr=opts.kernel.itr, rank=opts.reduction.rank, zero_init=opts.solver.zero_init)
    # decompose, reduce and reconstruct
    U, S, VT, low_rank = poisson_svd_2d(P=poisson_kernel, rank=safe_rank)

    return U, S, VT, low_rank, safe_rank


def poisson_svd_2d(P, rank=1):
    """ *Singular value decomposition (SVD)* of the 2D Poisson kernel.

    Decompose, reduce and reconstruct a low rank kernel by truncating the singular values based on the given \
    desired rank.

    .. note::
        **Useful links**

        - `Simple reduction <https://www.analyticsvidhya.com/blog/2019/08/5-applications-singular-value-decomposition-svd-data-science/>`_
        - `PCA and SVD <https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8>`_
        - `Intuition <https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.8-Singular-Value-Decomposition/>`_
        - `Truncated SVD using sklearn <https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/>`_

    :param ndarray P: input Poisson kernel matrix
    :param int rank: desired rank. Must be a safe rank, i.e. the minimum of the Poisson kernel actual maximum possible \
        rank and the desired rank input by the user. See **notes** and :func:`rank_safety_clamp_2d`
    :return: :math:`\mathbf{U}`, :math:`\mathbf{S}`, and :math:`\mathbf{V^T}` components in \
        :code:`svd(P)` = :math:`\mathbf{USV^T}`, as well as \
        the low rank kernel with the same size as the input and reduced to have maximum a rank as the \
        desired input rank.
    """
    # Compute full SVD
    U, S, VT = np.linalg.svd(P)
    # reduce
    low_rank = U[:, :rank] @ np.diag(S[:rank]) @ VT[:rank, :]

    return U, S, VT, low_rank


def get_filters_from_svd_components_2d(rank, U, S, VT):
    r""" Compute separable Poisson filters using the *Canonical Polyadic (CP)* view of the kernel matrix.

    We do not need to explicitly perform *Canonical Polyadic Decomposition (CPD)* in 2D. Instead, we can decompose
    the matrix into a summation of rank-1 matrices (as CPD does) by manipulating the components obtained from
    *Singular Value Decomposition (SVD)* of
    the same matrix. This involves reconstructing each rank-1 matrix by multiplying columns of :math:`U` and rows of
    :math:`V^T` scaled by the squared root of the corresponding singular value.

    Given the *SVD* of a square kernel :math:`\mathbf{F}`

    .. math::
        \mathbf{F} \approx \mathbf{USV^T}

    where:
        - The columns of :math:`\mathbf{U}` are the left-singular vectors of :math:`\mathbf{F}`
        - The columns of :math:`\mathbf{V}` are the right-singular vectors of :math:`\mathbf{F}`
        - The values along the diagonal of :math:`\mathbf{S}` are the singular values of :math:`\mathbf{F}`

    The *CP* view is obtained by

    .. math::
        \mathbf{F} \approx \displaystyle\sum_{j=1}^n s_j \mathbf{u}_j \otimes \mathbf{v}_j

    where :math:`\mathbf{u}_j` and :math:`\mathbf{v}_j` are the :math:`j` -th columns of the left- and right-singular
    vectors, :math:`s_j` is the :math:`j` -th singular value, and :math:`\otimes` is the outer product.
    The summation gives :math:`n` rank-1 matrices, each of which meets the rank-1 separability condition to
    compute our filters, where :math:`n` can be the maximum rank of :math:`\mathbf{F}` or less.

    With the above formulation the :math:`j` -th component corresponds to the :math:`r` -th rank in the *CP* view.
    This means the :math:`\mathbf{u}_j` and :math:`\mathbf{v}_j` can be taken as the separable filters needed to
    reconstruct the :math:`r` -th mode :math:`\mathbf{F}_r`, scaled by its singular value :math:`s_r` \
    (which is the same as its eigenvalue, as explained in the next section)

    .. math::
        \mathbf{F_r} = s_r (\mathbf{u_r} \otimes \mathbf{v_r})=s_r \left[\begin{array}{c}
        u_{r_1} \\
        u_{r_2} \\
        \vdots \\
        u_{r_n}
        \end{array}\right] \otimes \left[\begin{array}{c}
        v_{r_1} \\
        v_{r_2} \\
        \vdots \\
        v_{r_n}
        \end{array}\right] = s_r \left[\begin{array}{cccc}
        u_{r_1} v_{r_1} & u_{r_1} v_{r_2} & \cdots & u_{r_1} v_{r_n} \\
        u_{r_2} v_{r_1} & u_{r_2} v_{r_2} & \cdots & u_{r_2} v_{r_n} \\
        \vdots & \vdots & \ddots & \vdots \\
        u_{r_n} v_{r_1} & u_{r_n} v_{r_2} & \cdots & u_{r_n} v_{r_n}
        \end{array}\right]

    where :math:`\mathbf{F_r}` is the :math:`r` -th rank-1 square matrix component of :math:`\mathbf{F}` in
    the *CP* view. As the final step to get the Poisson filters we need to absorb the singular values into the filters.

    **Absorbing singular values into separable filters**

    Because the Poisson kernel is always square with non-negative eigenvalues, the singular values \
    obtained from *SVD* coincide with the eigenvalues. This is not true in general, and is only true in our case \
    given the aforementioned properties of the kernel. To absorb the eigenvalues in filters, we scale each of the \
    two ranked filters by :math:`\sqrt{\sigma_{r}}`, where :math:`\sigma_r = s_j = s_r`, \
    the :math:`r` -th singular value.

    The :math:`r` -th vertical :math:`f_v` and horizontal :math:`f_h` Poisson filters reconstructing \
    the :math:`r` -th mode are

    .. math::
        \mathbf{f_{v_r}}=\sqrt{\sigma_{r}} \left[\begin{array}{c}
        u_{r_1} \\
        u_{r_2} \\
        \vdots \\
        u_{r_n}
        \end{array}\right], \; \; \; \mathbf{f_{h_r}}=\sqrt{\sigma_{r}} \left[\begin{array}{c}
        v_{r_1} \\
        v_{r_2} \\
        \vdots \\
        v_{r_n}
        \end{array}\right]

    which are then used to compute :math:`\mathbf{F_r} = \mathbf{f_{v_r}} \otimes \mathbf{f_{h_r}}`



    .. note::
        In theory, the horizontal and vertical Poisson filters are the same due to the symmetry in the full Poisson \
        kernel. However, and in practice, *SVD* might give singular vectors with opposing signs.
        To avoid any problem, we save and apply each of the filters separately.

        This is not a problem in 3D because of the way *Sym-CP* does the decomposition.

    :param int rank: desired rank. Does not have to be a safe rank
    :param ndarray U: :math:`\mathbf{U}` in :math:`\mathbf{USV^T}`
    :param ndarray S: :math:`\mathbf{S}` in :math:`\mathbf{USV^T}`
    :param ndarray VT: :math:`\mathbf{V^T}` in :math:`\mathbf{USV^T}`
    :return: horizontal and vertical Poisson filters
    """
    # rank can't be higher than the size of S
    safe_rank = min(rank, S.shape[0])

    # Separated Vectors (horizontal filters/row vectors in VT) with absorbed singular values
    v_hor = np.sqrt(np.diag(S[:safe_rank])) @ VT[:safe_rank, :]

    # Separated Vectors (vertical filters/column vectors in U) with absorbed singular values
    v_ver = U[:, :safe_rank] @ np.sqrt(np.diag(S[:safe_rank]))

    return v_hor, v_ver


def compute_separable_filters_truncated_2d(U, S, VT, rank, trunc_method, trunc_factor, preserve_shape):
    """ Compute and truncate 2D Poisson filters.

    Truncation is either based on cutting off a certain percentage of the filter, or using a fixed cut-off \
    threshold (*adaptive truncation*).

    :param ndarray U: :math:`\mathbf{U}` in :math:`\mathbf{USV^T}`
    :param ndarray S: :math:`\mathbf{S}` in :math:`\mathbf{USV^T}`
    :param ndarray VT: :math:`\mathbf{V^T}` in :math:`\mathbf{USV^T}`
    :param int rank: desired rank. Does not have to be a safe rank
    :param TruncationMode trunc_method: either :code:`PERCENTAGE` or :code:`FIXED_THRESHOLD`
    :param float trunc_factor: percentage in [0, 1] if :code:`PERCENTAGE`,
        fixed cut-off threshold if :code:`FIXED_THRESHOLD`
    :param bool preserve_shape: if :code:`True` keep the original shape and fill them with zeros,
        else return the shrunk filter
    :return: stacked array of truncated horizontal and vertical separable filters for :math:`n` ranks, in the form of
        a 2-tuple (rank, 1d filter). The clamped safe rank is also returned.
    """
    # getting horizontal and vertical filters
    if trunc_method == com.TruncationMode.PERCENTAGE:
        v_hor, v_ver, safe_rank = compute_separable_filters_trunc_percent_2d(U=U, S=S, VT=VT, rank=rank,
                                                                             trunc_factor=trunc_factor)

    elif trunc_method == com.TruncationMode.FIXED_THRESHOLD:  # adaptive truncation
        v_hor, v_ver, safe_rank = compute_separable_filters_trunc_adaptive_2d(U=U, S=S, VT=VT, rank=rank,
                                                                              trunc_threshold=trunc_factor,
                                                                              preserve_shape=preserve_shape)
    else:
        assert False, "Unknown truncation method"

    return v_hor, v_ver, safe_rank


def compute_separable_filters_trunc_percent_2d(U, S, VT, rank, trunc_factor):
    """ Compute the Poisson filters from *SVD* components and truncate a certain percentage of them.

    Since filters are symmetrical, the truncation is applied to both sides of the filters.

    :param ndarray U: :math:`\mathbf{U}` in :math:`\mathbf{USV^T}`
    :param ndarray S: :math:`\mathbf{S}` in :math:`\mathbf{USV^T}`
    :param ndarray VT: :math:`\mathbf{V^T}` in :math:`\mathbf{USV^T}`
    :param int rank: desired rank. It will be safely clamp if larger than the input matrix actual rank.
    :param float trunc_factor: truncation percentage in [0, 1]
    :return: stacked array of truncated horizontal and vertical separable filters for :math:`n` ranks, in the form of
        a 2-tuple (rank, 1d filter). The clamped safe rank is also returned.
    """

    # rank can't be higher than the size of S
    safe_rank = min(rank, S.shape[0])
    # extract filters
    v_hor, v_ver = get_filters_from_svd_components_2d(rank=rank, U=U, S=S, VT=VT)

    # --- NOTE:
    # to directly construct the matrix from absorbed matrices you should follow ths ordeR:
    # low_rank = Ver @ Hor
    # where Ver and Hor are matrices with absorbed singular values

    v_hor = mlib.truncate_percent_filter_1d(arr=v_hor, trunc_percent=trunc_factor)
    # vertical components transpose: [filter, rank] -> [rank, filter]
    v_ver = mlib.truncate_percent_filter_1d(arr=np.transpose(v_ver), trunc_percent=trunc_factor)

    return v_hor, v_ver, safe_rank


def compute_separable_filters_trunc_adaptive_2d(U, S, VT, rank, trunc_threshold, preserve_shape=True):
    """Compute the Poisson filters from *SVD* components and truncate them based on a fixed cut-off threshold.

    .. note::
        It does not guarantee the returned filters have the same size due to adaptive truncation
        unless :code:`preserve_shape=True` (Default).

    :param ndarray U: :math:`\mathbf{U}` in :math:`\mathbf{USV^T}`
    :param ndarray S: :math:`\mathbf{S}` in :math:`\mathbf{USV^T}`
    :param ndarray VT: :math:`\mathbf{V^T}` in :math:`\mathbf{USV^T}`
    :param int rank: desired rank. It will be safely clamp if larger than the input matrix actual rank.
    :param trunc_threshold: truncation threshold (absolute value)
    :param preserve_shape: if :code:`True` keep the original shape and fill them with zeros (Default= :code:`True`)
    :return: stacked array of truncated horizontal and vertical separable filters for :math:`n` ranks, in the form of
        a 2-tuple (rank, 1d filter). The clamped safe rank is also returned.
    """
    # rank can't be higher than the size of S
    safe_rank = min(rank, S.shape[0])
    # extract filters
    v_hor, v_ver = get_filters_from_svd_components_2d(rank=rank, U=U, S=S, VT=VT)
    # transpose: [filter, rank] -> [rank, filter]
    v_ver = np.transpose(v_ver)
    # truncation
    v_hor = mlib.apply_adaptive_truncation_1d(array_1d=v_hor, safe_rank=safe_rank,
                                              cut_off=trunc_threshold, preserve_shape=preserve_shape)

    v_ver = mlib.apply_adaptive_truncation_1d(array_1d=v_ver, safe_rank=safe_rank,
                                              cut_off=trunc_threshold, preserve_shape=preserve_shape)

    return v_hor, v_ver, safe_rank


def compute_nth_kernel_mode_2d(hor, ver, rank):
    """Reconstruct the rank-1 matrix corresponding to the :math:`r` -th rank (mode) from horizontal and vertical
    Poisson filters for a desired rank.

    .. warning::
        Singular values must be already absorbed into the filters.

    :param ndarray hor: 2d array: horizontal filters as 2-tuple (rank, filter)
    :param ndarray ver: 2d array: vertical filters as 2-tuple (rank, filter)
    :param int rank: desired safe rank. rank needs to be equal or less than the original decomposed matrix.
    :return: :math:`r` -th mode square matrix constructed from the :math:`r` -th pair of filters
    """
    rank_index = rank - 1
    return mlib.outer_product_2d(ver[rank_index], hor[rank_index])


def compute_low_rank_kernel_from_filters_2d(hor, ver, safe_rank):
    """ Compute a low rank kernel from the given Poisson filters. Filters can be in the original form or
    can be truncated.

    :param ndarray hor: 2d array: horizontal filters as 2-tuple (rank, filter)
    :param ndarray ver: 2d array: vertical filters as 2-tuple (rank, filter)
    :param int safe_rank: cumulative rank, i.e. maximum rank to be included. must be safe.
    :return: square low rank kernel
    """

    low_rank_kernel = []
    for r in range(1, safe_rank + 1):
        nth_mode = compute_nth_kernel_mode_2d(hor=hor, ver=ver, rank=r)
        if r == 1:
            low_rank_kernel = nth_mode
        else:
            low_rank_kernel = nth_mode + low_rank_kernel

    return low_rank_kernel


def get_max_rank_2d(itr, zero_init):
    """ Get the maximum possible rank of a Poisson kernel with the given target Jacobi iteration.

    :param int itr: target Jacobi iteration
    :param bool zero_init: if we are zero-initializing the Jacobi solution, in which case the corresponding Poisson
        kernel will have a smaller size. Typical value is :code:`True` because we usually start from :math:`x=0`,
        i.e. no warm starting.
    :return: maximum possible rank
    """
    return int(gen.get_kernel_size(itr=itr, zero_init=zero_init) / 2) + 1


def rank_safety_clamp_2d(itr, rank, zero_init):
    """
    Clamp the given desired rank based on the maximum possible rank of a
    Poisson kernel with the given target Jacobi iteration.

    The Poisson kernel is always half-rank, meaning its rank is equal to half of its size along any of its dimensions.
    A safe rank is the minimum of the Poisson kernel actual maximum possible rank and the desired rank.

    :param int itr: target Jacobi iteration
    :param int rank: desired rank
    :param bool zero_init: if we are zero-initializing the Jacobi solution, in which case the corresponding Poisson
        kernel will have a smaller size. Typical value is :code:`True` because we usually start from :math:`x=0`,
        i.e. no warm starting.
    :return: safe rank
    """
    assert rank > 0, "Rank must be positive."
    return min(rank, get_max_rank_2d(itr=itr, zero_init=zero_init))


# ============================== 3D ==============================

def poisson_filters_3d(opts, rank_filter_reorder, preserve_shape=True):
    """Generate 3D Poisson filters (with truncation if desired).

    First generate the kernel then compute the filters using Eigenvalue Decomposition of a 3D kernel tensor.
    See :func:`poisson_decomposition_components_3d` for decomposition methods.

    .. note::
        Check out :func:`get_filters_from_svd_components_2d` to learn about the basics of Poisson filter \
        computation in 2D.
        For 3D filters apply the same principles except for an extra filter in 3D dimension. The outer product
        :math:`\\otimes` in 2D becomes tensor product in 3D.

    .. note::
        To set rank, order, and other relevant parameters you need to pack :code:`OptionsKernel` and
        :code:`OptionsReduction` in :code:`OptionsGeneral` and send it to this function. \
        These dataclasses are in :code:`helper.commons.py`.
        To see how to pack options, look at main demos, or see :func:`helper.commons.generic_options`.

    :param OptionsGeneral opts: parameters bundle
    :param bool rank_filter_reorder: if :code:`True` change 2-tuple order from default (filters, ranks) to
        (ranks, filters), which is more consistent with 2D filters. Typical value is :code:`True`.
    :param preserve_shape: if :code:`True` keep the original shape and fill them with zeros,
        else return the shrunk filter (Default= :code:`True`)
    :return:
        - 3 Poisson filters for horizontal, vertical and depth (fiber) passes. The filters can be already truncated if \
            truncation parameters are set in :code:`opts`
        - generated Poisson kernel tensor
        - reduced (low rank) reconstructed Poisson kernel corresponding to the desired rank specified in :code:`opts`
        - safe rank
    """

    # compute filters
    low_rank, cores, factors, full_kernel = poisson_decomposition_components_3d(opts=opts)
    filters = absorb_cores_3d(cores=cores, factors=factors)

    # Reorder [rank, filter] True: change from default (filters, ranks) to (ranks, filters): consistent with 2d
    filters = filters.transpose() if rank_filter_reorder else filters

    assert rank_filter_reorder, "The following truncate filters function needs [rank, filter] order"
    # rank can't be higher than the size of S
    safe_rank = min(opts.reduction.rank, filters.shape[0])

    # truncate filters, either by percentage or a fixed threshold
    filters = mlib.truncate_filters(truncation_method=opts.reduction.truncation_method,
                                    truncation_value=opts.reduction.truncation_value,
                                    safe_rank=safe_rank,
                                    filters_1d=filters,
                                    preserve_shape=preserve_shape)

    return filters, low_rank, full_kernel, safe_rank


def absorb_cores_3d(cores, factors):
    """ Absorb the core weights (eigenvalues) into filters (eigenvectors). Scale each filter by the cubic root of
    the corresponding core weight to get the Poisson filters.

    .. warning::
        - The absorption of eigenvalues only works when doing the tensor decomposition using *Symmetric CP*.
        - Without this step we do not really have Poisson filters, only eigenvectors.
        - This is based on (filter, rank) order, which is the opposite of 2D filter order.

    .. note::
        *Symmetric-CP* decomposition of 3D Poisson kernel gives a set of factors and cores, where cores can \
        be taken as the best approximation of true tensor eigenvalues (this should always be taken with a grain of  \
        salt for tensor eigen decomposition as there is no consensus on rank definition in 3D and the tensor eigen \
        decomposition is *NP-hard*). To absorb the eigenvalues in filters, we scale each of the three \
        ranked filters by :math:`\\sqrt[3]{\\sigma_{r}}`, the eigenvalue corresponding to rank :math:`r`.

    :param ndarray cores: eigenvalues
    :param ndarray factors: eigenv ectors
    :return: 3D Poisson filters
    """

    return factors * np.cbrt(cores.reshape((1, np.shape(cores)[0])))  # using original order from Sym-CP


def poisson_compute_modes_trim_filters_3d(opts, rank_filter_reorder, filter_trim_zeros=False, preserve_shape=True):
    """ Compute 3D ranked modes.

    :param OptionsGeneral opts: parameters bundle
    :param bool rank_filter_reorder: if :code:`True` change 2-tuple order from default (filters, ranks) to
        (ranks, filters), which is more consistent with 2D filters. Typical value is :code:`True`.
    :param bool filter_trim_zeros: if :code:`True` do not preserve shape, cut the zeros on each side of the filter \
        (Default= :code:`False`)
    :param bool preserve_shape: if :code:`True` keep the original shape and fill them with zeros,
        else return the shrunk filter (Default= :code:`True`)
    :return: 3D ranked modes, Poisson filters, a low rank reconstructed kernel along with the full kernel
    """
    filters, low_rank, full_kernel, safe_rank = poisson_filters_3d(opts=opts,
                                                                   rank_filter_reorder=rank_filter_reorder,
                                                                   preserve_shape=preserve_shape)

    assert rank_filter_reorder, "The following filter truncation function needs [rank, filter] order"

    all_modes = []
    all_filters = []
    for r in range(0, safe_rank):
        ranked_filter = filters[r - 1]
        # mode
        nth_mode = compute_nth_kernel_mode_3d(ranked_filter=ranked_filter)
        all_modes.append(nth_mode)
        # filter
        ranked_filter = ranked_filter[np.nonzero(ranked_filter)] if filter_trim_zeros else ranked_filter
        if np.size(ranked_filter) == 0:
            ranked_filter = np.zeros(5)  # fixed number just to make the interpolation work, "if" we are interpolating
        all_filters.append(ranked_filter)  # assuming horizontal and vertical filters are identical

    return all_filters, all_modes, low_rank, full_kernel, safe_rank


def poisson_decomposition_components_3d(opts):
    """
    Compute 3D cores, factors, low rank kernel and full kernel.

    .. note::
        We use *Symmetric CP* as the main decomposition method. *Tucker* decomposition is also available \
        for experimentation.

    :param OptionsGeneral opts: parameters bundle
    :return: reduced low rank kernel, cores and factors after decomposition, as well as the full (i.e. unreduced) kernel
    """
    decomp_method = opts.reduction.decomp_method

    full_kernel = gen.poisson_kernel_3d(opts=opts)

    if decomp_method == com.DecompMethod.TUCKER_3D:
        # Tucker Decomposition - an experimental approach
        # reduce kernel
        low_rank, cores, factors = poisson_tucker_3d(kernel=full_kernel, rank=opts.reduction.rank)

    elif decomp_method == com.DecompMethod.SYM_CP_3D:
        # Symmetrical CP - This is the main method in the paper
        # reduce kernel
        low_rank, cores, factors = poisson_symcp_3d(kernel=full_kernel, rank=opts.reduction.rank)

    else:
        assert False, "Unknown decomposition method"

    return low_rank, cores, factors, full_kernel


def poisson_tucker_3d(kernel, rank):
    """Decompose a full 3D Poisson functions using tensor Tucker decomposition.

    :param int rank: target decomposition rank
    :param ndarray kernel: full 3D Poisson kernel
    :return: reduced (lowe rank) kernel and decomposition cores and factors
    """

    core_tucker, factors_tucker = tucker(kernel, rank=[rank, rank, rank])
    low_rank_tucker = tl.tucker_to_tensor((core_tucker, factors_tucker))

    return low_rank_tucker, core_tucker, factors_tucker


def poisson_symcp_3d(kernel, rank):
    """Decompose a full 3D Poisson kernel using symmetric CP (Canonical Polyadic) decomposition.

    .. warning::
        For a kernel of shape (1, 1, 1) corresponding to a target iteration of 1, Sym-CP does not work. \
        To avoid bugs or crashes, we return a set of zero :math:`1 \\times 1` vectors (as many as target rank), \
        with the first factor being the only non-zero vector and with the value equal to the only scalar value we have \
        from the (1, 1, 1) kernel. Cores will be all ones. The reduced kernel will be the same as the original full \
        kernel (there is no reduction).

    :param int rank: target decomposition rank
    :param ndarray kernel: full 3D Poisson kernel
    :return: reduced (lowe rank) kernel and decomposition cores and factors
    """
    # ----Special treatment for itr=1, kernel has only 1 scalar value (basically no reduction):
    # 1. factors are (1, rank), where only rank 1 filter is nonzero, which is the kernel value
    # 2. cores would be all 1
    # low rank will be the same as the full rank kernel
    if kernel.shape == (1, 1, 1):
        # NOTE: for kernel of shape(1, 1, 1) SYmCP does not work. We can just a
        factors_sym_cp = np.zeros(shape=(1, rank))  # native SymCP shape (filter, rank)
        factors_sym_cp[0, 0] = kernel[0, 0, 0] # getting the kernel only element, a signle scalar
        core_sym_cp = np.ones(shape=rank)
        low_rank_sym_cp = kernel
        return low_rank_sym_cp, core_sym_cp, factors_sym_cp

    rank = int(rank)
    n_repeat = 10  # default 10
    n_power_iteration = 10  # default 10
    verbose = True

    # ------- NOTE: for kernel of shape(1, 1, 1) this does not work.
    sym_cp = SymmetricCP(rank=rank, n_repeat=n_repeat, n_iteration=n_power_iteration, verbose=verbose)
    decomp = sym_cp.fit_transform(kernel)

    core_sym_cp = decomp[0]
    factors_sym_cp = decomp[1]

    # 1st rank
    s1 = core_sym_cp[0]
    r1 = factors_sym_cp[:, 0]
    # Get the outer product to construct the tensor
    # Make sure the shape of the vector is consistent with the
    # einsum notation:
    # if (1,n) use ai, aj, ak
    # if (n, 1) use i1, ja, ka
    r1 = r1.reshape((1, r1.shape[0]))
    low_rank_sym_cp = s1 * np.einsum('ai,aj,ak->ijk', r1, r1, r1)

    if rank > 1:
        for rr in range(1, rank):
            s = core_sym_cp[rr]
            r = factors_sym_cp[:, rr]
            r = r.reshape((1, r.shape[0]))
            low_rank_sym_cp += s * np.einsum('ai,aj,ak->ijk', r, r, r)

    return low_rank_sym_cp, core_sym_cp, factors_sym_cp


def compute_nth_kernel_mode_3d(ranked_filter):
    """Reconstruct the rank-1 tensor corresponding to the :math:`r` -th rank (mode) from the Poisson filter of a \
    certain rank.

    .. note::
        All 3 dimensions use the same filter.

    .. warning::
        Core values (eigenvalues) must be already absorbed into the filters.

    :param ndarray ranked_filter: single Poisson filter, 1d array
    :return: :math:`r` -th mode tensor constructed from the :math:`r` -th ranked filter
    """
    arr = np.copy(ranked_filter).reshape((1, ranked_filter.shape[0]))
    return np.einsum('ai,aj,ak->ijk', arr, arr, arr)


def demo_test_svd_absorb():
    """*Sanity check*: Testing the eigenvalue absorption into the separable filters in 2D.

    We expect the low-rank matrix obtained from the reduced \
    space exactly matching the reconstruction from the outer product of horizontal and vertical filters. Here by \
    *"matching exactly"* we mean machine precision.

    Because the Poisson kernel is always square with non-negative eigenvalues, the singular values \
    obtained from SVD coincide with the eigenvalues. This is not true in general, and is only true in our case \
    given the special properties of the Poisson kernel, like symmetry.

    To absorb the eigenvalues in filters, we scale each of the two ranked filters by\
     :math:`\\sqrt{\\sigma_{r}}`, the eigenvalue corresponding to rank :math:`r`.

    See :func:`get_filters_from_svd_components_2d` for more details.
    """
    # generate a sample 2D Poisson kernel - you can also use a full rank random matrix P = np.random.rand(rank, rank)
    opts = com.generic_options()  # with generic parameter values
    rank = opts.reduction.rank
    P = gen.poisson_kernel_2d(opts=opts)

    # Compute full SVD of P => USV^T
    U, S, VT = np.linalg.svd(P)
    # reduce
    low_rank = U[:, :rank] @ np.diag(S[:rank]) @ VT[:rank, :]

    # to expand S you can also use scipy..
    # import scipy.linalg as la
    # B = U @ la.diagsvd(S, *P.shape) @ VT

    # Separated vector (horizontal/top vectors) - rows of VT scaled by the squared root of singular values
    Hor = np.sqrt(np.diag(S[:rank])) @ VT[:rank, :]

    # Separated vector (vertical/left vectors) - columns of U scaled by the squared root of singular values
    Ver = U[:, :rank] @ np.sqrt(np.diag(S[:rank]))

    low_rank_test = Ver @ Hor

    print("Difference between the two methods: ")
    diff = low_rank_test - low_rank
    print(diff)
    print(f'max absolute difference = {np.max(np.abs(diff))} ')


if __name__ == '__main__':
    # testing absorbing the eigenvalues into 2D filters
    demo_test_svd_absorb()
