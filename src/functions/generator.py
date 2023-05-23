""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`

    .. _generator_method:

    Technicals
    ===============
    We use a recursive function to generate the Poisson kernel. Due to similarities between sub-problems, we enhance \
    the recursion performance by *memoization*, i.e. saving the solution and reusing it multiple times. \
    See :func:`poisson_kernel_2d` and :func:`poisson_kernel_3d`.



    Notes
    ===============

    Choosing the kernel type :code:`UNIFIED` vs :code:`STANDARD`:

        UNIFIED
            The :code:`UNIFIED` type produces the exact same parametric kernel for both :code:`INVERSE` and :code:`FORWARD`
            Poisson equations. While for the :code:`FORWARD` version the kernel already supports warm starting,
            for the :code:`INVERSE`  version it only works with a zero initial guess. Also, we will need to negate
            the input to make it work for :code:`INVERSE` applications.

            - *Upside*: Same kernel generator for both :code:`INVERSE`  and :code:`FORWARD`. No pre-assumption about \
                the type of the Poisson equation we are solving.
            - *Downside*: for :code:`INVERSE` Poisson (e.g. poisson pressure) you would need to negate the input \
                divergence before feeding it to the solver.


        STANDARD (*Safer*)
            The :code:`STANDARD` type has an alternating sign for :math:`\\alpha` in the parametric kernel, \
            so the two kernels
            for :code:`FORWARD` and :code:`INVERSE`  are not exactly the same, but it supports warm starting for both \
            :code:`FORWARD` and particularly :code:`INVERSE`. An example of warm starting is when if you want \
            to initialize the first guess of the pressure in Poisson-pressure solve from *RHS* (divergence).

            - *Upside*: Supports warm starting (but within the constraints of Poisson filters applications - \
            see paper). Less confusion in applying filters and convergence analysis.
            - *Downside*: Needs a separate call to each :code:`FORWARD` and :code:`INVERSE` Poisson kernel generators, \
            which is not a major inconvenience.

            :code:`STANDARD` is the recommended default type.

    Application Notes:
        - In addition to the target Jacobi iteration, :math:`\\alpha` and :math:`\\beta` are the two major players \
        when generating the Poisson kernel. Things like the unit cell size :math:`dx`, time step :math:`dt` and \
        diffusivity factor :math:`\\kappa` (where :math:`dt` and :math:`\\kappa` are only used by the :code:`FORWARD` \
        Poisson kernel), are captured by :math:`\\alpha` and :math:`\\beta`. \
        Check out :func:`compute_alpha` and :func:`compute_beta` to see how they are set. \
        In practice, you would only need to provide \
        :math:`dx`, :math:`dt` and :math:`\\kappa`, and the generator automatically computes :math:`\\alpha` \
        and :math:`\\beta` based on  the problem dimension and the Poisson equation type.

        - :math:`\\alpha` has a negative sign in the original Jacobi equation for pressure. To make \
        the kernel generation consistent for both :code:`FORWARD` and :code:`INVERSE` we treat :math:`\\alpha` \
        as positive for both cases. So for example to solve Poisson-pressure (inverse Poisson) you should multiply \
        the *rhs* divergence by :math:`-1` to account for the sign change.

        - **Warm starting**: often found as :code:`zero_init` parameter in some functions, Poisson filters have limited \
        support for warm starting the linear solve (as discussed in the paper). See :func:`i_want_to_warm_start` and \
        :func:`is_ok_to_zero_init`.

    .. tip::

        If you decide to change/improve things:

        - You can generate Poisson kernels with :math:`\\alpha=1` then just scale :math:`B` in :math:`L*X = B` \
        (the input matrix) with the actual :math:`\\alpha` you are interested in. \
        This means if doing only :code:`INVERSE` you can factorize :math:`\\alpha` from the kernel generation and change it in \
        real time in your application (multiplying filter convolution results by :math:`\\alpha`). \
        However, in the :code:`FORWARD` case, and due to the presence of :math:`\\alpha` in the way :math:`\\beta` \
        is computed, we can't do this factorization.

        - In the current implementation, we decided to explicitly include \
        :math:`\\alpha` in the kernel to have a uniform formulation that works for both :code:`FORWARD` \
        and :code:`INVERSE`.
"""


import numpy as np
import src.helper.commons as com
import sys
sys.path.insert(0, './')

# Memoization for recursions.
# MEMO is global. Use this to clear the leftover of computing a different kernel.
# You need to do this if alpha or beta is different from previous computations.
# For a fixed alpha beta value you don't need to clear the MEMO, which helps
# with even faster (batch) kernel computations.
MEMO = {}


# ======================================= 2D =======================================

def poisson_kernel_2d(opts):
    """ Compute a 2D Poisson kernel based on recursive Jacobi.

    Recursion is used to generate 2D analytical Poisson kernels based on 4 nearest neighbors only.
    Precomputing the Jacobi solution to :math:`Ax=b` in the *matrix-form* :math:`L*X = B` using the recursion formula \
    based on the Jacobi update step:

    :math:`x_{here} = (x_{left} + x_{right} + x_{down} + x_{up} + \\alpha * b_{here}) / \\beta`

    where :math:`x_{here}` and :math:`b_{here}` are the elements of :math:`X` and :math:`B` matrices that we would \
    like to update and evaluate, respectively, and :math:`\\alpha` and :math:`\\beta` are set based on the type of \
    the Poisson equation. \
    See :func:`compute_alpha` and :func:`compute_beta`.

    .. note::
        To set the kernel parameters like target iteration (*order*), :math:`dx`, and others, you need to pack
        :code:`OptionsKernel` and :code:`OptionsPoissonSolver` in :code:`OptionsGeneral` and send it to this function. \
        These dataclasses are in :code:`helper.commons.py`.
        To see how to pack options, look at the main demos, or see :func:`helper.commons.generic_options`.

    :param OptionsGeneral opts: parameters bundle
    :return: full Poisson kernel
    """

    if opts.kernel.kernel_type == com.PoissonKernelType.UNIFIED:
        return poisson_kernel_unified_2d(solver_type=opts.solver.solver_type, itr=opts.kernel.itr,
                                         alpha=opts.kernel.alpha, zero_init=opts.solver.zero_init,
                                         clear_memo=opts.kernel.clear_memo)

    elif opts.kernel.kernel_type == com.PoissonKernelType.STANDARD:
        return poisson_kernel_standard_2d(solver_type=opts.solver.solver_type, itr=opts.kernel.itr,
                                          alpha=opts.kernel.alpha, zero_init=opts.solver.zero_init,
                                          clear_memo=opts.kernel.clear_memo)

    else:
        assert False, 'Invalid Kernel Type'


def poisson_kernel_unified_2d(solver_type, itr, alpha, zero_init, clear_memo=True):
    """ Compute a 2D Poisson *Unified* kernel based on recursive Jacobi.
    See :func:`poisson_kernel_2d` for details.

    :param PoissonSolverType solver_type: inverse, forward
    :param int itr: target Jacobi iteration (*kernel order*)
    :param float alpha: see :func:`compute_alpha()`
    :param bool zero_init: zero as the initial guess. If :code:`True`, we produce a smaller kernel, else
        we are warm starting the Jacobi solver with a non-zero matrix, which makes the kernel slightly \
        larger (+2 along each dimension).
        We can use zero initial guess for Poisson pressure only (inverse Poisson).
        Diffusion step (forward Poisson) has to warm start with a non-zero initial guess.
    :param bool clear_memo: better set to be :code:`True` for recursion to avoid garbage cashing. Default= :code:`True`.
    :return: full Jacobi Poisson kernel
    """

    global MEMO

    if clear_memo:
        MEMO = {}
    return poisson_kernel_unified_2d_rec_memo(solver_type=solver_type, itr=itr, alpha=alpha, zero_init=zero_init)


def poisson_kernel_unified_2d_rec_memo(solver_type, itr, alpha, zero_init):
    """ Compute a 2D Poisson *Unified* kernel based on recursive Jacobi, using memoization, i.e. speed-up by \
    reusing sub-solutions.
    See :func:`poisson_kernel_2d` for details.

    :param PoissonSolverType solver_type: inverse, forward
    :param int itr: target Jacobi iteration (*kernel order*)
    :param float alpha: see :func:`compute_alpha()`
    :param bool zero_init: zero as the initial guess. If :code:`True`, we produce a smaller kernel, else
        we are warm starting the Jacobi solver with a non-zero matrix, which makes the kernel slightly \
        larger (+2 along each dimension).
        We can use zero initial guess for Poisson pressure only (inverse Poisson).
        Diffusion step (forward Poisson) has to warm start with a non-zero initial guess.
    :return: full Jacobi Poisson kernel
    """

    beta = compute_beta(alpha=alpha, solver_type=solver_type, dim=com.SolverDimension.D2)
    beta_inv = 1.0 / beta

    # Determine alpha and beta based on the solver type
    if zero_init:
        # If zero initial guess, we have a smaller base case
        # can use zero initial guess for Poisson pressure only
        offset = 0  # for n+offset in kernel size

        # define base kernel
        k_base = np.zeros((1, 1))
        k_base[0][0] = alpha / beta

    else:
        # If warm starting the initial guess, we have a larger base case
        # Diffusion has to use warm starting. No zero initial guess.
        offset = 1  # for n+offset in kernel size

        # define base kernel
        k_base = np.zeros((3, 3))
        k_base[1][1] = alpha / beta  # center
        k_base[0][1] = beta_inv  # up
        k_base[1][0] = beta_inv  # left
        k_base[1][2] = beta_inv  # right
        k_base[2][1] = beta_inv  # down

    # Base Case
    if itr == 1:
        key = 'K' + str(itr)
        if key not in MEMO:
            k = k_base
            MEMO[key] = k
            return k
        else:
            return MEMO[key]

    else:
        # Sub-problem case
        key = 'K' + str(itr)
        if key not in MEMO:
            dim = (2 * (itr + offset)) - 1
            k = np.zeros((dim, dim))
            k_block = poisson_kernel_unified_2d_rec_memo(solver_type=solver_type, itr=itr - 1, alpha=alpha, zero_init=zero_init)

            def add_block(x_center, y_center, block, K_full, halfSize):
                K_full[x_center - halfSize: x_center + halfSize + 1,
                y_center - halfSize: y_center + halfSize + 1] += block

            # Accumulating the smaller kernels:
            # K[:, :, :] = (PL + PR + PD + PT + alpha * dC) / beta
            # central cell index
            cx = int(dim / 2)
            cy = int(dim / 2)

            add_block(x_center=cx, y_center=cy - 1, block=beta_inv * k_block, K_full=k, halfSize=itr + offset - 2)  # LEFT: -X axis
            add_block(x_center=cx, y_center=cy + 1, block=beta_inv * k_block, K_full=k, halfSize=itr + offset - 2)  # RIGHT:+X axis
            add_block(x_center=cx - 1, y_center=cy, block=beta_inv * k_block, K_full=k, halfSize=itr + offset - 2)  # Up: +Y axis
            add_block(x_center=cx + 1, y_center=cy, block=beta_inv * k_block, K_full=k, halfSize=itr + offset - 2)  # DOWN: -Y axis

            k[cx][cy] += alpha / beta

            MEMO[key] = k
            return k
        else:
            return MEMO[key]


def poisson_kernel_standard_2d(solver_type, itr, alpha, zero_init, clear_memo=True):
    """ Compute a 2D Poisson *Standard* kernel based on recursive Jacobi.
    See :func:`poisson_kernel_2d` for details.

    :param PoissonSolverType solver_type: inverse, forward
    :param int itr: target Jacobi iteration (*kernel order*)
    :param float alpha: see :func:`compute_alpha()`
    :param bool zero_init: zero as the initial guess. If :code:`True`, we produce a smaller kernel, else
        we are warm starting the Jacobi solver with a non-zero matrix, which makes the kernel slightly \
        larger (+2 along each dimension).
        We can use zero initial guess for Poisson pressure only (inverse Poisson).
        Diffusion step (forward Poisson) has to warm start with a non-zero initial guess.
    :param bool clear_memo: better set to be :code:`True` for recursion to avoid garbage cashing. Default= :code:`True`.
    :return: full Jacobi Poisson kernel
    """
    global MEMO

    if clear_memo:
        MEMO = {}
    return poisson_kernel_standard_2d_rec_memo(solver_type=solver_type, itr=itr, alpha=alpha, zero_init=zero_init)


def poisson_kernel_standard_2d_rec_memo(solver_type, itr, alpha, zero_init):
    """ Compute a 2D Poisson *Standard* kernel based on recursive Jacobi using memoization, i.e. speed-up by \
    reusing sub-solutions.
    See :func:`poisson_kernel_2d` for details.

    :param PoissonSolverType solver_type: inverse, forward
    :param int itr: target Jacobi iteration (*kernel order*)
    :param float alpha: see :func:`compute_alpha()`
    :param bool zero_init: zero as the initial guess. If :code:`True`, we produce a smaller kernel, else
        we are warm starting the Jacobi solver with a non-zero matrix, which makes the kernel slightly \
        larger (+2 along each dimension).
        We can use zero initial guess for Poisson pressure only (inverse Poisson).
        Diffusion step (forward Poisson) has to warm start with a non-zero initial guess.
    :return: full Jacobi Poisson kernel
    """

    beta = compute_beta(alpha=alpha, solver_type=solver_type, dim=com.SolverDimension.D2)
    beta_inv = 1.0 / beta

    # this sign is the main difference between normal and warm start kernel types
    sign = get_alpha_sign(solver_type=solver_type, kernel_type=com.PoissonKernelType.STANDARD)

    # Determine alpha and beta based on the solver type
    if zero_init:
        # If zero initial guess, we have a smaller base case
        # can use zero initial guess for Poisson pressure only
        offset = 0  # for n+offset in kernel size

        # define base kernel
        k_base = np.zeros((1, 1))
        k_base[0][0] = sign * alpha / beta

    else:
        # If warm starting the initial guess, we have a larger base case
        # Diffusion has to use warm starting. No zero initial guess.
        offset = 1  # for n+offset in kernel size

        # define base kernel
        k_base = np.zeros((3, 3))
        k_base[1][1] = sign * alpha / beta  # center
        k_base[0][1] = beta_inv  # up
        k_base[1][0] = beta_inv  # left
        k_base[1][2] = beta_inv  # right
        k_base[2][1] = beta_inv  # down

    # Base Case
    if itr == 1:
        key = 'K' + str(itr)
        if key not in MEMO:
            k = k_base
            MEMO[key] = k
            return k
        else:
            return MEMO[key]

    else:
        # Sub-problem case
        key = 'K' + str(itr)
        if key not in MEMO:
            dim = (2 * (itr + offset)) - 1
            k = np.zeros((dim, dim))
            k_block = poisson_kernel_standard_2d_rec_memo(solver_type=solver_type, itr=itr - 1, alpha=alpha, zero_init=zero_init)

            def add_block(x_center, y_center, block, K_full, halfSize):
                K_full[x_center - halfSize: x_center + halfSize + 1,
                y_center - halfSize: y_center + halfSize + 1] += block

            # Accumulating the smaller kernels:
            # K[:, :, :] = (PL + PR + PD + PT + alpha * dC) / beta
            # central cell index
            cx = int(dim / 2)
            cy = int(dim / 2)

            add_block(x_center=cx, y_center=cy - 1, block=beta_inv * k_block, K_full=k, halfSize=itr + offset - 2)  # LEFT: -X axis
            add_block(x_center=cx, y_center=cy + 1, block=beta_inv * k_block, K_full=k, halfSize=itr + offset - 2)  # RIGHT:+X axis
            add_block(x_center=cx - 1, y_center=cy, block=beta_inv * k_block, K_full=k, halfSize=itr + offset - 2)  # Up: +Y axis
            add_block(x_center=cx + 1, y_center=cy, block=beta_inv * k_block, K_full=k, halfSize=itr + offset - 2)  # DOWN: -Y axis

            k[cx][cy] += sign * alpha / beta

            MEMO[key] = k
            return k
        else:
            return MEMO[key]


# ======================================= 3D =======================================


def poisson_kernel_3d(opts):
    """ Compute a 3D Poisson kernel based on recursive Jacobi.

    Recursion is used to generate 3D analytical Poisson kernels based on 6 nearest neighbors only.
    Precomputing the Jacobi solution to :math:`Ax=b` in the *matrix-form* :math:`L*X = B` (quantities are tensors) \
    using the recursion formula based on the Jacobi update step:

    :math:`x_{here} = (x_{left} + x_{right} + x_{down} + x_{up} + x_{front} + x_{back} + \\alpha * b_{here}) / \\beta`

    where :math:`x_{here}` and :math:`b_{here}` are the elements of :math:`X` and :math:`B` tensors that we would \
    like to update and evaluate, respectively, and :math:`\\alpha` and :math:`\\beta` are set based on the type of \
    the Poisson equation. \
    See :func:`compute_alpha` and :func:`compute_beta`.

    .. note::
        To set the kernel parameters like target iteration (*order*), :math:`dx`, and others, you need to pack
        :code:`OptionsKernel` and :code:`OptionsPoissonSolver` in :code:`OptionsGeneral` and send it to this function. \
        These dataclasses are in :code:`helper.commons.py`.
        To see how to pack options, look at the main demos, or see :func:`helper.commons.generic_options`.

    :param OptionsGeneral opts: parameters bundle
    :return: full Poisson kernel
    """

    if opts.kernel.kernel_type == com.PoissonKernelType.UNIFIED:
        return poisson_kernel_unified_3d(solver_type=opts.solver.solver_type, itr=opts.kernel.itr,
                                         alpha=opts.kernel.alpha, zero_init=opts.solver.zero_init,
                                         clear_memo=opts.kernel.clear_memo)

    elif opts.kernel.kernel_type == com.PoissonKernelType.STANDARD:
        return poisson_kernel_standard_3d(solver_type=opts.solver.solver_type, itr=opts.kernel.itr,
                                          alpha=opts.kernel.alpha, zero_init=opts.solver.zero_init,
                                          clear_memo=opts.kernel.clear_memo)

    else:
        assert False, 'Invalid Kernel Type'


def poisson_kernel_unified_3d(solver_type, itr, alpha, zero_init, clear_memo=True):
    """ Compute a 3D Poisson *Unified* kernel based on recursive Jacobi.
    See :func:`poisson_kernel_3d` for details.

    :param PoissonSolverType solver_type: inverse, forward
    :param int itr: target Jacobi iteration (*kernel order*)
    :param float alpha: see :func:`compute_alpha()`
    :param bool zero_init: zero as the initial guess. If :code:`True`, we produce a smaller kernel, else
        we are warm starting the Jacobi solver with a non-zero matrix, which makes the kernel slightly \
        larger (+2 along each dimension).
        We can use zero initial guess for Poisson pressure only (inverse Poisson).
        Diffusion step (forward Poisson) has to warm start with a non-zero initial guess.
    :param bool clear_memo: better set to be :code:`True` for recursion to avoid garbage cashing. Default= :code:`True`.
    :return: full Jacobi Poisson kernel
    """

    global MEMO

    if clear_memo:
        MEMO = {}
    return poisson_kernel_unified_3d_rec_memo(solver_type=solver_type, itr=itr, alpha=alpha, zero_init=zero_init)


def poisson_kernel_unified_3d_rec_memo(solver_type, itr, alpha, zero_init):
    """ Compute a 3D Poisson *Unified* kernel based on recursive Jacobi using memoization, i.e. speed-up by \
    reusing sub-solutions.
    See :func:`poisson_kernel_3d` for details.

    :param PoissonSolverType solver_type: inverse, forward
    :param int itr: target Jacobi iteration (*kernel order*)
    :param float alpha: see :func:`compute_alpha()`
    :param bool zero_init: zero as the initial guess. If :code:`True`, we produce a smaller kernel, else
        we are warm starting the Jacobi solver with a non-zero matrix, which makes the kernel slightly \
        larger (+2 along each dimension).
        We can use zero initial guess for Poisson pressure only (inverse Poisson).
        Diffusion step (forward Poisson) has to warm start with a non-zero initial guess.
    :return: full Jacobi Poisson kernel
    """

    beta = compute_beta(alpha=alpha, solver_type=solver_type, dim=com.SolverDimension.D3)
    beta_inv = 1.0 / beta

    # Determine alpha and beta based on the solver type
    if zero_init:
        # If zero initial guess, we have a smaller base case
        # can use zero initial guess for Poisson pressure only
        offset = 0  # for n+offset in kernel size

        # base kernel
        k_base = np.zeros((1, 1, 1))
        k_base[0][0][0] = alpha / beta

    else:
        # If warm starting the initial guess, we have a larger base case
        # Diffusion has to use warm starting. No zero initial guess.
        offset = 1  # for n+offset in kernel size

        # base kernel
        k_base = np.zeros((3, 3, 3))
        k_base[1][1][1] = alpha / beta  # center
        k_base[0][1][1] = beta_inv  # up
        k_base[1][0][1] = beta_inv  # left
        k_base[1][2][1] = beta_inv  # right
        k_base[2][1][1] = beta_inv  # down
        k_base[1][1][0] = beta_inv  # back
        k_base[1][1][2] = beta_inv  # front

    # Base Case
    if itr == 1:
        key = 'K' + str(itr)
        if key not in MEMO:
            k = k_base
            MEMO[key] = k
            return k
        else:
            return MEMO[key]

    else:
        # Sub-problem case
        key = 'K' + str(itr)
        if key not in MEMO:
            dim = (2 * (itr + offset)) - 1
            k = np.zeros((dim, dim, dim))
            k_block = poisson_kernel_unified_3d_rec_memo(solver_type=solver_type, itr=itr - 1, alpha=alpha, zero_init=zero_init)

            def add_block(x_center, y_center, z_center, block, K_full, half_size):
                K_full[x_center - half_size: x_center + half_size + 1,
                y_center - half_size: y_center + half_size + 1,
                z_center - half_size: z_center + half_size + 1] += block

            # Accumulating the smaller kernels:
            # K[:, :, :] = (PL + PR + PD + PT + PF + PB + alpha * dC) / beta
            # central cell index
            cx = int(dim / 2)
            cy = int(dim / 2)
            cz = int(dim / 2)

            add_block(x_center=cx, y_center=cy - 1, z_center=cz, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # LEFT: -X axis
            add_block(x_center=cx, y_center=cy + 1, z_center=cz, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # RIGHT:+X axis
            add_block(x_center=cx - 1, y_center=cy, z_center=cz, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # Up: +Y axis
            add_block(x_center=cx + 1, y_center=cy, z_center=cz, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # DOWN: -Y axis
            add_block(x_center=cx, y_center=cy, z_center=cz - 1, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # BACK: +Z axis
            add_block(x_center=cx, y_center=cy, z_center=cz + 1, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # FRONT:-Z axis

            k[cx][cy][cz] += alpha / beta

            MEMO[key] = k
            return k
        else:
            return MEMO[key]


def poisson_kernel_standard_3d(solver_type, itr, alpha, zero_init, clear_memo=True):
    """ Compute a 3D Poisson *Standard* kernel based on recursive Jacobi.
    See :func:`poisson_kernel_3d` for details.

    :param PoissonSolverType solver_type: inverse, forward
    :param int itr: target Jacobi iteration (*kernel order*)
    :param float alpha: see :func:`compute_alpha()`
    :param bool zero_init: zero as the initial guess. If :code:`True`, we produce a smaller kernel, else
        we are warm starting the Jacobi solver with a non-zero matrix, which makes the kernel slightly \
        larger (+2 along each dimension).
        We can use zero initial guess for Poisson pressure only (inverse Poisson).
        Diffusion step (forward Poisson) has to warm start with a non-zero initial guess.
    :param bool clear_memo: better set to be :code:`True` for recursion to avoid garbage cashing. Default= :code:`True`.
    :return: full Jacobi Poisson kernel
    """

    global MEMO

    if clear_memo:
        MEMO = {}
    return poisson_kernel_standard_3d_rec_memo(solver_type=solver_type, itr=itr, alpha=alpha, zero_init=zero_init)


def poisson_kernel_standard_3d_rec_memo(solver_type, itr, alpha, zero_init):
    """ Compute a 3D Poisson *Standard* kernel based on recursive Jacobi using memoization, i.e. speed-up by \
    reusing sub-solutions.
    See :func:`poisson_kernel_3d` for details.

    :param PoissonSolverType solver_type: inverse, forward
    :param int itr: target Jacobi iteration (*kernel order*)
    :param float alpha: see :func:`compute_alpha()`
    :param bool zero_init: zero as the initial guess. If :code:`True`, we produce a smaller kernel, else
        we are warm starting the Jacobi solver with a non-zero matrix, which makes the kernel slightly \
        larger (+2 along each dimension).
        We can use zero initial guess for Poisson pressure only (inverse Poisson).
        Diffusion step (forward Poisson) has to warm start with a non-zero initial guess.
    :return: full Jacobi Poisson kernel
    """

    beta = compute_beta(alpha=alpha, solver_type=solver_type, dim=com.SolverDimension.D3)
    beta_inv = 1.0 / beta

    # this sign is the main difference between universal and warm start kernel types
    sign = get_alpha_sign(solver_type=solver_type, kernel_type=com.PoissonKernelType.STANDARD)

    # Determine alpha and beta based on the solver type
    if zero_init:
        # If zero initial guess, we have a smaller base case
        # can use zero initial guess for Poisson pressure only
        offset = 0  # for n+offset in kernel size

        # base kernel
        k_base = np.zeros((1, 1, 1))
        k_base[0][0][0] = sign * alpha / beta

    else:
        # If warm starting the initial guess, we have a larger base case
        # Diffusion has to use warm starting. No zero initial guess.
        offset = 1  # for n+offset in kernel size

        # base kernel
        k_base = np.zeros((3, 3, 3))
        k_base[1][1][1] = sign * alpha / beta
        k_base[0][1][1] = beta_inv  # up
        k_base[1][0][1] = beta_inv  # left
        k_base[1][2][1] = beta_inv  # right
        k_base[2][1][1] = beta_inv  # down
        k_base[1][1][0] = beta_inv  # back
        k_base[1][1][2] = beta_inv  # front

    # Base Case
    if itr == 1:
        key = 'K' + str(itr)
        if key not in MEMO:
            k = k_base
            MEMO[key] = k
            return k
        else:
            return MEMO[key]

    else:
        # Sub-problem case
        key = 'K' + str(itr)
        if key not in MEMO:
            dim = (2 * (itr + offset)) - 1
            k = np.zeros((dim, dim, dim))
            k_block = poisson_kernel_standard_3d_rec_memo(solver_type=solver_type, itr=itr - 1, alpha=alpha, zero_init=zero_init)

            def add_block(x_center, y_center, z_center, block, K_full, half_size):
                K_full[x_center - half_size: x_center + half_size + 1,
                y_center - half_size: y_center + half_size + 1,
                z_center - half_size: z_center + half_size + 1] += block

            # Accumulating the smaller kernels:
            # K[:, :, :] = (PL + PR + PD + PT + PF + PB + alpha * dC) / beta
            # central cell index
            cx = int(dim / 2)
            cy = int(dim / 2)
            cz = int(dim / 2)

            add_block(x_center=cx, y_center=cy - 1, z_center=cz, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # LEFT: -X axis
            add_block(x_center=cx, y_center=cy + 1, z_center=cz, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # RIGHT:+X axis
            add_block(x_center=cx - 1, y_center=cy, z_center=cz, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # Up: +Y axis
            add_block(x_center=cx + 1, y_center=cy, z_center=cz, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # DOWN: -Y axis
            add_block(x_center=cx, y_center=cy, z_center=cz - 1, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # BACK: +Z axis
            add_block(x_center=cx, y_center=cy, z_center=cz + 1, block=beta_inv * k_block, K_full=k, half_size=itr + offset - 2)  # FRONT:-Z axis

            k[cx][cy][cz] += sign * alpha / beta

            MEMO[key] = k
            return k
        else:
            return MEMO[key]


# ======================================= Alpha & Beta, Solver Types, 2D and 3D =======================================


def get_alpha_sign(solver_type, kernel_type):
    """ Compute :math:`\\alpha` sign that is used in kernel generation, for both 2D and 3D (see :func:`poisson_kernel_2d` \
    and :func:`poisson_kernel_3d` for application).

    In forward and inverse Poisson equations we have different equations, and hence different :math:`\\alpha` signs:
        - *Forward*: :math:`\\alpha = +{(dx)}^2/{(\\kappa . dt)}`
        - *Inverse*: :math:`\\alpha = -{(dx)}^2`

    where:
        - :math:`dx:` cell size
        - :math:`dt:` time step (if *forward* Poisson is intended)
        - :math:`\\kappa`: diffusivity (if *forward* Poisson is intended)

    :param PoissonSolverType solver_type: inverse or forward
    :param PoissonKernelType kernel_type: unified or standard
    :return: sign of alpha
    """

    if solver_type == com.PoissonSolverType.FORWARD:
        return 1

    elif solver_type == com.PoissonSolverType.INVERSE:

        if kernel_type == com.PoissonKernelType.UNIFIED:
            return 1
        elif kernel_type == com.PoissonKernelType.STANDARD:
            return -1

        else:
            assert False, 'Invalid Kernel Type'
    else:
        assert False, 'Invalid Solver Type'


def compute_alpha(dx, dt, kappa, solver_type):
    """Compute :math:`\\alpha` that is used in kernel generation, for both 2D and 3D (see :func:`poisson_kernel_2d` \
    and :func:`poisson_kernel_3d` for application).

    In forward and inverse Poisson equations we have different equations:
        - *Forward*: :math:`\\alpha = +{(dx)}^2/{(\\kappa . dt)}`
        - *Inverse*: :math:`\\alpha = -{(dx)}^2`

    where:
        - :math:`dx:` cell size
        - :math:`dt:` time step (if *forward* Poisson is intended)
        - :math:`\\kappa`: diffusivity (if *forward* Poisson is intended)

    :param float dx: cell size
    :param float dt: time step
    :param float kappa: diffusivity
    :param PoissonSolverType solver_type: inverse or forward
    :return: :math:`\\alpha`
    """
    assert solver_type == com.PoissonSolverType.INVERSE or solver_type == com.PoissonSolverType.FORWARD

    return (dx * dx) if solver_type == com.PoissonSolverType.INVERSE else (dx * dx) / (dt * kappa)


def compute_beta(alpha, solver_type, dim):
    """Compute :math:`\\beta` that is used in kernel generation, for both 2D and 3D (see :func:`poisson_kernel_2d` \
    and :func:`poisson_kernel_3d` for application).

    :math:`\\beta` values based on dimension and Poisson solver type:
        - 2D *Inverse*: :math:`\\beta = 4`
        - 2D *Forward*: :math:`\\beta = 4 + \\alpha`
        - 3D *Inverse*: :math:`\\beta = 6`
        - 3D *Forward*: :math:`\\beta = 6 + \\alpha`

    :param float alpha: see :func:`compute_alpha`
    :param PoissonSolverType solver_type: inverse or forward
    :param SolverDimension dim: solver dimension, 2D or 3D
    :return: :math:`\\beta`
    """
    assert dim == com.SolverDimension.D2 or com.SolverDimension.D3
    assert solver_type == com.PoissonSolverType.INVERSE or solver_type == com.PoissonSolverType.FORWARD
    assert (alpha > 0), "Invalid Alpha: cannot be <= 0"

    const_divisor = 4.0 if dim == com.SolverDimension.D2 else 6.0  # 3D
    return const_divisor if solver_type == com.PoissonSolverType.INVERSE else const_divisor + abs(alpha)


def is_ok_to_zero_init(solver_type):
    """ Verify if a zero initial guess is possible based on the Poisson solver type.

    We can consider a zero initialization of the unknown quantity in the Jacobi solver when computing \
    the Poisson kernel.
    The current method only supports zero initial guess for the inverse Poisson equation (e.g. Poisson pressure).
    It is not possible to use the zero initialization for forward Poisson equation (e.g. diffusion).

    - *Pressure*: it is optional to zero start or warm start
    - *Diffusion*: it is mandatory to warm start

    See also :func:`i_want_to_warm_start`.

    :param PoissonSolverType solver_type: inverse or forward
    :return: if it is ok to zero start the solution
    """

    if solver_type is com.PoissonSolverType.INVERSE:
        return True
    elif solver_type is com.PoissonSolverType.FORWARD:
        return False
    else:
        assert False, 'Invalid Solver Type'


def i_want_to_warm_start(warm_start, solver_type):
    """Automatically generate parameters for the kernel and solver if we want to warm start the solution \
    to :math:`Ax=b`.

    .. note::
        If you are doing *inverse* Poisson and
            - want to avoid negating your input matrix :math:`B` (in the *matrix-form* :math:`LX=B`), **or**
            - want to warm start the solution with the input matrix

        then it is better to use warm start.

    See also :func:`is_ok_to_zero_init`.

    :param bool warm_start: boolean, if we want to warm start or not
    :param PoissonSolverType solver_type: inverse or forward
    :return:
    """

    _zero_init = not warm_start  # request to zero initialization if you do not want to warm start
    _zero_init = _zero_init and is_ok_to_zero_init(solver_type=solver_type)

    if warm_start:
        return com.PoissonKernelType.STANDARD, _zero_init

    else:
        return com.PoissonKernelType.UNIFIED, _zero_init


# ======================================= Aux functions =======================================

def get_kernel_size(itr, zero_init):
    """Get the expected Poisson kernel size along each dimension for a given target Jacobi iteration (*order*) and\
    based on whether we want to warm start or not.

    .. note::
        The kernel is always either square (2D matrix) or cube (3D tensor). The kernel actual size for a \
        given target Jacobi iteration :math:`i` is:
            - :code:`zero_init=True` : :math:`{(2*i-1)}^d`
            - :code:`zero_init=False` : :math:`{(2*i+1)}^d`

        for dimension :math:`d=2,3`. This function only returns the base size and not the kernel actual size.

    :param int itr: target Jacobi iteration
    :param bool zero_init: see :func:`is_ok_to_zero_init`
    :return: kernel size along each dimension
    """

    if zero_init:
        return (2 * itr) - 1
    else:
        return (2 * itr) + 1


def get_kernel_size_from_half_size(half_size):
    """ Given a half Poisson filter size compute the actual kernel size.

    :param int half_size: half filter size excluding the element in the middle.
    :return: kernel size
    """
    return 2 * half_size + 1
