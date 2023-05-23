""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`


    .. note::
        :code:`dataclass` needs python 3.7, replace it with :code:`NamedTuple` with python 3.6.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List
# from typing import NamedTuple


class SolverDimension(Enum):
    """Dimension of the solver, either 2D or 3D """
    D2 = 2
    D3 = 3
    UNKNOWN = 0


class PoissonSolverType(Enum):
    """
    :code:`INVERSE`: Solving :math:`Ax=b` for :math:`x` in *matrix-vector* setup, or in our *matrix-matrix* convolution \
    setup, solve for :math:`X` in :math:`L * X=B` for the given input :math:`B`, where *Laplacian* :math:`L` is \
    implicitly approximated.
    Examples: pressure-projection, seamless cloning.

    :code:`FORWARD`: Solving for *rhs* :math:`b` in :math:`Ax=b` in *matrix-vector* setup, or in our *matrix-matrix* \
    convolution setup, solve for :math:`B` in :math:`L*X=B` for the given input :math:`X`, where *Laplacian* :math:`L` \
    is implicitly approximated.
    Examples: density diffusion, heat equation, viscosity.
    """
    INVERSE = 1
    FORWARD = 2
    UNKNOWN = 0


class PoissonKernelType(Enum):
    """
    :code:`STANDARD`: (*Safe Default*) Takes care of :math:`\\pm \\alpha` sign in the kernel. Supports warm starting for \
    both *inverse* and *forward* Poisson equations.

    :code:`UNIFIED`: If using this, a negative sign must be absorbed in the input for *inverse* Poisson. Always uses \
    :math:`+ \\alpha` sign in the kernel. Only supports warm starting in *forward* Poisson.
    """
    STANDARD = 1
    UNIFIED = 2
    UNKNOWN = 0


def is_solver_2d(dim):
    return dim == SolverDimension.D2


def is_solver_3d(dim):
    return dim == SolverDimension.D3


class DataFileFormat(Enum):
    """IO file formats"""
    npy = 1
    npz = 2
    txt = 3
    hlsli = 4
    cpp = 5

    UNKNOWN = 0


class BoundaryType(Enum):
    """Boundary conditions.

    .. warning::
        Only :code:`NEUMANN_EDGE` is fully implemented and tested.

    :code:`CONST_FRAME` means simple single padded values; mostly no action is required.
    """
    NEUMANN_EDGE = 1
    NEUMANN_CENTER = 2
    DIRICHLET = 3
    CONST_FRAME = 4  # simple single padded values, mostly no action is required


class OutputFormat(Enum):
    """Choices to compute and pack filter values/components.

    Format choices:
        - :code:`ABSORBED_FILTERS`:
            - *Ready to use filters*
            - The core weights are already absorbed into the factors. Safe to use the filters directly.
        - :code:`ALL_COMPONENTS`:
            - Core weights of the decomposed kernel are not absorbed into the filters
            - Receive a break-down of components: kernel, cores, factors
            - *USE CASE*: we can experiment with either Tucker or CP for the decomposition method.

    """
    ABSORBED_FILTERS = 1
    ALL_COMPONENTS = 2
    UNKNOWN = 0


class ExportFormat(Enum):
    """C++/HLSL friendly export options for the filters

    Choices:
        - :code:`SEPARATE_RANK_FILTERS`: float[ float, float, ..] separate arrays for each ranked filter
        - :code:`GROUP_RANK_SEQUENTIAL`: float[ rank 1 floats.. // rank 2 floats.. // ..] packing 4 ranked filters sequentially
        - :code:`GROUP_RANK_FLOAT4_HLSL`: float4[ float4(rank 1 .. 4), float4(rank 1 .. 4), ..] packing hlsl friendly
    """
    SEPARATE_RANK_FILTERS = 1
    GROUP_RANK_SEQUENTIAL = 2
    GROUP_RANK_FLOAT4_HLSL = 3


class DecompMethod(Enum):
    """Decomposition methods for 2D and 3D.

    :code:`NONE`: no decomposition, forces a full kernel return. Anything but :code:`NONE` forces a return of a \
    reduced kernel.
    """
    NONE = 1
    SVD_2D = 2
    SYM_CP_3D = 3
    TUCKER_3D = 4
    UNKNOWN = 0


class ConvolutionOrientation(Enum):
    """ By convolution orientation we mean the filter orientation when convolving the domain.

    .. note::
        - :code:`FIBER` is the same as *depth* for 3D convolutions.
        - :code:`CENTER` only does one multiplication on the central cell, no orientation.
    """
    HORIZONTAL = 1
    VERTICAL = 2
    FIBER = 3
    CENTER = 4
    UNKNOWN = 0


class TruncationMode(Enum):
    """Methods to truncate the filters.

    Regardless of the method we are always cutting off both sides of the symmetrical filter equally.

    [....outer part....cut | ......inner part...... | cut ....outer part....]


    Choices:
        - :code:`PERCENTAGE`: truncate :math:`\%` of the filter; values in range :math:`[0, 1]`
        - :code:`FIXED_THRESHOLD`: **adaptive truncation** : zero out or cut off any element whose absolute value is \
            below a fixed threshold. \
            Note that in this technique we only find and truncate the small values on the outer side of the symmetrical \
            filter, i.e. we use the fixed threshold to find the first element that should be kept on either side of the \
            filter. Then mark the left and right places to cut. We do not test the "inner" elements against the threshold, \
            so after truncation is done, it is possible that the array has still elements below the cut-off threshold. \
            Plot filters to see what this means.
    """
    PERCENTAGE = 1
    FIXED_THRESHOLD = 2
    UNKNOWN = 0


class DataMatrixMode(Enum):
    """Making sample data matrices with different patterns.

    Choices:
        - :code:`CONST_VALUE`: constant value everywhere
        - :code:`RANDOMIZE_INT`: random data
        - :code:`SPECIAL_PATTERN`: special scripted patters
        - :code:`LINEAR_RANGE_FULL`: :math:`[1, 2, ..., n]`
        - :code:`LINEAR_RANGE_4BLOCK_SYMMETRIC_MIRRORED`: generates 4 sub-matrices that are mirrored of \
            each other. Also, each block is symmetric along the diagonal
        - :code:`RANDOMIZE_FLOAT_UNIFORM`: use uniform random data to generate the input matrix
        - :code:`RANDOMIZE_FLOAT_GAUSSIAN`: use Gaussian random data to generate the input matrix
    """
    CONST_VALUE = 1
    RANDOMIZE_INT = 2
    SPECIAL_PATTERN = 3
    LINEAR_RANGE_FULL = 4
    LINEAR_RANGE_4BLOCK_SYMMETRIC_MIRRORED = 5
    RANDOMIZE_FLOAT_UNIFORM = 6
    RANDOMIZE_FLOAT_GAUSSIAN = 7


class DataMatrixPattern(Enum):
    """Use this function to create options for scripting matrix data with special patterns.

    Current (only) implementation: :code:`CENTRAL_SQUARE`: Make a data matrix with a non-zero box at the center \
    and zero everywhere else.
    """
    CENTRAL_SQUARE = 1


class ColorMapModesVectorField(Enum):
    """Options for plotting vector fields.

    Choices:
        - :code:`VECTOR_ANGLE`: color map based on the vector angle
        - :code:`VECTOR_LENGTH`: color map based on the vector length
    """
    VECTOR_ANGLE = 1
    VECTOR_LENGTH = 2


class NormOptions(Enum):
    """ Matrix norms.

    Choices:
        - :code:`MSE`: mean squared error
        - :code:`FROBENIUS`: frobenius
        - :code:`INF`: infinity
    """
    MSE = 1
    FROBENIUS = 2
    INF = 3


def to_string_norm(norm):
    if norm == NormOptions.FROBENIUS:
        name = 'FROBENIUS'
    elif norm == NormOptions.MSE:
        name = 'MSE'
    elif norm == NormOptions.INF:
        name = 'INF'
    else:
        assert False, 'Unknown norm function'

    return name


# ------------ dataclass needs python 3.7, replace it with NamedTuple with python 3.6 ---------
@dataclass
class OptionsPoissonSolver:
    """Dataclass to pack options for the Poisson solver.

    .. note::
        To see how to pack options, look at main demos, or see :func:`generic_options`.

    :param SolverDimension dim: 2D or 3D
    :param PoissonSolverType solver_type: *inverse* or *forward* Poisson
    :param bool zero_init: (**Default** = :code:`True`) if the initial guess in the Jacobi solver is set to zero.\
        If not use :math:`b` in :math:`Ax=b`. If zero initial guess, we produce a smaller kernel. \
        If not using zero initial guess, we are warm starting the Jacobi solver with a nonzero matrix. \
        This makes the kernel slightly larger :math:`(+2)`. We can use zero initial guess for Poisson pressure \
        only (*inverse* Poisson). Diffusion step (*forward* Poisson) has to warm start with a nonzero initial guess. \
        See :func:`functions.generator.i_want_to_warm_start`.

    """
    dim: SolverDimension
    solver_type: PoissonSolverType  # inverse, forward
    zero_init: bool = True


@dataclass
class OptionsKernel:
    """Dataclass to pack options for the kernel generator.

    .. note::
        To see how to pack options, look at main demos, or see :func:`generic_options`.

    :param PoissonKernelType kernel_type: standard, unified
    :param int itr: (**Default** = :code:`30`) target Jacobi iteration, also Poisson filters order
    :param float dx: cell size (**Default** = :code:`1.0`)
    :param float dt: diffusivity rate per time step (different from simulation time step) (**Default** = :code:`1.0`)
    :param float kappa: diffusivity (**Default** = :code:`1.0`)
    :param float alpha: (**Default** = :code:`1.0`). See :func:`functions.generator.compute_alpha`
    :param float beta: (**Default** = :code:`1.0`). See :func:`functions.generator.compute_beta`
    :param bool clear_memo: better set to be :code:`True` for recursion to avoid garbage cashing when using \
        memoization (**Default** = :code:`True`)

    """
    kernel_type: PoissonKernelType
    itr: int = 30
    dx: float = 1.0
    dt: float = 1.0
    kappa: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    clear_memo: bool = True


@dataclass
class OptionsReduction:
    """Dataclass to pack options for reductions.

    .. note::
        To see how to pack options, look at main demos, or see :func:`generic_options`.

    :param bool reduce: (**Default** = :code:`False`)
    :param bool use_separable_filters: (**Default** = :code:`False`) use separable filters
        if :code:`True`, else use the low rank kernel.
    :param int rank: (**Default** = :code:`1`)
        desired reduction rank. will be ignored if not in the reduction mode. If in reduction, then
        if :code:`rank==None` get the low rank kernel, if :code:`rank >= 1` then get the separable filters.
        You can use :func:`decompositions.get_max_rank_2d()` for the best approximation
    :param TruncationMode truncation_method: (**Default** = :code:`PERCENTAGE`)
    :param float truncation_value: (**Default** = :code:`0.0`) depends on the truncation method: if :code:`PERCENTAGE`,
        truncation factor in
        :math:`[0, 1]`. If :code:`FIXED_THRESHOLD`, cut any elements smaller than a fixed value, only working with
        symmetrical filters, and only cutting the outer parts of the filters. This value will be ignored if not in
        the reduction mode.
    :param bool preserve_shape: (**Default** = :code:`True`) in case of adaptive truncation with \
        :code:`TruncationMode.FIXED_THRESHOLD` we can either keep the shape of the filter and just zero out the truncated
        values (:code:`preserve_shape=True`) or trim the filters (:code:`preserve_shape=False`).
        Forcing to preserve shape makes less complicated convolutions for different ranks as all the filters will
        have the same size. Also, the data generator function that automatically computes data size and dynamic padding
        does not currently support varying size filters for each rank.
    :param DecompMethod decomp_method: (**Default** = :code:`DecompMethod.UNKNOWN`)
    :param OutputFormat output_format: (**Default** = :code:`OutputFormat.ABSORBED_FILTERS`) only used for 3D filters
    """
    reduce: bool = False
    use_separable_filters: bool = False
    rank: int = 1
    truncation_method: TruncationMode = TruncationMode.PERCENTAGE
    truncation_value: float = 0.0
    preserve_shape: bool = True
    decomp_method: DecompMethod = DecompMethod.UNKNOWN
    output_format: OutputFormat = OutputFormat.ABSORBED_FILTERS


@dataclass()
class OptionsBoundary:
    """Dataclass to pack options for boundary treatment.

    .. note::
        To see how to pack options, look at main demos, or see :func:`generic_options`.

    :param BoundaryType condition: (**Default** = :code:`BoundaryType.CONST_FRAME`)
    :param bool enforce: (**Default** = :code:`False`) with or without boundary condition
    :param float val: (**Default** = :code:`1.0`) constant boundary condition value to be enforced
    :param int padding_size: (**Default** = :code:`1`) thickness of the boundary
    :param bool dynamic_padding: (**Default** = :code:`True`) automatically add padding to compensate for
        convolution shrinkage
    :param bool left_wall: (**Default** = :code:`True`) active left wall boundary
    :param bool right_wall: (**Default** = :code:`True`) active right wall boundary
    :param bool up_wall: (**Default** = :code:`True`) active up wall boundary
    :param bool down_wall: (**Default** = :code:`True`) active down wall boundary
    :param bool front_wall: (**Default** = :code:`True`) 3D: active front wall boundary
    :param bool back_wall: (**Default** = :code:`True`) 3D: active behind wall boundary
    :param bool obj_collide: (**Default** = :code:`True`) collide with the object in the domain
    :param bool post_solve_enforcement: (**Default** = :code:`True`) clean up boundary treatment;
        enforce boundary condition after Jacobi/Poisson solve
    """
    condition: BoundaryType = BoundaryType.CONST_FRAME
    enforce: bool = False
    val: float = 1.0
    padding_size: int = 1
    dynamic_padding: bool = True
    left_wall: bool = True
    right_wall: bool = True
    up_wall: bool = True
    down_wall: bool = True
    front_wall: bool = True
    back_wall: bool = True
    obj_collide: bool = True
    post_solve_enforcement: bool = True


@dataclass
class DataMatrixPatternParams:
    """Special data pattern generation parameters."""
    radius: float = 0.05


@dataclass
class OptionsDataMatrix:
    """Packing options for data matrix generation

    :param DataMatrixPatternParams pattern_params: special pattern parameters
    :param tuple shape: (**Default** = :code:`(41, 41)`) 2d input matrix size. Does not have to be square.
        For 3d use :code:`(int, int, int)`
    :param DataMatrixMode mode: (**Default** = :code:`DataMatrixMode.RANDOMIZE_INT`)
    :param bool force_symmetric_unit_range: (**Default** = :code:`False`) Data will be forced to be in :math:`[-1, +1]`
    :param float const_input: (**Default** = :code:`1.0`) using a matrix with the same element values.
        If randomized, this will be ignored
    :param int rand_seed: (**Default** = :code:`13`)
    :param tuple rand_range: (**Default** = :code:`(1, 10) `) - :math:`(int, int)`. integer in (low, high) to
        generate random matrix data. 'high' is excluded.
    :param DataMatrixPattern special_pattern: (**Default** = :code:`DataMatrixPattern.CENTRAL_SQUARE`)
        special data patterns

    """
    pattern_params: DataMatrixPatternParams = DataMatrixPatternParams()
    shape: (int, int) = (41, 41)
    mode: DataMatrixMode = DataMatrixMode.RANDOMIZE_INT
    force_symmetric_unit_range: bool = False
    const_input: float = 1.
    rand_seed: int = 13
    rand_range: (int, int) = (1, 10)
    special_pattern: DataMatrixPattern = DataMatrixPattern.CENTRAL_SQUARE


@dataclass
class OptionsGeneral:
    """Packing all options into one super pack.

    .. note::
        To see how to pack options, look at main demos, or see :func:`generic_options`.
    """
    solver: OptionsPoissonSolver
    kernel: OptionsKernel
    reduction: OptionsReduction
    boundary: OptionsBoundary
    input: OptionsDataMatrix


def generic_options(dim=SolverDimension.D2, solver_type=PoissonSolverType.INVERSE, zero_init=True,
                    kernel_type=PoissonKernelType.STANDARD, itr=60, rank=4, domain_size=(81, 81), rand_seed=50,
                    dx=1.0, dt=1.0, kappa=1.0, alpha=1.0, beta=4.0,
                    truncation_method=TruncationMode.PERCENTAGE, truncation_value=0.0,
                    obj_collide=True, force_symmetric_unit_range=True):
    """
    Generating a set of generic and safe paramteres. Use this as an example of how to pack different options.
    Look at parameter types for explanations.

    :param SolverDimension dim: See :func:`SolverDimension`
    :param PoissonSolverType solver_type: See :func:`PoissonSolverType`
    :param bool zero_init: See :func:`OptionsPoissonSolver`
    :param PoissonKernelType kernel_type: See :func:`PoissonKernelType`
    :param int itr: target Jacobi iteration, also Poisson filter order. See :func:`OptionsKernel`
    :param int rank: See :func:`OptionsReduction`
    :param tuple domain_size: See :func:`OptionsDataMatrix`
    :param rand_seed: See :func:`OptionsDataMatrix`
    :param dx: See :func:`OptionsKernel`
    :param dt: See :func:`OptionsKernel`
    :param kappa: See :func:`OptionsKernel`
    :param alpha: See :func:`OptionsKernel`
    :param beta: See :func:`OptionsKernel`
    :param truncation_method: See :func:`OptionsReduction`
    :param truncation_value: See :func:`OptionsReduction`
    :param obj_collide: See :func:`OptionsBoundary`
    :param force_symmetric_unit_range: See :func:`OptionsDataMatrix`
    :return:
    """
    opts_solver = OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init)

    # Options kernel
    opts_kernel = OptionsKernel(kernel_type=kernel_type,
                                itr=itr,
                                dx=dx,
                                dt=dt,
                                kappa=kappa,
                                alpha=alpha,
                                beta=beta,
                                clear_memo=True)

    # Options reduction #
    opts_reduction = OptionsReduction(decomp_method=DecompMethod.SVD_2D,
                                      reduce=True,
                                      use_separable_filters=True,
                                      rank=rank,
                                      truncation_method=truncation_method,
                                      truncation_value=truncation_value,
                                      preserve_shape=True if truncation_method == TruncationMode.FIXED_THRESHOLD
                                      else False)
    # data generation
    mode = DataMatrixMode.RANDOMIZE_FLOAT_GAUSSIAN
    # mode = DataMatrixMode.RANDOMIZE_FLOAT_UNIFORM

    opts_input = OptionsDataMatrix(shape=domain_size,
                                   mode=mode,
                                   rand_seed=rand_seed,
                                   rand_range=(-1, 1),  # integer, excluding high
                                   force_symmetric_unit_range=force_symmetric_unit_range,
                                   const_input=1.)
    # boundary treatment
    opts_boundary = OptionsBoundary(enforce=True,
                                    condition=BoundaryType.NEUMANN_EDGE,
                                    obj_collide=obj_collide,
                                    post_solve_enforcement=True,
                                    val=0,
                                    dynamic_padding=False,
                                    padding_size=get_default_padding(),
                                    left_wall=True,
                                    right_wall=True,
                                    up_wall=True,
                                    down_wall=True)

    # make a bundle
    opts_general = OptionsGeneral(solver=opts_solver, kernel=opts_kernel, reduction=opts_reduction,
                                  boundary=opts_boundary, input=opts_input)

    return opts_general


def get_default_padding():
    """
    :return: Fixed padding size
    """
    return 1


@dataclass
class OptionsSubPlots:
    """Subplot options"""
    layout: (int, int) = (2, 2)
    titles: List[List[str]] = field(default_factory=lambda: [[] for m in range(2)])
    highlights: List[List[np.array]] = field(default_factory=lambda: [[] for m in range(2)])
    cbar_percentiles: List[List[float]] = field(default_factory=lambda: [[] for m in range(2)])
    cmaps: List[List[str]] = field(default_factory=lambda: [[] for m in range(2)])


@dataclass
class OptionsPlots:
    """Plot options"""
    show_values: bool = False
    no_ticks: bool = False  # if True removes the axis label and ticks
    frame_on: bool = True  # border control [unused]. To use it do ax.set(frame_on=True/False)
    limits_x_tight: bool = False  # no horizontal space in the axis
    limits_y_tight: bool = False  # no vertical space in the axis
    aspect_ratio: float = 1.0
    line_widths: float = 0.2
    cmap: str = "rocket_r"  # as the convention in matplotlib any color map has a reversed version by adding _r
    blend_cmaps: bool = False  # blend between a cmap and an additional one (usually to make more detailed cmap)
    cmap_secondary_blend: str = 'flag'  # secondary cmap used in blending 2 cmaps
    fmt: str = ".2f"
    plt_show: bool = False
    cbar: bool = True
    cbar_border: bool = False
    cbar_outline_color: str = 'black'
    cbar_label: str = None
    cbar_label_color: str = 'white'
    cbar_ticks: bool = True
    cbar_tick_color: str = None
    cbar_only_min_max: bool = False  # only min and max ticks
    cbar_add_percentile: bool = False   # adding percentile ticks
    cbar_orientation: str = 'horizontal'  # horizontal or vertical
    cbar_location: str = 'bottom'  # 'right', 'left', 'top', 'bottom'
    cbar_shrink: float = 1.  # works with Seaborn
    cbar_scientific: bool = False  # use formatter to set cbar scientific format
    cbar_special: bool = False  # use this flag for any special cbar code, anywhere
    # https://towardsdatascience.com/heatmap-basics-with-pythons-seaborn-fb92ea280a6c
    show_contour: bool = False  # only used with imshow
    contour_res: int = 50  # contour resolution
    contour_show_values: bool = False  # show contour values in the plot
    contour_cmap: str = 'gist_heat'  # contour cmap
    contour_cbar: bool = False  # show contour cbar
    contour_cbar_orientation: str = 'vertical'  # or 'horizontal, but matplotlib might complain
    contour_cbar_location: str = 'right'  # 'right', 'left', 'top', 'bottom'
    contour_cbar_border: bool = False  # show contour cbar border
    contour_cbar_no_ticks: bool = False  # show contour cbar ticks
    beautify: bool = False
    projection: str = '2d'
    interpolate: bool = False  # interpolate image
    interp_1d: bool = False  # interpolate 1d array
    interp_1d_res: int = 10  # 1=no interpolation, 1> increased resolution
    interp_2d_mode: str = None  # used with imshow with the following modes:
    # [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
    #  'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    #  'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    # https: // matplotlib.org / stable / gallery / images_contours_and_fields / interpolation_methods.html
    interp_3d: bool = False  # 3d interpolation
    alpha: float = 1.0
    highlight_edgecolor: str = 'red'
    highlight_facecolor: str = 'pink'
    highlight_fill: bool = False
    highlight_line_width: int = 2
    axis_background_color: str = None
    canvas_background_color: str = None


@dataclass
class Vector2DIntData:
    v1: int = 0
    v2: int = 0


class Vector2DInt:

    def __init__(self, v1, v2):
        self.v_data = Vector2DIntData(v1=v1, v2=v2)

    def add(self, another):
        self.v_data.v1 += another.v_data.v1
        self.v_data.v2 += another.v_data.v2

    def mul(self, another):
        self.v_data.v1 *= another.v_data.v1
        self.v_data.v2 *= another.v_data.v2

    def set_v1(self, v1):
        self.v_data.v1 = v1

    def set_v2(self, v2):
        self.v_data.v2 = v2

    def set(self, v1, v2):
        self.set_v1(v1=v1)
        self.set_v2(v2=v2)

    def set_from(self, another):
        self.set_v1(v1=another.v_data.v1)
        self.set_v2(v2=another.v_data.v2)

    def value(self):
        return self.v_data

    def v1(self):
        return self.v_data.v1

    def v2(self):
        return self.v_data.v2


@dataclass
class Vector3DIntData:
    v1: int = 0
    v2: int = 0
    v3: int = 0


class Vector3DInt:

    def __init__(self, v1, v2, v3):
        self.v_data = Vector3DIntData(v1=v1, v2=v2, v3=v3)

    def add(self, another):
        self.v_data.v1 += another.v_data.v1
        self.v_data.v2 += another.v_data.v2
        self.v_data.v3 += another.v_data.v3

    def mul(self, another):
        self.v_data.v1 *= another.v_data.v1
        self.v_data.v2 *= another.v_data.v2
        self.v_data.v3 *= another.v_data.v3

    def set_v1(self, v1):
        self.v_data.v1 = v1

    def set_v2(self, v2):
        self.v_data.v2 = v2

    def set_v3(self, v3):
        self.v_data.v3 = v3

    def set(self, v1, v2, v3):
        self.set_v1(v1=v1)
        self.set_v2(v2=v2)
        self.set_v3(v3=v3)

    def set_from(self, another):
        self.set_v1(v1=another.v_data.v1)
        self.set_v2(v2=another.v_data.v2)
        self.set_v3(v3=another.v_data.v3)

    def value(self):
        return self.v_data

    def v1(self):
        return self.v_data.v1

    def v2(self):
        return self.v_data.v2

    def v3(self):
        return self.v_data.v3
