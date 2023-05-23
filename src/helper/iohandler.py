""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`

    .. _iohandler_notes:

    Technicals
    ============

    .. tip::
        A number of pre-generated 2D and 3D filters, and their corresponding decomposition components already exist \
        in different file formats in this repo.

        Path to files:
            - filters (*c++/hlsli/npy/npz*): :code:`data/preprocess/filters`
            - csv: :code:`data/preprocess/components`

    - Naming convention:
        - If *"min_x_max_y"* in the file name, it is a filter database with orders in the *min* and *max* range
        - if *"single_"* prefix in the file name, it is only a single order filter set

    - Useful 3D filter samples:
        *already available in the project*

        *[For Poisson pressure]*

        - A useful **database** to immediately get started with in simulation:

          :code:`poisson_D3_INVERSE_STANDARD_dx_1.0_itr_min_1_max_100.hlsli`

        - Useful pre-generated **single** 3D Poisson filters:

          :code:`single_poisson_D3_INVERSE_STANDARD_dx_1.0_itr_60.hlsli`
          :code:`single_poisson_D3_INVERSE_STANDARD_dx_1.0_itr_100.hlsli`
          :code:`single_poisson_D3_INVERSE_STANDARD_dx_1.0_itr_200.hlsli`

        - If you experience stability issues with the Poisson solution, try lowering :math:`dx`. Some examples are \
            pre-generated for you with :math:`dx=0.9` (note the :code:`_dx_0.9` key in the file name):

          :code:`single_poisson_D3_INVERSE_STANDARD_dx_0.9_itr_60.hlsli`
          :code:`single_poisson_D3_INVERSE_STANDARD_dx_0.9_itr_100.hlsli`
          :code:`single_poisson_D3_INVERSE_STANDARD_dx_0.9_itr_200.hlsli`

    - Export Filters: Working With Database
        Generating filters specially in 3D and for high orders can be very expensive. \
        You can generate and export filters (HLSL friendly) once and load them when using them from the database (npz).
        
        Check out \
        :func:`demo_generate_and_export_filters_database_2d` or \
        :func:`demo_generate_and_export_filters_database_3d` \
        to see how it is done.
        
        - If not already generated, you need to set :code:`load_database_from_file=False` in order to force generate \
            the database for the first time. Then you can use :code:`load_database_from_file=True` to load the filters.
        - Use :code:`only_single_itr` if you only want to export one filter. Set the desired target iteration in\
            :code:`max_itr`. There is safety code to save the database with *"single_"* in its name to avoid accidental \
            overwriting the actual database.
        - Use :func:`com.ExportFormat.GROUP_RANK_FLOAT4_HLSL` to make :code:`R14` and :code:`R58` array groups \
            (default). Better for the shader code; else each ranked filter will be stored in a single float array \
            in HLSL.
    
        - To export filters to simulation use:
    
            - Poisson-Pressure (Inverse Poisson):
                :code:`solver_type=PoissonSolverType.INVERSE`
                :code:`kernel_type=PoissonKernelType.UNIFIED` (and multiply the divergence by :math:`-1`), or

                :code:`kernel_type=PoissonKernelType.STANDARD` (no need to adjust divergence)
    
            - Diffusion (Forward Poisson):
                :code:`solver_type=PoissonSolverType.FORWARD`
                :code:`kernel_type=PoissonKernelType.STANDARD`

        - Example diffusivity values that worked well in the simulation
            :math:`\\kappa = [5e-5, 5e-4, 1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1.]`

"""
import sys
import os
import numpy as np
import src.helper.commons as com
import src.functions.decompositions as dec
import src.functions.generator as gen

sys.path.insert(0, '../../')

path_base_data = '../../data'
# if having difficulty with the base data path, use the following to access environment variables path:
# path_base_data = os.environ.get('PROJECT_KEY_NAME', '../../data')
# with '../../data' being the default value if the key does not exist

path_preprocess_filters_database = path_base_data + '/preprocess/filters/'
path_preprocess_components = path_base_data + '/preprocess/components/'

os.makedirs(path_preprocess_filters_database, exist_ok=True)
os.makedirs(path_preprocess_components, exist_ok=True)

path_preprocess_filters_database = os.path.normpath(path_preprocess_filters_database)
if not path_preprocess_filters_database.endswith(os.path.sep):
    path_preprocess_filters_database += os.path.sep

path_preprocess_components = os.path.normpath(path_preprocess_components)
if not path_preprocess_components.endswith(os.path.sep):
    path_preprocess_components += os.path.sep


def generate_dump_path_filter_database(max_itr, dx, kappa, dim, solver_type, kernel_type,
                                       only_single_itr, single_itr_num, file_format, min_itr=1):
    """Auto-generate path string.

    :param int max_itr: max target Jacobi iteration (filter order)
    :param float dx: cell size
    :param float kappa: diffusivity (*will be ignored if not* :code:`FORWARD`)
    :param SolverDimension dim: See :func:`SolverDimension`
    :param PoissonSolverType solver_type: See :func:`PoissonSolverType`
    :param PoissonKernelType kernel_type: See :func:`PoissonKernelType`
    :param bool only_single_itr: if we are interested only in generating a single set of filters for \
        a specific target Jacobi iteration/filter order.
    :param int single_itr_num: target Jacobi iteration/filter order if :code:`only_single_itr=True`
    :param DataFileFormat file_format:
    :param int min_itr: min target Jacobi iteration (filter order) (**Default=** :code:`1`)
    :return: path string
    """

    kappa_str = f'_kappa_{kappa}' if solver_type == com.PoissonSolverType.FORWARD else '' # add diffusivity only for FORWARD

    if min_itr == max_itr:
        only_single_itr = True
        single_itr_num = min_itr

    if only_single_itr:
        fname = f'poisson_{dim.name}_{solver_type.name}_{kernel_type.name}_dx_{dx}' + kappa_str + \
                f'_itr_{single_itr_num}.{file_format.name}'
        fname = 'single_' + fname
    else:
        fname = f'poisson_{dim.name}_{solver_type.name}_{kernel_type.name}_dx_{dx}' + kappa_str + \
                f'_itr_min_{min_itr}_max_{max_itr}.{file_format.name}'

    path = path_preprocess_filters_database + fname

    return path


def save_filter_database(filters_iterations, max_itr, dx, kappa, dim, solver_type, kernel_type, min_itr=1):
    """Save filter values either as :code:`npz` (many filters) or :code:`npy` (single filter).
    If :code:`min_itr==max_itr` take it as a signal to save only a single filter.

    Saved format: *(iteration, rank, 1darray filter values)*.

    :param List[int] filters_iterations: list of target Jacobi iterations (filter orders)
    :param int max_itr: max target Jacobi iteration (filter order)
    :param float dx: cell size
    :param float kappa: diffusivity (*will be ignored if not* :code:`FORWARD`)
    :param SolverDimension dim: See :func:`SolverDimension`
    :param PoissonSolverType solver_type: See :func:`PoissonSolverType`
    :param PoissonKernelType kernel_type: See :func:`PoissonKernelType`
    :param int min_itr: min target Jacobi iteration (filter order) (**Default=** :code:`1`)
    :return: path string where the file is saved
    """

    single_itr = min_itr == max_itr

    print('saving filters database...')
    file_format = com.DataFileFormat.npy if single_itr else com.DataFileFormat.npz
    path = generate_dump_path_filter_database(max_itr=max_itr, dx=dx, kappa=kappa, dim=dim, solver_type=solver_type,
                                              kernel_type=kernel_type, file_format=file_format,
                                              only_single_itr=single_itr,
                                              single_itr_num=min_itr,
                                              min_itr=min_itr)
    # packing format:
    # (iteration, rank, 1darray filter values)

    print('Attempting to save filter database ' + path)

    if single_itr:
        np.save(path, filters_iterations, allow_pickle=True)
    else:
        np.savez(path, *filters_iterations, allow_pickle=True)
    # elif dim == com.SolverDimension.D2:
    #     list_copy = filters_iterations[:]  # explicit copy, no copy by reference

    #     np.savez(path, *filters_iterations, allow_pickle=True)

    # elif dim == com.SolverDimension.D3:
    #     list_copy = filters_iterations[:]  # explicit copy, no copy by reference

    #     np.savez(path, *list_copy, allow_pickle=True)
    # else:
    #     assert False, "Unknownw Solver Dimension"

    print('done')
    print('File saved in ' + path)

    return path


def load_filter_database(max_itr, dx, kappa, dim, solver_type, kernel_type, min_itr=1):
    """Load filter values. If :code:`min_itr==max_itr` take it as a signal to load only a single filter, else return filters
    for all target Jacobi iterations between :code:`min_itr` and :code:`max_itr` if the database exists.

    :param int max_itr: max target Jacobi iteration (filter order)
    :param float dx: cell size
    :param float kappa: diffusivity (*will be ignored if not* :code:`FORWARD`)
    :param SolverDimension dim: See :func:`SolverDimension`
    :param PoissonSolverType solver_type: See :func:`PoissonSolverType`
    :param PoissonKernelType kernel_type: See :func:`PoissonKernelType`
    :param int min_itr: min target Jacobi iteration (filter order) (**Default=** :code:`1`)
    :return: list of filters for each order if loading a database of filters, else a single set of filter values
    """
    single_itr = min_itr == max_itr

    print('loading database...')
    file_format = com.DataFileFormat.npy if single_itr else com.DataFileFormat.npz
    path = generate_dump_path_filter_database(max_itr=max_itr, dx=dx, kappa=kappa, dim=dim, solver_type=solver_type,
                                              kernel_type=kernel_type, file_format=file_format,
                                              only_single_itr=single_itr,  # forces to load the whole database
                                              single_itr_num=min_itr,  # don't care matter since we do not use it here
                                              min_itr=min_itr
                                              )

    print('Attempting to load filter database ' + path)

    if single_itr:
        return np.load(path, allow_pickle=True)
    else:
        data = np.load(path, allow_pickle=True)
        print('done.')

        filters_iterations = data
        # filters_iterations = data['filters_iterations']

        # if dim == com.SolverDimension.D3:
        #     # ------- This is annoying but savez somehow does not like the shape of filters for 3D; this is while
        #     # we have assumed there is no difference between 2D and 3D, just bunch of (rank, filter) pairs for
        #     # all iterations. Seems like the shape broadcasting has difficulty with 3D pairs.
        #     #
        #     # A quick fix is to swap the (rank, filter) other for all iterations. We need to do the same
        #     # when saving 3D.
        #     # for i in range(len(filters_iterations)):
        #     for i in range(len(filters_iterations)-1):
        #         name = "arr_{}".format(i)
        #         # print("{}: {}".format(i, name))
        #         setattr(filters_iterations, name, filters_iterations[name])
        #         # filters_iterations[i] = filters_iterations[i].transpose()

        return filters_iterations


def format_scientific(x, dynamic_format=False):
    """
    :param float x: input value
    :param bool dynamic_format: fixed or dynamic format
    :return: string
    """
    if dynamic_format:
        a = '%E' % x
        return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

    else:  # fixed format
        from decimal import Decimal
        return '%.1E' % Decimal(str(x))


def print_cpp_friendly(v, name, no_last_element_comma=True, print_content=True,
                       same_line=False, additional_info=False):
    """Generate string for and print a vector whose format is c/c++ friendly (easy copy & paste).

    :param ndarray v: vector values
    :param str name: given name
    :param bool no_last_element_comma: if we do not want to have comma added to the last element
    :param bool print_content: print the generated content
    :param bool same_line: printing values in the same line instead of new line
    :param bool additional_info: print additional info if there is any
    :return: content string
    """
    content = name
    disp_format = "{:.2e}"
    separator = ' ' if same_line else '\n'
    if additional_info:
        length = len(v)
        count_nonzeros = np.count_nonzero(v)
        content += ' ['
        content += 'size ' + str(length)
        content += ', nonzeros ' + str(count_nonzeros)
        content += ' (' + str(100 * count_nonzeros / length) + ' %)'
        content += ']' + ' '

    for i in range(len(v)):
        # last element does not need comma separation
        if no_last_element_comma:
            content += separator + disp_format.format(v[i]) + (',' if i != len(v) - 1 else '')
        else:
            content += separator + disp_format.format(v[i]) + ','

    if print_content:
        print(content)

    return content


def print_float4_group_array_hlsl_friendly(v1, v2, v3, v4):
    """Given a 4d array, pack the :math:`i_{th}` element of all arrays in one :code:`float4` tuple (hlsl friendly).

    Works with at least one not :code:`None` array (:code:`v1`). :code:`0` will be inserted if any of vectors \
    are :code:`None`.

    Output format:

    :code:`float4(v1_0, v2_0, v3_0, v4_0),`

    :code:`float4(v1_1, v2_1, v3_2, v4_3),`

    :code:`...`

    :param ndarray v1: 1d array, must be not :code:`None`
    :param ndarray v2: 1d array, if :code:`None`, :code:`0` will be inserted in :code:`float4()`
    :param ndarray v3: 1d array, if :code:`None`, :code:`0` will be inserted in :code:`float4()`
    :param ndarray v4: 1d array, if :code:`None`, :code:`0` will be inserted in :code:`float4()`
    :return: content string
    """

    assert v1 is not None, "The first array can not be None"

    size = v1.shape[0]

    if v2 is not None:
        assert size == v2.shape[0], "All arrays must have the same size"
    if v3 is not None:
        assert size == v3.shape[0], "All arrays must have the same size"
    if v4 is not None:
        assert size == v4.shape[0], "All arrays must have the same size"

    const_absent = 0 # for the None vectors insert this value

    content = ''
    for i in range(size):
        content += '\n'
        content += str_float4_hlsl_friendly(v1[i],
                                            v2[i] if v2 is not None else const_absent,
                                            v3[i] if v3 is not None else const_absent,
                                            v4[i] if v4 is not None else const_absent)

        content += '\n' + '' if i == size - 1 else',' # taking care of the , for the last row

    return content


def str_float4_hlsl_friendly(x1, x2, x3, x4):
    """Pack 4 scalars in :code:`float4()`.

    String format: :code:`float4(x1, x2, x3, x4)`

    :param float x1: scalar
    :param float x2: scalar
    :param float x3: scalar
    :param float x4: scalar
    :return: string of :code:`float4(x1, x2, x3, x4)`
    """
    disp_format = "{:.2e}"
    content = 'float4(' + disp_format.format(x1) + ', ' + \
              disp_format.format(x2) + ', ' + \
              disp_format.format(x3) + ', ' + \
              disp_format.format(x4) + ')'

    return content


def print_components_3d(cores, factors, decomp_method):
    """Print filters and their core values, only for the reduced case (separable filters).

    .. warning::
        Filters must be in (rank, filters) order.

    .. warning::
        Assumption: filter is symmetrical along all 3 dimensions.

    .. warning::
        Decomposition method must be :code:`SYM_CP_3D`. For :code:`TUCKER_3D` use the same order but with an extra \
        dimension.

        :code:`filter_1d = factors[0][rr, ::]` for *Tucker*,

        as opposed to

        :code:`filter_1d = factors[rr, ::]` for *SymCP*.

    :param ndarray cores: core values from *CP* decomposition
    :param ndarray factors: factor values from *CP* decomposition
    :param DecompMethod decomp_method: decomposition method
    """

    # only interested in SymCP. For Tucker use the same order but with an extra dimension:
    # filter_1d = factors[0][rr, ::] Tucker
    # as opposed to
    # filter_1d = factors[rr, ::] SymCP
    assert decomp_method == com.DecompMethod.SYM_CP_3D
    # ---- this order needs to be consistent with 2d --
    ranks, filter_size = np.shape(factors)

    # Factors
    for rr in range(0, int(ranks)):
        # ---- this order needs to be consistent with 2d --
        filter_1d = factors[rr, :].astype(float)
        print_cpp_friendly(filter_1d, name=f'//=======Rank {rr + 1} ======')

    # Core weights
    print_cpp_friendly(np.cbrt(cores), name="===== CUBIC root of the cores =====")


def print_filters(filters, same_line=True, additional_info=True):
    """Print filters for the reduced case (separable filters).

    .. warning::
        Filters must be in (rank, filters) order.

    :param ndarray filters: (rank, 1darray filter values)
    :param bool same_line: printing values in the same line instead of new line
    :param bool additional_info: print additional info if there is any
    """
    # ---- this order needs to be consistent with 2d --
    ranks, filter_size = np.shape(filters)

    # Ranked filters
    for rr in range(0, int(ranks)):
        # ---- this order needs to be consistent with 2d --
        f = filters[rr, :].astype(float)
        print_cpp_friendly(f, name=f'//=======Rank {rr + 1} ======',
                           same_line=same_line, additional_info=additional_info)


def export_components_csv(dim, solver_type, order, safe_rank, filters, modes, full_kernel):
    """Export full Poisson ernel, its modes and separable filters to :code:`.csv`. Find them in \
    :code:`data/preprocess/components`.

    :param SolverDimension dim: 2D or 3D
    :param PoissonSolverType solver_type: See :func:`PoissonSolverType`
    :param int order: filter order (target Jacobi iteration)
    :param int safe_rank: maximum acceptable rank
    :param ndarray filters: filter values
    :param List[ndarray] modes: list of modes with square matrix shape
    :param ndarray full_kernel: full Poisson kernel matrix
    """
    postfix = '_' + solver_type.name + '_' + dim.name + '_itr_' + str(order)
    file_format = '.csv'
    directory = path_preprocess_components

    # full kernel
    file_name = 'full_kernel'
    path = directory + file_name + postfix + file_format
    if dim == com.SolverDimension.D3:  # flattening - dont forget to reshape when loading in Mathematica
        full_kernel = np.ravel(full_kernel)
    np.savetxt(path, full_kernel, delimiter=',')

    # modes
    for rr in range(safe_rank):
        file_name = 'mode_' + str(rr + 1)
        path = directory + file_name + postfix + file_format
        component = modes[rr]
        if dim == com.SolverDimension.D3:  # flattening - dont forget to reshape when loading in Mathematica
            component = np.ravel(component)
        np.savetxt(path, component, delimiter=',')

    # filters
    for rr in range(safe_rank):
        file_name = 'filter_' + str(rr + 1)
        path = directory + file_name + postfix + file_format
        component = filters[rr]
        if dim == com.SolverDimension.D3:  # flattening - dont forget to reshape when loading in Mathematica
            component = np.ravel(component)
        np.savetxt(path, component, delimiter=',')


def make_options_batch_filter_generation(max_itr, max_rank, dx, dt, kappa, solver_type, kernel_type, zero_init, dim):
    """Generate options for the batch task.

    .. note::
        To see how to pack options, look at main demos, or see :func:`helper.commons.generic_options`.

    :param int max_itr: used to generate a database key in case of loading filters from database, otherwise it is simply
        the target iteration we are interested in.
    :param int max_rank: maximum cumulative rank, i.e. maximum rank to be included. must be safe.
    :param float dx: See :func:`OptionsKernel`
    :param float dt: See :func:`OptionsKernel`
    :param float kappa: See :func:`OptionsKernel`
    :param PoissonSolverType solver_type: See :func:`PoissonSolverType`
    :param PoissonKernelType kernel_type: See :func:`PoissonKernelType`
    :param bool zero_init: See :func:`OptionsPoissonSolver`
    :param SolverDimension dim: See :func:`SolverDimension`
    :return: packed options
    :rtype: OptionsGeneral
    """

    if dim == com.SolverDimension.D2:
        decomp_method = com.DecompMethod.SVD_2D
    elif dim == com.SolverDimension.D3:
        decomp_method = com.DecompMethod.SYM_CP_3D
    else:
        assert False, "Unknown Solver Dimension"

    # =============== Options ===============

    opts_solver = com.OptionsPoissonSolver(dim=dim, solver_type=solver_type, zero_init=zero_init)

    # --- functions Parameters ----
    # both for diffusion and pressure
    alpha = gen.compute_alpha(dx=dx, dt=dt, kappa=kappa, solver_type=solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=solver_type, dim=dim)
    opts_kernel = com.OptionsKernel(kernel_type=kernel_type, itr=max_itr, dx=dx, dt=dt, kappa=kappa,
                                alpha=alpha, beta=beta, clear_memo=True)

    # reduction
    truncation_method = com.TruncationMode.PERCENTAGE
    truncation_value = 0.
    opts_reduction = com.OptionsReduction(decomp_method=decomp_method,
                                          reduce=True,
                                          use_separable_filters=True,
                                          rank=max_rank,
                                          truncation_method=truncation_method,
                                          truncation_value=truncation_value,
                                          preserve_shape=True if truncation_method == com.TruncationMode.FIXED_THRESHOLD
                                          else False)

    # packing all..
    opts = com.OptionsGeneral(solver=opts_solver, kernel=opts_kernel, reduction=opts_reduction,
                              boundary=com.OptionsBoundary(), input=com.OptionsDataMatrix())

    return opts


def batch_generate_filters(opts, only_single_itr, single_itr_num, rank_filter_reorder_3d=True):
    """Generate a batch of Poisson filters for a range of target Jacobi iterations (filter orders). The function \
    supports a single filter order generation as well.

    :param OptionsGeneral opts:
    :param bool only_single_itr: do not generate the whole batch, only a single filter corresponding to \
        :code:`single_itr_num`.
    :param int single_itr_num: target Jacobi iteration (filter order) for the single filter export. \
        Make sure it is :code:`=< max_itr`
    :param bool rank_filter_reorder_3d: only 3d; if :code:`True`, change from default (filters, ranks) to \
        (ranks, filters) to make it consistent with 2d
    :return: list of filters corresponding to each target Jacobi iteration
    """
    # =============== Batch Generate ===============
    max_itr = opts.kernel.itr  # we are going to overwrite itr

    filters_iterations = []

    for i in range(max_itr):
        # if only interested in generating 1 filter, skip everything else
        if only_single_itr and (i + 1) != single_itr_num:
            continue

        print(f'generating ranked filters itr = {i + 1}')
        opts.kernel.itr = i + 1

        if opts.solver.dim == com.SolverDimension.D2:
            v_hor, v_ver, safe_rank = dec.poisson_filters_2d(opts)
            filters = v_hor
        elif opts.solver.dim == com.SolverDimension.D3:
            filters, low_rank, full_kernel, safe_rank = dec.poisson_filters_3d(opts, rank_filter_reorder=rank_filter_reorder_3d)
        else:
            assert False, "Unknown Solver Dimension"

        filters_iterations.append(filters)

    return filters_iterations


def generate_specs_info(opts, min_itr=1):
    """Generate a string of specification information of the kernel and solver properties.

    :param OptionsGeneral opts:
    :param int min_itr: only included when it is :code:`>0`, in which case it means we are batch generating filters
        between :code:`min_itr` and :code:`max_itr`.
    :return: spec string
    """

    alpha = gen.compute_alpha(dx=opts.kernel.dx, dt=opts.kernel.dt,
                              kappa=opts.kernel.kappa, solver_type=opts.solver.solver_type)
    beta = gen.compute_beta(alpha=alpha, solver_type=opts.solver.solver_type, dim=opts.solver.dim)

    # aux markers
    nl = '\n'
    cpp_comment = '//'
    new_entry = nl + cpp_comment

    content = new_entry + f'Specs:'
    content += new_entry + f'solver_dim = {opts.solver.dim.name}'
    content += new_entry + f'solver_type = {opts.solver.solver_type.name}'
    content += new_entry + f'kernel_type = {opts.kernel.kernel_type.name}'
    if min_itr > 0:
        content += new_entry + f'min_itr = {min_itr}' + ' (used in the database signature)'
    content += new_entry + f'max_itr = {opts.kernel.itr}' + ' (used in the database signature)'
    content += new_entry + f'max_rank = {opts.reduction.rank}'
    content += new_entry + f'dx = {opts.kernel.dx}'
    if opts.solver.solver_type == com.PoissonSolverType.FORWARD:  # diffusion
        content += new_entry + f'dt = {opts.kernel.dt}'
        content += new_entry + f'kappa = {opts.kernel.kappa}'
    content += new_entry + f'alpha = {alpha}'
    content += new_entry + f'beta = {beta}'
    content += new_entry + f'decomp_method = {opts.reduction.decomp_method.name}'

    return content

def convert_database_to_str(filters_iterations, specs_info_str, only_single_itr, single_itr_num, max_itr,
                            solver_type, pack_mode=com.ExportFormat.GROUP_RANK_FLOAT4_HLSL, min_itr=1):
    """Generate string format of the filters to be exported to C++ or HLSL firendly formats.

    :param List[int] filters_iterations: list of target Jacobi iterations (filter orders)
    :param str specs_info_str: specs header string
    :param bool only_single_itr: do not generate the whole set of filters, only a single filter corresponding to \
        :code:`single_itr_num`.
    :param int single_itr_num: target Jacobi iteration (filter order) for the single filter export. \
        Make sure it is :code:`=< max_itr`
    :param int max_itr: max target Jacobi iteration (filter order)
    :param PoissonSolverType solver_type: See :func:`PoissonSolverType`
    :param ExportFormat pack_mode: how to arrange the filter values
    :param int min_itr: min target Jacobi iteration (filter order) (**Default=** :code:`1`)
    :return: database as formatted string
    """

    print('converting database to string...')
    # aux markers
    nl = '\n'
    cpp_comment = '//'
    new_entry = nl + nl + cpp_comment

    content = '//auto-generated file from python - c++ friendly'
    content += nl + specs_info_str

    # format (iterations, ranks, filters)
    num_itr = len(filters_iterations)

    if min_itr == 0:
        min_itr = min_itr + 1

    if not only_single_itr:  # in case of assertion failure remember min_itr default used to be 0 when this code
        # was last working
        assert num_itr == (max_itr - min_itr + 1), "Iteration list and the max itr do not match"

    # ======= float4[ float4(rank 1 .. 4), float4(rank 1 .. 4), ..] packing hlsl friendly =======
    if pack_mode == com.ExportFormat.GROUP_RANK_FLOAT4_HLSL:
        for i in range(0, num_itr):
            itr_ = min_itr + i
            if only_single_itr:  # single filter export
                assert len(filters_iterations) == 1
                itr_ = single_itr_num

            content += new_entry + f'------------------itr = {itr_}------------------'
            filters = filters_iterations[i]
            # ---- this is the opposite order to the 3d filters --
            ranks, filter_size = np.shape(filters)
            content += nl + f'static const int {solver_type.name}_Itr_{itr_}_Filter_Size = {filter_size};'
            content += nl + f'static const int {solver_type.name}_Itr_{itr_}_Half_Filter_Size = {int(filter_size / 2)};'

            # Ranked filters

            # R14
            content += nl + f'static const float4 {solver_type.name}_Itr_{itr_}_R14_Filters[] = ' + '{'
            assert ranks >= 1  # need at least 1 rank
            f1 = filters[0, :].astype(float)  # rank 1
            f2 = None
            f3 = None
            f4 = None
            if ranks >= 2:
                f2 = filters[1, :].astype(float)  # rank 2
            if ranks >= 3:
                f3 = filters[2, :].astype(float)  # rank 3
            if ranks >= 4:
                f4 = filters[3, :].astype(float)  # rank 4
            content += nl + '//Rank 1  //Rank 2   //Rank 3   //Rank 4'
            content += print_float4_group_array_hlsl_friendly(v1=f1, v2=f2, v3=f3, v4=f4)
            content += '};'

            # R58
            if ranks >= 5:
                content += nl + f'static const float4 {solver_type.name}_Itr_{itr_}_R58_Filters[] = ' + '{'
                f5 = filters[4, :].astype(float)  # rank 5
                f6 = None
                f7 = None
                f8 = None
                if ranks >= 6:
                    f6 = filters[5, :].astype(float)  # rank 6
                if ranks >= 7:
                    f7 = filters[6, :].astype(float)  # rank 7
                if ranks == 8:
                    f8 = filters[7, :].astype(float)  # rank 8
                content += nl + '//Rank 5  //Rank 6   //Rank 7   //Rank 8'
                content += print_float4_group_array_hlsl_friendly(v1=f5, v2=f6, v3=f7, v4=f8)
                content += '};'

    # ======= float[ rank 1 floats.. // rank 2 floats.. // ..] packing 4 ranked filters sequentially =======
    elif pack_mode == com.ExportFormat.GROUP_RANK_SEQUENTIAL:
        for i in range(min_itr, max_itr):
            itr_ = i + 1
            if only_single_itr:  # single filter export
                assert len(filters_iterations) == 1
                itr_ = single_itr_num

            content += new_entry + f'------------------itr = {itr_}------------------'
            filters = filters_iterations[i]
            # ---- this is the opposite order to the 3d filters --
            ranks, filter_size = np.shape(filters)
            content += nl + f'static const int {solver_type.name}_Itr_{itr_}_Filter_Size = {filter_size};'
            content += nl + f'static const int {solver_type.name}_Itr_{itr_}_Half_Filter_Size = {int(filter_size / 2)};'

            # R14
            content += nl + f'static const float {solver_type.name}_Itr_{itr_}_R14_Filters[] = ' + '{'
            max_r14_rank = min(4, int(ranks))
            for rr in range(0, max_r14_rank):
                f = filters[rr, :].astype(float)
                content += nl + print_cpp_friendly(f, name=f'//=======Rank {rr + 1} ======', print_content=False,
                                                   no_last_element_comma=rr == max_r14_rank - 1)  # no last comma
            content += nl + '};'

            # R58
            if ranks > 4:
                content += nl + f'static const float {solver_type.name}_Itr_{itr_}_R58_Filters[] = ' + '{'
                max_r58_rank = min(8, int(ranks))
                for rr in range(4, max_r58_rank):
                    f = filters[rr, :].astype(float)
                    content += nl + print_cpp_friendly(f, name=f'//=======Rank {rr + 1} ======', print_content=False,
                                                       no_last_element_comma=rr == max_r58_rank - 1) # no last comma
                content += nl + '};'

    # ======= float4[ float4(rank 1 .. 4), float4(rank 1 .. 4), ..] packing hlsl friendly =======
    elif pack_mode == com.ExportFormat.GROUP_RANK_SEQUENTIAL:
        for i in range(min_itr, max_itr):
            itr_ = i + 1
            if only_single_itr:  # single filter export
                assert len(filters_iterations) == 1
                itr_ = single_itr_num

            content += new_entry + f'------------------itr = {itr_}------------------'
            filters = filters_iterations[i]
            # ---- this is the opposite order to the 3d filters --
            ranks, filter_size = np.shape(filters)
            content += nl + f'static const int {solver_type.name}_Itr_{itr_}_Filter_Size = {filter_size};'
            content += nl + f'static const int {solver_type.name}_Itr_{itr_}_Half_Filter_Size = {int(filter_size / 2)};'

            # Ranked filters
            for rr in range(0, int(ranks)):
                # ---- this is the opposite order to the 3d filters --
                f = filters[rr, :].astype(float)
                content += nl + f'static const float {solver_type.name}_Itr_{itr_}_R{rr + 1}_Filter[] = ' + '{'
                content += nl + print_cpp_friendly(f, name=f'//=======Rank {rr + 1} ======', print_content=False,
                                                   no_last_element_comma=True)
                content += nl + '};'

    else:
        assert False, "Unknown export format."

    print('conversion done.')
    return content



def generate_or_load_filters(opts, max_itr, load_database_from_file, only_single_itr, single_itr_num):
    """Generate a Poisson filter database or load it from file. \
    It can also be a single set of filter values only for one target iteration.

    :param OptionsGeneral opts:
    :param int max_itr: max target Jacobi iteration (filter order)
    :param bool load_database_from_file: load database; it has to exist
    :param bool only_single_itr: do not generate the whole set of filters, only a single filter corresponding to \
        :code:`single_itr_num`.
    :param int single_itr_num: target Jacobi iteration (filter order) for the single filter export. \
        Make sure it is :code:`=< max_itr`
    :return: list of filters for each target Jacobi iteration (filter order)
    """
    print('Dimension ' + opts.solver.dim.name)
    if load_database_from_file:
        database = load_filter_database(max_itr=max_itr, dx=opts.kernel.dx, kappa=opts.kernel.kappa, dim=opts.solver.dim,
                                        solver_type=opts.solver.solver_type, kernel_type=opts.kernel.kernel_type)
        if only_single_itr:
            assert len(database) >= single_itr_num
            # keep the list structure
            filters_iterations = []
            name = "arr_{}".format(single_itr_num - 1)
            filters_iterations.append(database[name]) # accessing the single filter
        else:
            filters_iterations = database  # explicit copy... can be droped though
    else:
        # generate for all iterations
        filters_iterations = batch_generate_filters(opts=opts, only_single_itr=only_single_itr,
                                                    single_itr_num=single_itr_num)
                                                    
        # Saving the database is not working
        # only save the database if it is batch, ban saving database when only dealing with a single filter export
        if not only_single_itr:
            save_filter_database(filters_iterations=filters_iterations, max_itr=max_itr,
                                 dx=opts.kernel.dx, kappa=opts.kernel.kappa, dim=opts.solver.dim,
                                 solver_type=opts.solver.solver_type, kernel_type=opts.kernel.kernel_type)

    return filters_iterations


def export_filters(max_itr, max_rank, dx, dt, dim, kappa, single_itr_num, solver_type, kernel_type,
                   load_database_from_file, only_single_itr, print_content=True,
                   export_pack_mode=com.ExportFormat.GROUP_RANK_FLOAT4_HLSL,
                   export_file_format=com.DataFileFormat.hlsli):
    """Generate or load filters, and export them to the desired file format (C++, HLSL, etc.)

    .. note::
        Check out :code:`data/preprocess/` to find exported files.

    :param int max_itr: max target Jacobi iteration (filter order)
    :param int max_rank: maximum cumulative rank, i.e. maximum rank to be included. must be safe.
    :param float dx: cell size
    :param float dt: diffusion timestep. See :func:`OptionsKernel`
    :param SolverDimension dim: 2D or 3D.
    :param float kappa: diffusivity. See :func:`OptionsKernel`
    :param int single_itr_num: target Jacobi iteration (filter order) for the single filter export. \
        Make sure it is :code:`=< max_itr`
    :param PoissonSolverType solver_type: See :func:`PoissonSolverType`
    :param PoissonKernelType kernel_type: See :func:`PoissonKernelType`
    :param bool load_database_from_file: load database; it has to exist
    :param bool only_single_itr: do not generate the whole set of filters, only a single filter corresponding to \
        :code:`single_itr_num`.
    :param bool print_content: also print in the terminal
    :param ExportFormat export_pack_mode: how to arrange the filter values (**Default=** :code:`GROUP_RANK_FLOAT4_HLSL`)
    :param DataFileFormat export_file_format: (**Default=** :code:`hlsli`)
    """

    # ==================== MAKE OPTIONS ====================
    # kernel/solver parameters
    zero_init = True  # is_ok_to_zero_init(solver_type=solver_type) # cannot use zero init for diffusion
    opts = make_options_batch_filter_generation(max_itr=max_itr, max_rank=max_rank,
                                                dx=dx, dt=dt, kappa=kappa,
                                                solver_type=solver_type,
                                                kernel_type=kernel_type,
                                                zero_init=zero_init,
                                                dim=dim)

    # ==================== GENERATE OR LOAD ====================
    filters_iterations = generate_or_load_filters(opts=opts, max_itr=max_itr,
                                                  load_database_from_file=load_database_from_file,
                                                  only_single_itr=only_single_itr, single_itr_num=single_itr_num)

    # ==================== OUTPUT ====================
    # Export / Print
    specs_info_str = generate_specs_info(opts=opts)
    content = convert_database_to_str(filters_iterations=filters_iterations,
                                      specs_info_str=specs_info_str,
                                      pack_mode=export_pack_mode,
                                      only_single_itr=only_single_itr,
                                      single_itr_num=single_itr_num,
                                      max_itr=max_itr,
                                      solver_type=solver_type)

    if print_content:
        print(content)

    # exporting to HLSL / C++
    path = generate_dump_path_filter_database(max_itr=max_itr, dx=dx, kappa=kappa, dim=dim, solver_type=solver_type,
                                              kernel_type=kernel_type, file_format=export_file_format,
                                              only_single_itr=only_single_itr,
                                              single_itr_num=single_itr_num)

    print('writing to file...')
    with open(path, 'w') as f:
        f.write(content)
    print('writing done.')
    print('file saved in ' + path)


def print_poisson_filters(dim, itr, max_rank, dx, dt, kappa, kernel_type, solver_type,
                          load_database_from_file, database_max_itr_key,
                          rank_filter_reorder=True):
    """Print Poisson filters for a desired target Jacobi iteration (filter order). Force generate the filters \
    if they do not exist in the database in case of loading from file.

    Output is already transposed by default to match (rank, filter) order: consistent with 2d. You can
    disable it by setting :code:`rank_filter_order=False`.

    :param int database_max_itr_key: if loading from database, the database must already exist in
        :code:`data/preprocess/filters/` with :code:`max_itr` key
    :param bool load_database_from_file: database file must already exist with a valid :code:`database_max_itr_key`
    :param PoissonSolverType solver_type: See :func:`PoissonSolverType`
    :param PoissonKernelType kernel_type: See :func:`PoissonKernelType`
    :param float kappa: diffusivity. See :func:`OptionsKernel`
    :param float dt: diffusion timestep. See :func:`OptionsKernel`
    :param float dx: cell size
    :param int max_rank: maximum cumulative rank, i.e. maximum rank to be included. must be safe.
    :param int itr: target Jacobi iteration (filter order)
    :param SolverDimension dim: 2D or 3D. See :func:`SolverDimension`
    :param bool rank_filter_reorder: if :code:`True`, change from (filters, ranks) to (ranks, filters);
        consistent with 2d
    """
    # ==================== MAKE OPTIONS ====================
    # kernel/solver parameters
    zero_init = True  # if not sure, use is_ok_to_zero_init(solver_type=solver_type) to see what to do.
    # cannot use zero init for diffusion.
    opts = make_options_batch_filter_generation(max_itr=itr, max_rank=max_rank,
                                                dx=dx, dt=dt, kappa=kappa,
                                                solver_type=solver_type,
                                                kernel_type=kernel_type,
                                                zero_init=zero_init,
                                                dim=dim)

    # ==================== GENERATE OR LOAD ====================
    # call the batch function but only for a single iteration
    filters_batch_iterations = generate_or_load_filters(opts=opts, only_single_itr=True, single_itr_num=itr,
                                                        max_itr=database_max_itr_key,
                                                        load_database_from_file=load_database_from_file)

    # The batch contains only one set of filters because we are interested in a single iteration (only_single_itr=True)
    filters = filters_batch_iterations[0]

    if filters is None:  # if for any reason filters are not already provided, force generate them
        print('WARNING: input filters are none... force generating them')
        if opts.solver.dim == com.SolverDimension.D2:
            v_hor, v_ver, safe_rank = dec.poisson_filters_2d(opts=opts)
            filters = v_hor
            # NOTE: in principle horizontal and vertical filters must be the same, but since we are using SVD
            # sometimes these are have flipped signs. Verify this before applying them. If so, use them separately.
        elif opts.solver.dim == com.SolverDimension.D3:
            filters, low_rank, full_kernel, safe_rank = dec.poisson_filters_3d(opts=opts,
                                                                               rank_filter_reorder=rank_filter_reorder)
            assert rank_filter_reorder, "print function only works with (rank, filter) order"
        else:
            assert False, "Unknown dimension"

    # ==================== PRINT ====================
    print_filters(filters=filters, same_line=True, additional_info=True)


def print_poisson_components_3d(opts, rank_filter_reorder=True):
    """Print reduced components of the Poisson kernel. Decomposition method can be Tucker or CP.

    :param OptionsGeneral opts:
    :param bool rank_filter_reorder: if :code:`True`, change from (filters, ranks) to (ranks, filters);
        consistent with 2d
    :return: filters or factors, depending on whether the cores are absorbed or not. (rank, filter) shape.
    """

    low_rank, cores, factors, full_kernel = dec.poisson_decomposition_components_3d(opts=opts)

    # True: change from (filters, ranks) to (ranks, filters): consistent with 2d

    if opts.reduction.decomp_method == com.DecompMethod.TUCKER_3D:
        factors = factors[0].transpose() if rank_filter_reorder else factors[0]

    elif opts.reduction.decomp_method == com.DecompMethod.SYM_CP_3D:
        factors = factors.transpose() if rank_filter_reorder else factors
    else:
        assert False, "Unknown decomposition method"

    # print filter values in a format that is easy to copy-paste into C/C++ arrays
    assert rank_filter_reorder, "print function only works with (rank, filter) order"
    print_components_3d(cores=cores, factors=factors, decomp_method=opts.reduction.decomp_method)

    return factors


def demo_generate_and_export_filters_database_2d(load_database_from_file=False, solver=com.PoissonSolverType.FORWARD, kernel=com.PoissonKernelType.STANDARD, max_it=100):
    """Example showcasing how to generate 2D filter database (or reload an existing database) and export them to file.
        
    :param bool load_database_from_file: if :code:`False`, forces it to generate everything, else database with a \
        proper key must be available. See  :code:`database_max_itr_key` in :func:`print_poisson_filters`.\
        (**Default=** :code:`False`).
    :param PoissonSolverType solver: See :func:`PoissonSolverType` (**Default=** :code:`com.PoissonSolverType.INVERSE`).
    :param PoissonKernelType kernel: See :func:`PoissonKernelType` (**Default=** :code:`com.PoissonKernelType.UNIFIED`).
    :param int max_it: max target Jacobi iteration (filter order) (**Default=** :code:`100`).
    """
    dim = com.SolverDimension.D2
    # database
    database_max_itr_key = max_it  # this database must already exist in data/preprocess/filters/ with this max itr key
    # 250 iteration seems like the maximum we can get before running into memory issues

    # single file handling
    only_single_itr = False
    single_itr_num = 1  # only used if we are interested in a single file generation or load when exporting
    assert single_itr_num <= database_max_itr_key

    export_filters(dim=dim,
                   max_itr=database_max_itr_key,
                   max_rank=8,
                   dx=1.0, dt=1.0, kappa=2.5e-2,
                   kernel_type=kernel,  # use UNIFIED for INVERSE and STANDARD for FORWARD
                   solver_type=solver,  # use UNIFIED for INVERSE and STANDARD for FORWARD
                   load_database_from_file=load_database_from_file,  # if False, forces it to generate everything,
                   # else database must be available.
                   #
                   # single filter export, better to load database if available
                   # for a single filter computation make sure its itr is less than or equal to max_itr
                   only_single_itr=only_single_itr, single_itr_num=single_itr_num,
                   export_pack_mode=com.ExportFormat.GROUP_RANK_FLOAT4_HLSL,
                   export_file_format=com.DataFileFormat.hlsli) # can be hlsl or c++


def demo_generate_and_export_filters_database_3d(load_database_from_file=False, solver=com.PoissonSolverType.INVERSE, kernel=com.PoissonKernelType.UNIFIED, max_it=100):
    """Example showcasing how to generate 3D filter database (or reload an existing database) and export them to file.

    :param bool load_database_from_file: if :code:`False`, forces it to generate everything, else database with a \
        proper key must be available. See  :code:`database_max_itr_key` in :func:`print_poisson_filters`.\
        (**Default=** :code:`False`).
    :param PoissonSolverType solver: See :func:`PoissonSolverType` (**Default=** :code:`com.PoissonSolverType.INVERSE`).
    :param PoissonKernelType kernel: See :func:`PoissonKernelType` (**Default=** :code:`com.PoissonKernelType.UNIFIED`).
    :param int max_it: max target Jacobi iteration (filter order) (**Default=** :code:`100`).
    """

    dim = com.SolverDimension.D3
    # database
    database_max_itr_key = max_it  # this database must already exist in data/preprocess/filters/ with this max itr key
    # 250 iteration seems like the maximum we can get before running into memory issues

    # single file handling
    only_single_itr = False
    single_itr_num = 1  # only used if we are interested in a single file generation or load when exporting
    assert single_itr_num <= database_max_itr_key

    export_filters(dim=dim,
                   max_itr=database_max_itr_key,
                   max_rank=8,
                   dx=1.0, dt=1.0, kappa=2.5e-2,
                   kernel_type=kernel,  # use UNIFIED for INVERSE and STANDARD for FORWARD
                   solver_type=solver,  # use UNIFIED for INVERSE and STANDARD for FORWARD
                   load_database_from_file=load_database_from_file,  # if False, forces it to generate everything,
                   # else database should be available.
                   #
                   # single filter export, better to load database if available
                   # for a single filter computation make sure its itr is less than or equal to max_itr
                   only_single_itr=only_single_itr, single_itr_num=single_itr_num,
                   export_pack_mode=com.ExportFormat.GROUP_RANK_FLOAT4_HLSL,
                   export_file_format=com.DataFileFormat.hlsli) # can be hlsl or c++


def demo_export_single_order_filters_2d(load_database_from_file=False, solver=com.PoissonSolverType.INVERSE, kernel=com.PoissonKernelType.UNIFIED, target_itr = 60):
    """Example showcasing exporting a single set of 2D Poisson filters to file for a desired target Jacobi iteration. \
    Try loading it from a database (if available) because it is faster. \
    Generate fresh filters by default instead of loading them.

    :param bool load_database_from_file: if :code:`False`, forces it to generate everything, else database with a \
        proper key must be available. See  :code:`database_max_itr_key` in :func:`print_poisson_filters`.\
        (**Default=** :code:`False`).
    :param PoissonSolverType solver: See :func:`PoissonSolverType` (**Default=** :code:`com.PoissonSolverType.INVERSE`).
    :param PoissonKernelType kernel: See :func:`PoissonKernelType` (**Default=** :code:`com.PoissonKernelType.UNIFIED`).
    :param int target_itr: target Jacobi iteration (filter order) (**Default=** :code:`60`).
    """
    dim = com.SolverDimension.D2
    # target_itr = 60  # Poisson filter order

    # database
    database_max_itr_key = 500  # if loading from database, this database must already exist in
    # data/preprocess/filters/ with this max itr key
    # 250 iteration seems like the maximum we can get before running into memory issues

    # single file handling
    only_single_itr = True
    single_itr_num = target_itr  # only used if we are interested in a single file generation or load when exporting
    assert single_itr_num <= database_max_itr_key

    export_filters(dim=dim,
                   max_itr=database_max_itr_key,
                   max_rank=8,
                   dx=1.0, dt=1.0, kappa=2.5e-2,
                   kernel_type=kernel,  # use UNIFIED for INVERSE and STANDARD for FORWARD
                   solver_type=solver,  # use UNIFIED for INVERSE and STANDARD for FORWARD
                   load_database_from_file=load_database_from_file,  # if False, forces it to generate everything,
                   # else database should be available.
                   #
                   # single filter export, better to load database if available
                   # for a single filter computation make sure its itr is less than or equal to max_itr
                   only_single_itr=only_single_itr, single_itr_num=single_itr_num,
                   export_pack_mode=com.ExportFormat.GROUP_RANK_FLOAT4_HLSL,
                   export_file_format=com.DataFileFormat.hlsli)  # can be hlsl or c++


def demo_export_single_order_filters_3d(load_database_from_file=False, solver=com.PoissonSolverType.INVERSE, kernel=com.PoissonKernelType.UNIFIED, target_itr = 60):
    """Example showcasing exporting a single set of 2D Poisson filters to file for a desired target Jacobi iteration. \
    Try loading it from a database (if available) because it is faster. \
    Generate fresh filters by default instead of loading them.

    :param bool load_database_from_file: if :code:`False`, forces it to generate everything, else database with a \
        proper key must be available. See  :code:`database_max_itr_key` in :func:`print_poisson_filters`.\
        (**Default=** :code:`False`).
    :param PoissonSolverType solver: See :func:`PoissonSolverType` (**Default=** :code:`com.PoissonSolverType.INVERSE`).
    :param PoissonKernelType kernel: See :func:`PoissonKernelType` (**Default=** :code:`com.PoissonKernelType.UNIFIED`).
    :param int target_itr: target Jacobi iteration (filter order) (**Default=** :code:`60`).
    """

    dim = com.SolverDimension.D3
    # target_itr = 60  # Poisson filter order

    # database
    database_max_itr_key = 100  # if loading from database, this database must already exist in
    # data/preprocess/filters/ with this max itr key

    # single file handling
    only_single_itr = True
    single_itr_num = target_itr  # only used if we are interested in a single file generation or load when exporting
    assert single_itr_num <= database_max_itr_key

    export_filters(dim=dim,
                   max_itr=database_max_itr_key,
                   max_rank=8,
                   dx=1.0, dt=1.0, kappa=2.5e-2,
                   kernel_type=kernel,  # use UNIFIED for INVERSE and STANDARD for FORWARD
                   solver_type=solver,  # use UNIFIED for INVERSE and STANDARD for FORWARD
                   load_database_from_file=load_database_from_file,  # if False, forces it to generate everything,
                   # else database should be available.
                   #
                   # single filter export, better to load database if available
                   # for a single filter computation make sure its itr is less than or equal to max_itr
                   only_single_itr=only_single_itr, single_itr_num=single_itr_num,
                   export_pack_mode=com.ExportFormat.GROUP_RANK_FLOAT4_HLSL,
                   export_file_format=com.DataFileFormat.hlsli)  # can be hlsl or c++


def demo_print_filters_2d():
    """Example showcasing generating and printing 2D Poisson filters."""
    dim = com.SolverDimension.D2
    itr = 50  # target Jacobi iteration (filter order)

    # database
    # You have the choice to either fresh generate filters or load them from the database
    load_database_from_file = False  # if False, forces it to generate everything, else database should be available.
    database_max_itr_key = 500  # if loading from database, this database must already exist in
    # data/preprocess/filters/ with this max itr key

    print_poisson_filters(dim=dim, itr=itr, max_rank=8, dx=1.0,
                          dt=1.0, kappa=2.5e-2,  # dt and kappa are only used for diffusion (forward poisson)
                          # otherwise will be ignored for inverse poisson
                          kernel_type=com.PoissonKernelType.UNIFIED, solver_type=com.PoissonSolverType.INVERSE,
                          load_database_from_file=load_database_from_file, database_max_itr_key=database_max_itr_key,
                          rank_filter_reorder=True)


def demo_print_filters_3d():
    """Example showcasing generating and printing 3D Poisson filters."""
    dim = com.SolverDimension.D3
    itr = 50  # target Jacobi iteration (filter order)

    # database
    # You have the choice to either fresh generate filters or load them from the database
    load_database_from_file = True  # if False, forces it to generate everything, else database should be available.
    database_max_itr_key = 100  # if loading from database, this database must already exist in
    # data/preprocess/filters/ with this max itr key

    print_poisson_filters(dim=dim, itr=itr, max_rank=8, dx=1.0,
                          dt=1.0, kappa=2.5e-2,  # dt and kappa are only used for diffusion (forward poisson)
                          # otherwise will be ignored for inverse poisson
                          kernel_type=com.PoissonKernelType.UNIFIED, solver_type=com.PoissonSolverType.INVERSE,
                          load_database_from_file=load_database_from_file, database_max_itr_key=database_max_itr_key,
                          rank_filter_reorder=True)


def demo_generate_and_export_single_order_filters_2d(target_itr = 120):
    """Example showcasing generating and exporting 2D Poisson filters."""
    dim = com.SolverDimension.D2
    # target_itr = 120  # target Jacobi iteration (filter order)
    max_rank = 8
    # ==================== MAKE OPTIONS ====================
    kernel_type = com.PoissonKernelType.STANDARD
    solver_type = com.PoissonSolverType.INVERSE
    dx = 1.0
    dt = 1.0  # only if computing for diffusion (forward poisson)
    kappa = 2.5e-2  # only if computing for diffusion (forward poisson)
    # kernel/solver parameters
    zero_init = True  # is_ok_to_zero_init(solver_type=solver_type) # cannot use zero init for diffusion
    opts = make_options_batch_filter_generation(max_itr=target_itr, max_rank=max_rank,
                                                dx=dt, dt=dt, kappa=kappa,
                                                kernel_type=kernel_type,
                                                solver_type=solver_type,
                                                zero_init=zero_init,
                                                dim=dim)

    # ==================== GENERATE OR LOAD ====================
    v_hor, v_ver, safe_rank = dec.poisson_filters_2d(opts=opts)
    filters = v_hor
    # NOTE: in principle horizontal and vertical filters must be the same, but since we are using SVD
    # sometimes these are have flipped signs. Verify this before applying them. If so, use them separately.

    filters_iterations = []
    filters_iterations.append(filters)
    # ==================== OUTPUT ====================
    # Export / Print
    specs_info_str = generate_specs_info(opts=opts)
    content = convert_database_to_str(filters_iterations=filters_iterations,
                                      specs_info_str=specs_info_str,
                                      pack_mode=com.ExportFormat.GROUP_RANK_FLOAT4_HLSL,
                                      only_single_itr=True,
                                      single_itr_num=target_itr,
                                      max_itr=target_itr,
                                      solver_type=solver_type)

    print(content)

    # exporting to HLSL / C++
    path = generate_dump_path_filter_database(file_format=com.DataFileFormat.hlsli,
                                              max_itr=target_itr, dx=dx, kappa=kappa, dim=dim, solver_type=solver_type,
                                              kernel_type=kernel_type,
                                              only_single_itr=True,
                                              single_itr_num=target_itr)

    print('writing to file...')
    with open(path, 'w') as f:
        f.write(content)
    print('writing done.')
    print('file saved in ' + path)


def demo_generate_and_export_single_order_filters_3d(target_itr = 60):
    """Example showcasing generating and exporting 3D Poisson filters."""
    dim = com.SolverDimension.D3

    # target_itr = 60  # target Jacobi iteration (filter order)
    max_rank = 8
    # ==================== MAKE OPTIONS ====================
    # +++++++++++
    kernel_type = com.PoissonKernelType.STANDARD
    solver_type = com.PoissonSolverType.INVERSE

    dx = 1.0  # for both inverse and forward poisson
    dt = 1.0  # only if computing for diffusion (forward poisson)
    kappa = 2.5e-2  # only if computing for diffusion (forward poisson)
    # kernel/solver parameters
    zero_init = True  # you can use `is_ok_to_zero_init(solver_type=solver_type)` # cannot use zero init for diffusion
    opts = make_options_batch_filter_generation(max_itr=target_itr, max_rank=max_rank,
                                                dx=dt, dt=dt, kappa=kappa,
                                                kernel_type=kernel_type,
                                                solver_type=solver_type,
                                                zero_init=zero_init,
                                                dim=dim)

    # ==================== GENERATE OR LOAD ====================
    filters, low_rank, full_kernel, safe_rank = dec.poisson_filters_3d(opts=opts, rank_filter_reorder=True)
    filters_iterations = []
    filters_iterations.append(filters)
    # ==================== OUTPUT ====================
    # Export / Print
    specs_info_str = generate_specs_info(opts=opts)
    content = convert_database_to_str(filters_iterations=filters_iterations,
                                      specs_info_str=specs_info_str,
                                      pack_mode=com.ExportFormat.GROUP_RANK_FLOAT4_HLSL,
                                      only_single_itr=True,
                                      single_itr_num=target_itr,
                                      max_itr=target_itr,
                                      solver_type=solver_type)

    print(content)

    # exporting to HLSL / C++
    path = generate_dump_path_filter_database(file_format=com.DataFileFormat.hlsli,
                                              max_itr=target_itr, dx=dx, kappa=kappa, dim=dim, solver_type=solver_type,
                                              kernel_type=kernel_type,
                                              only_single_itr=True,
                                              single_itr_num=target_itr)

    print('writing to file...')
    with open(path, 'w') as f:
        f.write(content)
    print('writing done.')
    print('file saved in ' + path)


if __name__ == "__main__":
    # UNCOMMENT TO RUN THE DEMOS...

    # A) =========================== Export Filters: Basic Functions =======================

    # 1. Print the filters to the console.
    # Inside, you have the choice to either fresh generate filters or load them from the database.
    # By default, it generates the filters

    # 1.1
    # UNCOMMENT------------------------------------------------
    # demo_print_filters_2d()

    # 1.2
    # UNCOMMENT------------------------------------------------
    # demo_print_filters_3d()

    # 2. Generate filters for a given order (target iteration) and export them to hlsli or cpp

    # 2.1
    # UNCOMMENT------------------------------------------------
    # demo_generate_and_export_single_order_filters_2d()

    # 2.2
    # UNCOMMENT------------------------------------------------
    # demo_generate_and_export_single_order_filters_3d()

    # B) =========================== Export Filters: Working With Database ========================
    # 3. Computes a database of filters between a min_itr and max_itr range

    # 3.1
    # UNCOMMENT------------------------------------------------
    demo_generate_and_export_filters_database_2d()

    # 3.2
    # UNCOMMENT------------------------------------------------
    # demo_generate_and_export_filters_database_3d()

    # 4. Exporting a poisson filters for a given order (target iteration). You have the
    # option to generate them fresh by load_database_from_file=False
    # or loading them from an already existing database. Check the code for more info on
    # how to use the database

    # 4.1
    # UNCOMMENT------------------------------------------------
    # demo_export_single_order_filters_2d(load_database_from_file=True)

    # 4.2
    # UNCOMMENT------------------------------------------------
    # demo_export_single_order_filters_3d(load_database_from_file=True)
