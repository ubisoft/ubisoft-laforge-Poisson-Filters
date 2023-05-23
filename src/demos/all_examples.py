""" :author: Shahin (Amir Hossein) Rabbani
    :contact: shahin.rab@gmail.com
    :copyright: See :ref:`License <license_page>`

    .. _quick_access_demos_header:

    Quick access to all demos
    ---------------------------

    *Example*: To print 3D Poisson filters, from :code:`__main__` run:

    .. code-block:: python
        :caption: Print 3D Poisson filters

        run_demo(which=DemoName.IO_PRINT_FILTERS_3D)

    Check out :func:`DemoName` for all demos. See each method in :code:`src.demos` for explanation.

    .. warning::
        If calling the demos from other paths than the current module, you might get errors with data paths when loading
        filter or mask data. Adjust your system path accordingly.

    Technicals
    ===========

    - In all demos try to use odd numbers for the domain size because the data domain needs to be symmetrical for most of \
        the demos to work properly.
    - A rank range of :math:`6 \\cdots 8` for 3D, and :math:`1 \\cdots 4` for 2D is usually sufficient \
        regardless of the truncation value.
    - Observe when generating the kernel using :code:`com.PoissonSolverType.INVERSE`, you should multiply the \
        right hand side :math:`b` in :math:`Ax=b` by :math:`-1`. When using :code:`com.PoissonSolverType.FORWARD` this \
        multiplication is not necessary.

        .. note::

            If going with :code:`UNIFIED` kernel instead of :code:`STANDARD` you need the following sign flip in the
            *RHS* data to make it work. This is the only downside of benefiting from a :code:`UNIFIED` kernel.

            .. code-block:: python

                data_domain *= -1. if opts.kernel.kernel_type == com.PoissonKernelType.UNIFIED else 1.

        See :func:`demos.reductions2d.demo_compare_standard_and_unified_kernels_unreduced_2d`.

"""

# Add project root to path for local import to work (in case no IDE is used)
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import src.demos.reductions2d as red2d
import src.demos.reductions3d as red3d
import src.helper.iohandler as io
import src.helper.commons as com
import src.demos.convergence as conv
import src.demos.boundary as bo
import src.functions.decompositions as dec
from enum import Enum


class DemoName(Enum):
    # =========== Reductions 2D
    REDUCTIONS_2D_PLOT_KERNEL_MODES_FILTERS_CSV_EXPORT = 1
    REDUCTIONS_2D_COMPARE_UNIFIED_STANDARD_KERNELS = 2

    # =========== Reductions 3D
    REDUCTIONS_3D_PLOT_KERNEL_MODES_FILTERS_CSV_EXPORT = 3
    REDUCTIONS_3D_PLOT_FILTERS_TRUNCATION_BEAUTIFIED = 4

    # =========== Boundary
    BOUNDARY_TILED_MIRROR = 5
    BOUNDARY_MULTI_MODAL_COMPLEX_OBJECT = 6
    BOUNDARY_VERSATILE_ERROR_SCENARIOS = 7

    # =========== Convergence
    CONVERGENCE_COMPARE_3_METHODS_INFINITE_DOMAIN = 8
    CONVERGENCE_COMPARE_3_METHODS_WALL_NEUMANN_BOUNDARY = 9
    CONVERGENCE_ADAPTIVE_TRUNCATION_ANALYSIS = 10
    CONVERGENCE_RESIDUAL_COMPARISON_INFINITE_DOMAIN = 11

    # =========== IO / EXPORTS
    IO_PRINT_FILTERS_2D = 12
    IO_PRINT_FILTERS_3D = 13
    IO_GENERATE_FILTERS_EXPORT_HLSLI_2D = 14
    IO_GENERATE_FILTERS_EXPORT_HLSLI_3D = 15
    IO_DATABASE_GENERATE_2D = 16
    IO_DATABASE_GENERATE_3D = 17
    IO_DATABASE_SINGLE_FILTER_2D = 18
    IO_DATABASE_SINGLE_FILTER_3D = 19

    # =========== DECOMPOSITION
    DECOMPOSITION_EIGENVALUE_ABSORPTION_TEST_2D = 20


def run_demo(which):
    """
    :param DemoName which:
    """
    # ============================================ IO / EXPORTS: SIMPLE ============================================

    if which == DemoName.IO_PRINT_FILTERS_2D:
        # 2D - generate and print filters
        io.demo_print_filters_2d()

    elif which == DemoName.IO_PRINT_FILTERS_3D:
        # 3D - generate and print filters
        io.demo_print_filters_3d()

    elif which == DemoName.IO_GENERATE_FILTERS_EXPORT_HLSLI_2D:
        # 2D - Generate filters for a given order (target iteration) and export them to hlsli or cpp
        io.demo_generate_and_export_single_order_filters_2d()

    elif which == DemoName.IO_GENERATE_FILTERS_EXPORT_HLSLI_3D:
        # 3D - Generate filters for a given order (target iteration) and export them to hlsli or cpp
        io.demo_generate_and_export_single_order_filters_3d()

    # ====================================== IO / EXPORTS: Working With Database ======================================
    # Export filters (HLSL), option to generate or load from the database (npz).

    elif which == DemoName.IO_DATABASE_GENERATE_2D:
        # 2D - Generate hlsli database
        io.demo_generate_and_export_filters_database_2d(load_database_from_file=False, max_it=500)
        # for solver in com.PoissonSolverType:
        #     for kernel in com.PoissonKernelType:
        #         for max_it in [100, 500]:
        #             if com.PoissonSolverType(solver.value) != com.PoissonSolverType.UNKNOWN and com.PoissonKernelType(kernel.value)  != com.PoissonKernelType.UNKNOWN:
        #                 io.demo_generate_and_export_filters_database_2d(load_database_from_file=False, solver=com.PoissonSolverType(solver.value), kernel=com.PoissonKernelType(kernel.value), max_it=max_it)

    elif which == DemoName.IO_DATABASE_GENERATE_3D:
        # 3D - Generate hlsli database
        io.demo_generate_and_export_filters_database_3d(load_database_from_file=False, max_it=100)
        # for solver in com.PoissonSolverType:
        #     for kernel in com.PoissonKernelType:
        #         for max_it in [100]:
        #             if com.PoissonSolverType(solver.value) != com.PoissonSolverType.UNKNOWN and com.PoissonKernelType(kernel.value)  != com.PoissonKernelType.UNKNOWN:
        #                 io.demo_generate_and_export_filters_database_3d(load_database_from_file=False, solver=com.PoissonSolverType(solver.value), kernel=com.PoissonKernelType(kernel.value), max_it=max_it)

    elif which == DemoName.IO_DATABASE_SINGLE_FILTER_2D:
        # 2D - Loading single order filters from database. Force generate them by 'load_database_from_file=False'
        io.demo_export_single_order_filters_2d(load_database_from_file=False, target_itr=120)

    elif which == DemoName.IO_DATABASE_SINGLE_FILTER_3D:
        # 3D - Loading single order filters from database. Force generate them by 'load_database_from_file=False'
        io.demo_export_single_order_filters_3d(load_database_from_file=False, target_itr=60)

    # ============================================ Reductions 2D ============================================

    elif which == DemoName.REDUCTIONS_2D_PLOT_KERNEL_MODES_FILTERS_CSV_EXPORT:
        # Compute the full Poisson kernel, reduce it based on a desired rank, and plot
        # the full kernel, the reduced one, the modes and separable filters. Option to export as csv.
        red2d.plot_and_csv_export_truncated_filters_and_modes_2d(order=50, rank=8, export_csv=True,
                                                                 filter_trim_zeros=True)

    elif which == DemoName.REDUCTIONS_2D_COMPARE_UNIFIED_STANDARD_KERNELS:
        # Comparing standard and unified kernel types
        red2d.demo_compare_standard_and_unified_kernels_unreduced_2d(order=50)

    # ============================================ Reductions 3D ============================================

    elif which == DemoName.REDUCTIONS_3D_PLOT_KERNEL_MODES_FILTERS_CSV_EXPORT:
        # Compute and plot all components of 3D filter decomposition : CSV export is available,
        red3d.plot_and_csv_export_truncated_filters_and_modes_3d(order=30, rank=8, export_csv=True,
                                                                 filter_trim_zeros=True)

    elif which == DemoName.REDUCTIONS_3D_PLOT_FILTERS_TRUNCATION_BEAUTIFIED:
        # Plot beautified adaptive truncation of filters
        red3d.demo_plot_filters_adaptive_truncation_beautified_3d(order=60, rank=8)

    # ============================================ Boundary ============================================

    elif which == DemoName.BOUNDARY_TILED_MIRROR:
        # Enforcing Neumann boundary using reflection tiles for the Poisson solve suing Poisson filters convolution
        bo.demo_wall_neumann_tiled_mirrored_2d(order=15, rank=8)

    elif which == DemoName.BOUNDARY_MULTI_MODAL_COMPLEX_OBJECT:
        # Break down of a Poisson filters solve with Neumann boundary for individual rank contributions.
        # With wall and complex object Neumann boundary treatment.
        bo.demo_multi_modal_solve_neumann_complex_object_2d(order=60, rank=4)

    elif which == DemoName.BOUNDARY_VERSATILE_ERROR_SCENARIOS:
        # Experimenting with various scenarios with different truncation values and Poisson filters order
        # (target Jacobi iteration).
        # Solving with current settings might take a while, be patient!
        bo.demo_versatile_error_scenarios_neumann_2d()

    # ============================================ Convergence ============================================

    elif which == DemoName.CONVERGENCE_COMPARE_3_METHODS_INFINITE_DOMAIN:
        # Sanity check: compare jacobi solutions; no boundary condition: infinite domain
        conv.demo_3methods_comparison_no_bc_2d(order=50, rank=8, use_full_poisson_kernel=True)

    elif which == DemoName.CONVERGENCE_COMPARE_3_METHODS_WALL_NEUMANN_BOUNDARY:
        # Sanity check: testing 3 methods with Neumann bc
        conv.demo_3methods_comparison_with_wall_neumann_bc_2d(order=20, rank=8)

    elif which == DemoName.CONVERGENCE_ADAPTIVE_TRUNCATION_ANALYSIS:
        # Plot adaptive truncation info
        conv.demo_adaptive_truncation_analysis_2d(order=100, rank=8)

    elif which == DemoName.CONVERGENCE_RESIDUAL_COMPARISON_INFINITE_DOMAIN:
        # Compare Jacobi solution and its residual in the *matrix-form* to that of the Poisson kernel convolution.
        # Option to use full kernel (machine precision error is expected compared to Jacobi), or
        # truncated Poisson filters (use_full_poisson_kernel=False)
        conv.demo_residual_comparison_jacobi_poisson_filters_infinite_domain(order=60, rank=8, truncation_value=1e-3,
                                                                             use_full_poisson_kernel=True)

    # ============================================ Decomposition ============================================

    elif which == DemoName.DECOMPOSITION_EIGENVALUE_ABSORPTION_TEST_2D:
        # testing absorbing the eigenvalues into 2D filters
        dec.demo_test_svd_absorb()


if __name__ == '__main__':
    # run_demo(which=DemoName.REDUCTIONS_2D_PLOT_KERNEL_MODES_FILTERS_CSV_EXPORT)
    # run_demo(which=DemoName.REDUCTIONS_2D_COMPARE_UNIFIED_STANDARD_KERNELS)
    # run_demo(which=DemoName.REDUCTIONS_3D_PLOT_KERNEL_MODES_FILTERS_CSV_EXPORT)
    # run_demo(which=DemoName.REDUCTIONS_3D_PLOT_FILTERS_TRUNCATION_BEAUTIFIED)
    # run_demo(which=DemoName.BOUNDARY_TILED_MIRROR)
    # run_demo(which=DemoName.BOUNDARY_MULTI_MODAL_COMPLEX_OBJECT)
    # run_demo(which=DemoName.BOUNDARY_VERSATILE_ERROR_SCENARIOS)
    # run_demo(which=DemoName.CONVERGENCE_COMPARE_3_METHODS_INFINITE_DOMAIN)
    # run_demo(which=DemoName.CONVERGENCE_COMPARE_3_METHODS_WALL_NEUMANN_BOUNDARY)
    # run_demo(which=DemoName.CONVERGENCE_ADAPTIVE_TRUNCATION_ANALYSIS)
    # run_demo(which=DemoName.CONVERGENCE_RESIDUAL_COMPARISON_INFINITE_DOMAIN)
    # run_demo(which=DemoName.IO_PRINT_FILTERS_2D)
    # run_demo(which=DemoName.IO_PRINT_FILTERS_3D)
    # run_demo(which=DemoName.IO_GENERATE_FILTERS_EXPORT_HLSLI_2D)
    # run_demo(which=DemoName.IO_GENERATE_FILTERS_EXPORT_HLSLI_3D)
    # run_demo(which=DemoName.IO_DATABASE_GENERATE_2D)
    # run_demo(which=DemoName.IO_DATABASE_GENERATE_3D)
    run_demo(which=DemoName.IO_DATABASE_SINGLE_FILTER_2D)
    run_demo(which=DemoName.IO_DATABASE_SINGLE_FILTER_3D)
    # run_demo(which=DemoName.DECOMPOSITION_EIGENVALUE_ABSORPTION_TEST_2D)
    
    # # test all demo
    # for data in DemoName:
    #     # print("run_demo(which=DemoName.{})".format(data.name))
    #     print("Running demo: {}".format(data.name))
    #     run_demo(which=DemoName(data.value))
