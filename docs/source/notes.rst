
.. _notes_page:

Notes
======

Read these notes to help understand the method and how it is best implemented.

Technical
-------------------

See the modules technical notes for:

   - :ref:`Quick Access To All Demos<quick_access_demos_header>` : domain data requirements, sign flip when using the
     `unified` Poisson kernel, and the choice of rank for 2D and 3D.

   - :ref:`Generator <generator_method>`: *unified* vs *standard* kernel types, application notes on choosing
     :math:`\alpha` and :math:`\beta` when generating kernels, and warm starting.

   - :ref:`Decomposition <decomposition_method>`: 2D (*SVD*) vs 3D (*Symmetric CP*) decompositions, absorbing eigenvalues,
     the meaning and usage of rank for tensors in 3D, and what a *safe rank* means.

   - :ref:`Mathlib <mathlib_technicals>`: *inverse* vs *forward* Poisson equations, *matrix-form* vs *vector-form* linear
     solvers, Poisson filter computation and convolution.

   - :ref:`Convergence <convergence_notes>`: how adaptive truncation works.

   - :ref:`IO Handler <iohandler_notes>`: how to load pre-generated Poisson filters database, *csv* and *hlsli/c++* exports.


General
---------------

- This work is based on MAC-Grids.

- In the context of kernel and filter generation we use **target Jacobi iteration** and **Poisson filters order**
  interchangeably in the code. *Order* means the target Jacobi iteration of a filter/kernel.

- See :ref:`this section<mathlib_convolution_order>` in :ref:`Mathlib Technicals <mathlib_technicals>` for the order
  of convolution when applying Poisson filters in 2D and 3D.

- When solving an **inverse** Poisson equation (like Poisson pressure) if going with :code:`UNIFIED` kernel instead of
  :code:`STANDARD` you need the following sign flip on the right hand side :math:`b` in :math:`Ax=b` to make it work.
  This is the only downside of benefiting from a :code:`UNIFIED` kernel.

  .. code-block:: python

      data_domain *= -1. if opts.kernel.kernel_type == com.PoissonKernelType.UNIFIED else 1.

- The Jacobi convergence curve provides a lower bound on the Poisson filter possible solutions. This means
  if there is no numerical loss due to filter reduction or truncation, Poisson filters must exactly match
  the convergence behaviour of Jacobi within machine precision,
  which will be the best case for numerical quality, and worst case performance-wise.
  Any reduction or truncation results in degradation of the convergence behaviour in exchange for performance improvement.

- *Problem with SymCP Eigenvalues in tensor decomposition*: contrary to 2D, eigenvalues are not necessarily sorted
  in terms of variance explained in 3D. When using grouped ranks, solutions might get worse before getting better
  when including higher ranks. See *Safe Rank* explanation in :ref:`Decomposition Technicals <decomposition_method>`
  for more details on how rank definition is different in 3D.

- Remember there is no consensus on the definition of tensor eigenvalues. We use the definition that best suits
  our case, and remains consistent for both 2D and 3D filter computations. Such definition is only possible because
  the Poisson kernel is symmetric. See `"All Real Eigenvalues of Symmetric Tensors", 2014 <https://arxiv.org/abs/1403.3720>`_
  and `"A Spectral Theory for Tensors", 2010 <https://arxiv.org/abs/1008.2923>`_ .

- Eigenvalues from *SymCP* are real (no complex part).

Potential Applications and Future Work
----------------------------------------

- **Multi-grid smoother**: multi-grid solvers have smoothing steps that are typically iterative. The performance of multi-grid
  solvers can be improved by using high-order Poisson filters in the smoothing step.
  The results will be interesting to observe as we might only need one or very a few multi-grid cycle iterations to converge.

- **Dirichlet boundary**: Current method only supports Neumann boundaries. Addressing Dirichlet boundary treatment with Poisson
  filters will be highly desirable to expand the scope of our method to many application domains that need Dirichlet boundary treatment.

  For instance, one can consider using Poisson filters for *Seamless Cloning*. You can check out
  `Fourier implementation of Poisson image editing, 2012 <https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/FourierPoissonEditing.pdf>`_
  to see how the Dirichlet boundary can be transformed into a Neumann boundary problem when using spectral methods.
  This method is consistent with the spectral point of view that Poisson filters is based on.

- **Improving Mirror Marching algorithm**: when dealing with Neumann boundaries, current method only marches parallel to
  spatial axes at no angle, which results in perfect solutions around walls but non-ideal artifacts around corners.
  A *Scattering Marching* algorithm that uses the object surface normals would likely remedy this artifact.

Suggested Reads/Codes
----------------------
- `Why Poisson's equation is so important <https://mattferraro.dev/posts/poissons-equation?fbclid=IwAR01mUB0wg4O7hNVjYPWqBAC0mB2K1mNFZyIqgQ-7aIT8fbh5zvERblCdf4>`_
- Multi-grid sources for the interested reader in applying Poisson filters as smoother:

   - **Reads**
      - `Book of numerical recipes, section 20.6 <http://numerical.recipes/book/book.html>`_
      - `An overview of multigrid methods, 2014 <https://www.wias-berlin.de/people/john/LEHRE/MULTIGRID/multigrid.pdf>`_
      - `A simple multigrid scheme for solving the Poisson equation with arbitrary domain boundaries, 2011 <https://arxiv.org/pdf/1104.1703.pdf>`_
      - `A parallel multigrid Poisson solver for fluids simulation on large grids, 2010 <https://www.math.ucla.edu/~jteran/papers/MST10.pdf>`_
      - `OpenMG: A new multigrid implementation in Python, 2012 <https://conference.scipy.org/proceedings/scipy2012/pdfs/tom_bertalan.pdf>`_
      - `Programming of multigrid methods <https://www.math.uci.edu/~chenlong/226/MGcode.pdf>`_

   - **Codes**

      - `Sample Python code <https://julianroth.org/documentation/multigrid/index.html>`_
      - `C/C++ NVIDIA GPU Accelerated AGMX <https://github.com/NVIDIA/AMGX>`_ and `its reference <https://github.com/NVIDIA/AMGX/blob/main/doc/AMGX_Reference.pdf>`_
      - `PyAMG <https://github.com/pyamg/pyamg>`_ and `PyAMGX <https://github.com/shwina/pyamgx>`_
      - `Python Hydro <https://python-hydro.github.io/pyro2/multigrid_basics.html>`_
      - `Python Geometric Multigrid <https://github.com/AbhilashReddyM/GeometricMultigrid>`_
