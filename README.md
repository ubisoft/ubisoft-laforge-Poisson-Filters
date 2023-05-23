# Compact Poisson Filters for Fast Fluid Simulation

<img src="docs/source/images/cube_field.gif" width=750></img>
<img src="docs/source/images/teaser_filters.png" width=750></img>

*Compact Poisson filters* are powerful alternatives to iterative linear methods to solve \
Poisson's equation $\nabla^{2} \phi = f$, particularly for real-time applications, where interactivity is of high importance.

Poisson filter-based method is a one-shot, convolutional linear solver that balances between the strengths of spectral and
iterative methods with a parallel implementation on GPU. Our solver precludes the need for careful preconditioning,
scales favorably with the size of the problem, handles Neumann boundary conditions, and is a drop-in replacement with
controllable error tolerance for existing solvers in high-performance settings.

While the main application presented in this work is enhancing the 
runtime performance of Eulerian fluids simulation, Poisson filters have the potential to be used in other domains 
that require a fast solution to Poisson's equation, 
such as image processing, cloth simulation, Newtonian gravity, and electrostatics. 

# Paper
["*Compact Poisson Filters for Fast Fluid Simulation*"](https://dl.acm.org/doi/10.1145/3528233.3530737),
ACM SIGGRAPH 2022 Conference.

You can also find the paper and its supplements in `docs/paper/`.

You might want to choose which experiments to run in `all_examples.py`.

# <a name="documentation"></a> Code Documentation
Download the contents of `docs/build` to be able to view and browse the `html` pages properly.

Start with `docs/build/html/index.html`. You will find a quick start on how to run the demos, technical notes and code documentations.

Note that we provide already computed filters in `.hlsi` and `.npz` formats accessible in `data/preprocess/filters`. To use them in your own applications you don't have to run anything! 

For example, filter values for solving the Poisson pressure (inverse Poisson) corresponding to 100 Jacobi iteration are found here `data/preprocess/filters/single_poisson_D3_INVERSE_STANDARD_dx_0.9_itr_100.hlsli`. 

You can also generate new filters with different paramteres than those available in the database. For details see the documentation. 

# Install & Run

First, install the the depedencies in `requirements.txt`:
```bash
pip install -r requirements.txt
```
Move to the demo folder  `src\demos`:
```bash
cd src\demos
```
Then, just run the provided demos:
```bash
python .\all_examples.py
```
You can select the demo you want to run in `src\demos\all_examples.py`. For more information, refer to the [Code Documentation](#documentation)


# Cite
BibTeX
<pre><code>
@inproceedings{10.1145/3528233.3530737,
author = {Rabbani, Amir Hossein and Guertin, Jean-Philippe and Rioux-Lavoie, Damien and Schoentgen, Arnaud and Tong, Kaitai and Sirois-Vigneux, Alexandre and Nowrouzezahrai, Derek},
title = {Compact Poisson Filters for Fast Fluid Simulation},
year = {2022},
isbn = {9781450393379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3528233.3530737},
doi = {10.1145/3528233.3530737},
abstract = {Poisson equations appear in many graphics settings including, but not limited to, physics-based fluid simulation. Numerical solvers for such problems strike context-specific memory, performance, stability and accuracy trade-offs. We propose a new Poisson filter-based solver that balances between the strengths of spectral and iterative methods. We derive universal Poisson kernels for forward and inverse Poisson problems, leveraging careful adaptive filter truncation to localize their extent, all while maintaining stability and accuracy. Iterative composition of our compact filters improves solver iteration time by orders-of-magnitude compared to optimized linear methods. While motivated by spectral formulations, we overcome important limitations of spectral methods while retaining many of their desirable properties. We focus on the application of our method to high-performance and high-fidelity fluid simulation, but we also demonstrate its broader applicability. We release our source code at https://github.com/Ubisoft-LaForge/CompactPoissonFilters .},
booktitle = {ACM SIGGRAPH 2022 Conference Proceedings},
articleno = {35},
numpages = {9},
keywords = {iterative methods, reduced modeling},
location = {Vancouver, BC, Canada},
series = {SIGGRAPH '22}
}
</code></pre>
# License
CC 4.0 : see details in Licence file


(c) 2023 All right reserved Ubisoft
