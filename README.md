# OCIMP

### Introduction
This codebase consists of implementations of algorithms &amp; simulation scripts for Online Contextual Influence Maximization Problem (OCIMP). We have support for both contextual and non-contextual simulations, as well as the variants of edge-level feedback (OCIMP), node-level feedback (OCIMP_Node) and active learning setting (OCIMP_Active).

You are free to use these codes for any purpose. The raw results and plots provided in Misc folder are from our work and are only for inspection. If you intend to use these implementations or refer to COIN, COIN+, COIN-HD or PureExploitation algorithms, please cite our paper:

> A. O. Saritac, A. Karakurt and C. Tekin, ["Online contextual influence maximization in social networks"](http://ieeexplore.ieee.org/document/7852372/), in Proc. 54th Allerton Conference, September 2016, Monticello, Illinois.

Detailed descriptions and related references of the algorithms are provided in this paper. For any further questions, report problems or suggest improvements feel free to contact me. Third-party contributions are more than welcome.

### Installation &amp; Usage
This is a Python 3 codebase and only depends on NumPy in run-time and Cython for linking C++ implementations to Python. We use [TIM+](https://github.com/nd7141/TIM) as the offline influence maximization solver. A refactored version of this library is hosted at TIM+ folder. We require the user to cythonize (link C++ source to Python) and compile the C++ source of TIM+ according to their own architecture. To do so,

    python ./TIM+/Tim/setup.py build_ext --inplace
    python ./TIM+/UnderTim/setup.py build_ext --inplace

Then, replace the pytim.\* and undertim.\* files in each of the algorithm's folders. These are the compilations based on the author's computer. Notice that, the graph files and these C++ shared object files are copied in each of the folder independently. This prevents the racing conditions and makes it possible to run multiple algorithms simultaneously without any segmentation fault.

Each algorithm has a class implementation that implements a base class *IM_Base\*.py*, which handles the low-level tasks about the simulations. There are currently 2 context space implementations, IM_Base.py corresponds to the pyramidal surfaces and IM_Base2.py to the sigmoid-based surfaces. The results of these settings are stored in the folders of te same name under Misc/\*_results folders. To use one, just make sure that the algorithm extends the correct base class. To run an algorithm, you can simply use \*_script.py. The algorithm parameters are explicitly declared in these scripts and can easily be changed.

