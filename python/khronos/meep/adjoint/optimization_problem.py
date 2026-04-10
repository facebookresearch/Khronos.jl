# Copyright (c) Meta Platforms, Inc. and affiliates.
"""OptimizationProblem: orchestrates forward/adjoint runs and gradient computation."""

import warnings

from .._units import meep_to_khronos_freq, meep_to_khronos_time
from ..geom import Vector3


class OptimizationProblem:
    """Adjoint-based topology optimization problem.

    Orchestrates forward simulation, adjoint simulation, and gradient
    computation through the Khronos Julia backend.

    Parameters
    ----------
    simulation : Simulation
        The meep-compatible Simulation object.
    objective_functions : list of callable
        Differentiable functions of the objective arguments.
        Each function takes the objective argument values and returns a scalar.
    objective_arguments : list of ObjectiveQuantity
        EigenmodeCoefficient, FourierFields, etc.
    design_regions : list of DesignRegion
        Parameterized design regions.
    frequencies : array-like, optional
        Frequencies at which to evaluate objectives.
    fcen, df, nf : float, float, int
        Alternative frequency specification.
    decay_by : float
        DFT convergence threshold.
    minimum_run_time : float
        Minimum simulation time in meep units.
    maximum_run_time : float, optional
        Maximum simulation time in meep units.
    """

    def __init__(self, simulation, objective_functions, objective_arguments,
                 design_regions, frequencies=None, fcen=None, df=None, nf=None,
                 decay_by=1e-11, decimation_factor=0,
                 minimum_run_time=0, maximum_run_time=None,
                 finite_difference_step=1e-3, step_funcs=None):
        self.simulation = simulation
        self.objective_functions = objective_functions
        self.objective_arguments = objective_arguments
        self.design_regions = design_regions
        self.decay_by = decay_by
        self.minimum_run_time = minimum_run_time
        self.maximum_run_time = maximum_run_time
        self.finite_difference_step = finite_difference_step

        # Resolve frequencies
        if frequencies is not None:
            self.frequencies = list(frequencies)
        elif fcen is not None and nf is not None:
            if nf == 1:
                self.frequencies = [fcen]
            else:
                df = df or 0
                self.frequencies = [
                    fcen - df / 2 + i * df / (nf - 1) for i in range(nf)
                ]
        else:
            self.frequencies = [0.15]  # default

        self.nf = len(self.frequencies)
        self._khronos_opt = None

    def __call__(self, rho_vector, need_value=True, need_gradient=True,
                 beta=None):
        """Evaluate objective function and/or gradient.

        Parameters
        ----------
        rho_vector : list of arrays
            Design variables, one array per design region.
        need_value : bool
            Whether to compute the objective value.
        need_gradient : bool
            Whether to compute the gradient.
        beta : float, optional
            Projection beta to apply.

        Returns
        -------
        (f0, gradient) : tuple
            f0 is array of objective values (one per objective function).
            gradient is list of arrays (one per design region per objective).
        """
        import numpy as np
        # Update design
        self.update_design(rho_vector, beta)

        if need_value and need_gradient:
            # Forward + adjoint
            f0 = self.forward_run()
            self.adjoint_run()
            gradient = self.calculate_gradient()
            return f0, gradient
        elif need_value:
            f0 = self.forward_run()
            return f0, [np.zeros(dr.num_design_params)
                        for dr in self.design_regions]
        elif need_gradient:
            self.forward_run()
            self.adjoint_run()
            gradient = self.calculate_gradient()
            return np.zeros(len(self.objective_functions)), gradient
        else:
            return np.zeros(len(self.objective_functions)), []

    def forward_run(self):
        """Run the forward simulation and evaluate objectives.

        Returns
        -------
        f0 : array of objective values.
        """
        import numpy as np
        from ..._bridge import get_khronos, get_jl
        K = get_khronos()
        jl = get_jl()

        if self._khronos_opt is None:
            self._build_khronos_opt()

        K.forward_run_b(self._khronos_opt)

        # Evaluate objective functions on the monitor results
        # For now, return placeholder
        return np.zeros(len(self.objective_functions))

    def adjoint_run(self):
        """Run the adjoint simulation."""
        from ..._bridge import get_khronos
        K = get_khronos()

        if self._khronos_opt is None:
            raise RuntimeError("Must call forward_run() first")

        K.adjoint_run_b(self._khronos_opt)

    def calculate_gradient(self):
        """Compute the gradient from forward/adjoint field overlap.

        Returns
        -------
        list of gradient arrays, one per design region.
        """
        import numpy as np
        from ..._bridge import get_khronos
        K = get_khronos()

        if self._khronos_opt is None:
            raise RuntimeError("Must call forward_run() and adjoint_run() first")

        grad = np.array(K.calculate_gradient_b(self._khronos_opt))
        return [grad]

    def calculate_fd_gradient(self, num_gradients=1, db=1e-4,
                              design_variables_idx=0, filter=None):
        """Compute finite-difference gradient for validation.

        Parameters
        ----------
        num_gradients : int
            Number of random design variables to perturb.
        db : float
            Finite-difference step size.

        Returns
        -------
        (indices, fd_gradient) : tuple
        """
        import numpy as np
        from ..._bridge import get_khronos
        K = get_khronos()

        if self._khronos_opt is None:
            self._build_khronos_opt()

        result = K.calculate_fd_gradient(
            self._khronos_opt,
            num_gradients=num_gradients,
            db=db,
        )
        return np.array(result)

    def update_design(self, rho_vector, beta=None):
        """Update the design variables.

        Parameters
        ----------
        rho_vector : list of arrays
            One weight array per design region.
        beta : float, optional
            Projection beta to update.
        """
        import numpy as np
        for i, dr in enumerate(self.design_regions):
            if i < len(rho_vector):
                weights = np.asarray(rho_vector[i]).flatten()
                dr.update_design_parameters(weights)
                if beta is not None:
                    dr.update_beta(beta)

        # Invalidate cached optimization problem
        if self._khronos_opt is not None:
            from ..._bridge import get_khronos
            K = get_khronos()
            for i, dr in enumerate(self.design_regions):
                if i < len(rho_vector) and dr._khronos_dr is not None:
                    K.update_design_b(dr._khronos_dr,
                                     np.asarray(rho_vector[i]).flatten())

    def get_fdf_funcs(self):
        """Get standalone f and df functions for use with optimizers.

        Returns
        -------
        (f_func, df_func) : tuple of callables
        """
        def f(x):
            f0, _ = self([x], need_gradient=False)
            return f0

        def df(x, grad):
            _, g = self([x], need_value=False)
            if g:
                grad[:] = g[0]
            return 0.0

        return f, df

    def plot2D(self, init_opt=False, **kwargs):
        """Visualization (NOT SUPPORTED)."""
        raise NotImplementedError("plot2D is not supported by Khronos.")

    def _build_khronos_opt(self):
        """Build the Khronos OptimizationProblem Julia object."""
        from ..._bridge import get_khronos, get_gp, get_jl
        K = get_khronos()
        gp = get_gp()
        jl = get_jl()

        # Build the simulation first
        self.simulation.init_sim()

        # Build design regions
        k_design_regions = []
        for dr in self.design_regions:
            k_design_regions.append(dr._to_khronos(K))

        # Build objectives
        k_objectives = []
        for obj_arg in self.objective_arguments:
            k_objectives.append(obj_arg._to_khronos(K, self.frequencies))

        # Build optimization problem
        self._khronos_opt = K.OptimizationProblem(
            simulation=self.simulation._jl_sim,
            objectives=k_objectives,
            design_region=k_design_regions[0] if k_design_regions else None,
            frequencies=[meep_to_khronos_freq(f) for f in self.frequencies],
        )
