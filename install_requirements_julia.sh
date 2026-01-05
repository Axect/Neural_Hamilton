#!/bin/bash
# Install Julia dependencies for scripts/true_trajectories.jl

set -e

echo "Installing Julia dependencies..."

julia -e '
using Pkg

packages = [
    "DifferentialEquations",  # ODE solvers including KahanLi8 symplectic integrator
    "DataFrames",             # Data manipulation
    "Parquet2",               # Parquet file I/O
    "Parameters",             # @with_kw macro for struct parameters
    "PCHIPInterpolation",     # PCHIP interpolation for potential splines
    "ForwardDiff",            # Automatic differentiation for dV/dq
    "ProgressMeter"           # Progress bar for simulations
]

for pkg in packages
    println("Installing $pkg...")
    Pkg.add(pkg)
end

println("Precompiling packages...")
Pkg.precompile()

println("Julia dependencies installed successfully!")
'
