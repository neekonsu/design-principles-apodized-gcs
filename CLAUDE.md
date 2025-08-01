# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements the apodized grating coupler design principles from "Design Principles of Apodized Grating Couplers" by Zhao and Fan (2020). It provides a comprehensive Python implementation of optimization algorithms for maximizing coupling efficiency in photonic grating couplers while respecting fabrication constraints.

## Repository Structure

- `theory.md`: Complete theoretical documentation and Python implementation
  - Core optimization algorithm (Algorithm 1 from the paper)
  - Physical structure mapping between scattering strength and etch lengths
  - 2D extensions for complex beam profiles (vortex beams, focusing gratings)
  - Visualization and analysis tools

## Key Components

### Core Algorithm Classes

1. **ApodizedGratingDesigner**: Main optimizer that solves for optimal scattering strength distribution
   - Implements Algorithm 1 from the paper (backward recursion with transcendental equation solving)
   - Handles fabrication constraints (αmin, αmax bounds)
   - Supports various target modes (Gaussian, custom)

2. **GratingStructureMapper**: Converts between scattering strengths and physical parameters
   - Maps scattering strength to etch lengths using lookup tables
   - Calculates grating pitch for phase matching
   - Handles emission phase corrections

3. **GratingCouplerDesign**: Complete design workflow
   - Combines optimization with physical mapping
   - Generates fabrication-ready grating specifications
   - Calculates trench positions with phase corrections

### Extensions

- **GratingCoupler2D**: For complex 2D beam profiles (e.g., vortex beams)
- **FocusingGratingCoupler**: Fan-shaped focusing gratings in polar coordinates

## Development Environment

This is a Python-based scientific computing project. Required dependencies:
- `numpy`: Numerical computations
- `matplotlib`: Visualization
- `scipy`: Optimization and interpolation

## Common Development Tasks

Since this is primarily a research/theory implementation repository:

1. **Run the main example**: Execute the complete design flow demonstration
   ```python
   python -c "from theory import main; main()"
   ```

2. **Test specific algorithms**: The code is structured as a module with well-defined classes
   ```python
   from theory import ApodizedGratingDesigner
   designer = ApodizedGratingDesigner(alpha_min=0.02, alpha_max=0.09, grating_length=17.0)
   alpha_optimal = designer.solve_optimal_alpha()
   ```

3. **Visualize results**: Use built-in visualization functions
   ```python
   from theory import visualize_design_results
   visualize_design_results(results, "Custom Design")
   ```

## Architecture Notes

### Algorithm Implementation
- **Backward Recursion**: The core algorithm works backwards from grating end to beginning because optimal scattering at each position depends on future positions
- **Transcendental Equation Solving**: Uses iterative methods to solve Equation 29 from the paper
- **Constraint Handling**: Implements Equation 28 decision logic for fabrication limits

### Physical Mapping
- **Lookup Tables**: Scattering strength vs etch length mapping typically comes from FDTD simulations
- **Phase Matching**: Grating pitch calculated from Equation 31 ensuring constructive interference
- **Fabrication Constraints**: Respects minimum feature sizes and maximum etch depths

### Extensions Strategy
- **2D Designs**: Each x-position optimized independently (no transverse power flow assumption)
- **Focusing Gratings**: Modified target amplitude includes √r factor for radial coordinate systems
- **Complex Modes**: Supports arbitrary target beam profiles (vortex beams, custom shapes)

## Key Equations Implemented

- **Optimal Scattering (Eq. 27)**: `αz = At²(z) / (2 * [coupling efficiency integral]²)`
- **Phase Matching (Eq. 31)**: `Λ = (λ + le(nwg - ne)) / (nwg - nc sin θ)`
- **Constraint Logic (Eq. 28)**: Decision tree for αmin/αmax bounds and fabrication feasibility

## Design Workflow

1. Define target mode and fabrication constraints
2. Run optimization algorithm to find scattering strength distribution
3. Map scattering strengths to physical etch lengths
4. Calculate grating pitch and trench positions
5. Apply phase corrections between adjacent trenches
6. Generate fabrication-ready specifications