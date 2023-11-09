# Qiskit Examples for Ising Models

This repository is a collection of Jupyter notebooks and Python scripts designed to demonstrate the use of the [Qiskit](https://qiskit.org/) library for simulating Ising models, which are critical for understanding magnetic material properties and are broadly applicable in optimization problems, as well as implementing the Variational Quantum Eigensolver (VQE) and Quantum Approximate Optimization Algorithm (QAOA). The Variational Quantum Eigensolver (VQE) and Quantum Approximate Optimization Algorithm (QAOA) are quantum algorithms designed to find the ground state of quantum systems and solve combinatorial optimization problems, respectively.

## Overview

The code in this repository covers a range of examples from basic Ising model simulations to more complex quantum algorithms like VQE and QAOA applied to solve optimization problems using quantum computing.

## Contents

- `QAOA_short.ipynb`: A brief introduction to the QAOA.
- `QAOA_tutorial.ipynb`: Comprehensive tutorial on implementing QAOA with Qiskit.
- `QRTE_trotter.ipynb`: Implementation of the Quantum Real Time Evolution (QRTE) using Trotterization.
- `hydrogen_eigenvalues.ipynb`: Calculation of hydrogen molecule eigenvalues using quantum computing.
- `ising_eigenvalues.ipynb`: Ising model eigenvalues simulation.
- `ising_ibm.ipynb`: Running Ising model simulation on IBM quantum hardware.
- `ising_recycler.py`: Script to demonstrate recycling qubits for Ising model simulations.
- `ising_simulations.py`: Basic Ising model simulations using Qiskit.
- `ising_simulations_cluster.ipynb`: Ising model simulations on a cluster.
- `qtev.ipynb`: Quantum Time Evolution (QTEV) examples.
- `quantumsim.ipynb`: General quantum simulations using Qiskit.
- `test.ipynb`: Test notebook for various Qiskit functionalities.
- `vqe_example.ipynb`: Variational Quantum Eigensolver example implementation.
- `vqe_functions.py`: Supporting Python functions for the VQE implementation.

## Prerequisites

To run these notebooks, you need to have the following installed:
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Qiskit

You can install Qiskit using pip:

```bash
pip install qiskit
```

## Deprecation Warnings
Please note that as of Qiskit Terra version 0.24.0, certain classes like CircuitOp and CircuitStateFn used in these scripts are deprecated. They will be removed in a future release, and it is recommended to visit the provided migration link in the deprecation warning messages for guidelines on updating the code.

## Acknowledgments
The Qiskit community for providing an excellent platform for quantum computation
Researchers and developers who have contributed to the theory and practice behind VQE and QAOA.

## References
To understand the theoretical background and the algorithms' workings, please refer to the following resources:

- "Quantum Computation and Quantum Information" by Michael A. Nielsen and Isaac L. Chuang
- [Qiskit Documentation](https://qiskit.org/documentation/)



