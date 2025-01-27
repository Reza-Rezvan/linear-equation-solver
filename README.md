# Linear Equation Solver: Gauss-Jordan and LU Decomposition

This repository contains Python implementations for solving systems of linear equations using two numerical methods:

1. **Gauss-Jordan Elimination**
2. **LU Decomposition**

These methods can be applied to solve any square system of linear equations. The code is written in a modular way for clarity and easy integration.

## Features
- Solve a system of linear equations using the **Gauss-Jordan Elimination** method.
- Solve the same system using **LU Decomposition** (with forward and backward substitution).
- Handles errors for systems with no unique solutions.

## Requirements
The following Python libraries are required:
- `numpy`

Install the dependencies using pip:
```bash
pip install numpy
```

## Usage
Clone the repository and run the `solver.py` script:

```bash
git clone <repository-url>
cd <repository-directory>
python solver.py
```

### Example Input
The script solves a system of equations, such as:

```
2x + y - z = 8
-3x - y + 2z = -11
-2x + y + 2z = -3
```

Represented in matrix form:

```
A = [[2,  1, -1],
     [-3, -1,  2],
     [-2,  1,  2]]

b = [8, -11, -3]
```

### Output
The script provides solutions using both methods:

```
Solution using Gauss-Jordan:
[ 2.  3. -1.]

Solution using LU Decomposition:
[ 2.  3. -1.]
```

## Functions
### Gauss-Jordan Method
- **Function**: `gauss_jordan(a, b)`
- **Description**: Solves a system of equations by transforming the coefficient matrix into reduced row echelon form.

### LU Decomposition Method
- **Functions**:
  - `lu_decomposition(a)`
  - `forward_substitution(L, b)`
  - `backward_substitution(U, y)`
  - `solve_lu(a, b)`
- **Description**: Decomposes the matrix into lower (L) and upper (U) triangular matrices, solves using substitution.

## Error Handling
- Raises errors for singular matrices or systems with no unique solutions.
  
---
Feel free to modify and expand the code for your specific use cases!
