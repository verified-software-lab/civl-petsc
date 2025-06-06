# CIVL-PETSc Verification Project

This repository is dedicated to verifying PETSc’s vector module using the CIVL model checker. PETSc (Portable, Extensible Toolkit for Scientific Computation) is a widely used numerical library for scientific computing, providing parallel solvers and mathematical abstractions. Given its extensive use in high-performance computing applications, ensuring its correctness is critical.

This project applies **symbolic execution and model checking** to verify PETSc’s vector functions, leveraging CIVL’s capabilities to detect subtle defects that conventional testing might miss. The verification approach includes:
- **Abstract vector representation** to facilitate symbolic reasoning.
- **Specification stubs** that define expected function behavior.
- **Modular verification** to analyze functions independently.

**Before running this project, you must install CIVL.** Please follow the [CIVL installation instructions](https://vsl.cis.udel.edu/trac/civl/wiki/Introduction) before proceeding.

## Current Structure
```
.
├── build_functions.sh
├── common.mk
├── include
│   ├── civlcomplex.cvh
│   ├── civlvec.cvh
│   ├── petscvec.h
│   └── scalars.cvh
├── Makefile
├── README
├── Reports-Paper
│   ├── Function_Stat’s_05-23-2025_18:36:29_big.csv
│   ├── Function_Stat’s_05-23-2025_23:18:37_small.csv
│   ├── Summary_05-23-2025_18:36:29_big_63.log
│   └── Summary_05-23-2025_23:18:37_small_63.log
├── src
│   ├── civlcomplex.cvl
│   ├── civlvec.cvl
│   └── petscvec.cvl
├── targets
│   ├── VecAXPBY
│   │   ├── Makefile
│   │   ├── VecAXPBY.c
│   │   ├── VecAXPBY_driver.cvl
│   │   └── VecAXPBY_test.cvl
│   ├── VecAXPBYPCZ
│   │   ├── Makefile
│   │   ├── VecAXPBYPCZ.c
│   │   ├── VecAXPBYPCZ_driver.cvl
│   │   └── VecAXPBYPCZ_test.cvl
.   .   .
.   .   .
.   .   .
│   ├── VecMaxPointwiseDivide
│   │   ├── Makefile
│   │   ├── VecMaxPointwiseDivide.c
│   │   ├── VecMaxPointwiseDivide_driver.cvl
│   │   └── VecMaxPointwiseDivide_test.cvl
│   ├── VecMaxPointwiseDivide_MPI
│   │   ├── Makefile
│   │   ├── VecMaxPointwiseDivide_MPI.c
│   │   └── VecMaxPointwiseDivide_MPI_driver.cvl
│   ├── VecMaxPointwiseDivide_orig
│   │   ├── Makefile
│   │   ├── VecMaxPointwiseDivide.c
│   │   └── VecMaxPointwiseDivide_driver.cvl
│   ├── VecNorm
│   │   ├── Makefile
│   │   ├── VecNorm.c
│   │   └── VecNorm_driver.cvl
│   ├── VecNorm_MPI
│   │   ├── Makefile
│   │   ├── VecNorm_MPI.c
│   │   └── VecNorm_MPI_driver.cvl
│   ├── VecNorm_MPI_orig
│   │   ├── Makefile
│   │   ├── VecNorm_MPI.c
│   │   └── VecNorm_MPI_driver.cvl
│   ├── VecNorm_Seq
│   │   ├── Makefile
│   │   ├── VecNorm_Seq.c
│   │   ├── VecNorm_Seq_driver.cvl
│   │   └── VecNorm_Seq_test.c
.   .   .
.   .   .
.   .   .
│   └── VecWAXPY_Seq
│       ├── Makefile
│       ├── VecWAXPY_Seq.c
│       ├── VecWAXPY_Seq_driver.cvl
│       └── VecWAXPY_Seq_test.cvl
└── test
    ├── civlcomplex_test.cvl
    ├── civlvec_test.cvl
    ├── equals.c
    ├── Makefile
    ├── petscToCivl.cvl
    └── vectorOwnershipTest.cvl

```

## CIVL-PETSc Verification Project Directory Overview

Below is a breakdown of the key directories and their contents:

- **`include/`** – This folder holds essential header files that define core components needed for verification.
  - `civlcomplex.cvh` – Defines complex number types.
  - `civlvec.cvh` – Specifies vector types.
  - `petscvec.h` – Declares PETSc’s vector interface.
  - `scalars.cvh` – Defines scalar types used in computations.

- **`src/`** – Contains the actual implementation of various components.
  - `civlcomplex.cvl` – Implements CIVL complex number operations.
  - `civlvec.cvl` – Implements CIVL vector operations.
  - `petscvec.cvl` – Implements PETSc’s vector functions.

- **`targets/`** – This directory contains the PETSc functions that are being verified, with each function in its own subdirectory.
  - Each function subdirectory (e.g., `VecAXPBY/`) includes:
    - The original PETSc function (`VecAXPBY.c`).
    - A CIVL verification driver (`VecAXPBY_driver.cvl`).
    - A test implementation (`VecAXPBY_test.cvl`).
    - Each function directory contains a Makefile with two main verification targets:
      - `make small`: Tests vectors of size 1-3 using 1 or 2 processors
      - `make big`: Tests vectors of size 1-5 using 1-5 processors

- **`test/`** – Contains test files to validate core functionality.
  - `civlcomplex_test.cvl` – Tests complex number operations.
  - `civlvec_test.cvl` – Tests vector operations.
  - `petscToCivl.cvl` – Tests conversion between PETSc and CIVL vector types.
  - `vectorOwnershipTest.cvl` – Ensures proper handling of vector ownership.

- **`Reports-Paper/`** – Contains test execution logs and verification statistics in CSV format. These records document verification times, detected issues, and validation results, providing a structured view of the testing process.

## Verification Process

The project provides two methods for running verifications:

### 1. Using the Super Makefile

The root `Makefile` provides basic verification across all functions:

```bash
make small  # Quick verification with vector sizes 1-3 and 1-2 processors
make big    # Comprehensive verification with vector sizes 1-5 and 1-5 processors
make clean  # Cleanup build artifacts
```

While simpler to use, this method doesn’t store execution results or statistics.

### 2. Using the build_functions.sh Script (Recommended)

The build_functions.sh script offers enhanced verification with logging and statistics:

```bash
./build_functions.sh small  # Run quick verification
./build_functions.sh big    # Run comprehensive verification
```

Key features:
- Stores detailed logs in `Reports/Summary_<timestamp>_<mode>_<count>.log`
- Generates CSV statistics in `Reports/Function_Stat’s_<timestamp>_<mode>.csv`
- Provides real-time progress updates with color-coded results
- Shows execution time for each function
- Generates a final summary with:
  - Total verification time
  - Number of passed/failed targets
  - Individual function statistics

The script automatically creates the `Reports` directory if it doesn’t exist and includes timestamps in filenames for tracking multiple verification runs.

**Note:** Use `.build_functions.sh -h` for usage information.

## Note

This structure is subject to change as the project develops. Please refer to this README for the most up-to-date information on the repository structure and verification process.