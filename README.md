```markdown
# A φ-Variance Wall Function and Spectral Scaling for Turbulent Channel Flow

This repository contains the code, data instructions, and reproducibility materials for the paper:

> **A φ-Variance Wall Function and Spectral Scaling for Turbulent Channel Flow**  
> Mahmoud F. Abdel-Sattar  
> *Department of Astronomy and Meteorology, Faculty of Science, Al-Azhar University, Cairo, Egypt*  
> Submitted to *Journal of Turbulence* (2026).

---

## Overview

The paper introduces a φ-based framework for near-wall modelling of turbulent channel flow. The key ingredients are:

- The inverse hyperbolic sine transformation  
  \[
  \phi = \operatorname{arcsinh}\left(\frac{U^+}{U_{\mathrm{ref}}^+}\right),
  \]
  with \(U_{\mathrm{ref}}^+ = 1\) (i.e., \(U_{\mathrm{ref}} = u_\tau\)).
- The exact quadratic relation  
  \[
  \langle\phi^2\rangle = \phi_0^2 + \langle\phi'^2\rangle,
  \]
  which follows directly from the Reynolds decomposition.
- A wall function for the φ-variance:
  \[
  \langle\phi'^2\rangle_{\mathrm{WF}}(y^+) = 
  \frac{\langle u'^2\rangle_{\mathrm{PCHIP}}(y^+)}{1 + U^{+2}(y^+)},
  \qquad 0 \le y^+ \le 30.
  \]
- A spectral scaling law:
  \[
  E_\phi(k) = \frac{E_u(k)}{1 + U^{+2}}.
  \]

The main results are:

- \(\phi\)-variance wall function: \(R^2 = 0.9792\) against DNS at \(Re_\tau = 395\) in the near-wall region (\(y^+ \le 30\)).
- Blended \(\phi_0\) mean-velocity wall function: \(R^2 = 0.9904\), competitive with but not superior to classical laws.

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── data/
│   └── README_data.md          # Instructions for obtaining DNS data
├── code/
│   ├── phi_wall_function.py    # φ-variance and mean-velocity wall functions
│   ├── spectral_scaling.py     # Spectral scaling and von Kármán model
│   └── validation_dns.py       # Reproduces all figures and metrics
├── figures/
│   ├── phi_variance_wall_function.pdf
│   ├── spectral_scaling.pdf
│   └── wall_functions.pdf
└── results/
    └── results.json            # Numerical outputs (R², parameters)
```

---

Requirements

The code is written in Python 3.9+ and requires the following packages:

```text
numpy>=1.21
scipy>=1.7
matplotlib>=3.4
```

Install them with:

```bash
pip install -r requirements.txt
```

---

Data

The validation uses the publicly available DNS data of Moser, Kim & Mansour (1999) for turbulent channel flow at Re_\tau = 395:

R. D. Moser, J. Kim, and N. N. Mansour,
“Direct numerical simulation of turbulent channel flow up to Re_\tau = 590,”
Phys. Fluids, vol. 11, no. 4, pp. 943–945, 1999.

The six DNS files (y_dns.dat, u_dns.dat, u2_dns.dat, v2_dns.dat, w2_dns.dat, uv_dns.dat) are not redistributed here. Please obtain them from the original source and place them in the data/ directory.

Instructions for obtaining the data are provided in data/README_data.md.

---

Usage

1. Reproduce all results and figures

```bash
python code/validation_dns.py
```

This script:

· Loads the DNS data.
· Constructs the \phi-variance wall function using PCHIP.
· Computes the blended \phi_0 wall function for the mean velocity.
· Generates the spectral scaling and the von Kármán model spectra.
· Saves the figures to figures/ and the numerical results to results/results.json.

2. Use the wall functions interactively

```python
from code.phi_wall_function import phi_variance_wall_function, blended_phi0_wall_function

# Example: compute φ-variance wall function
import numpy as np
y_plus = np.linspace(0, 30, 100)
u2_plus = np.loadtxt('data/u2_dns.dat')  # after placing the file
U_plus = np.loadtxt('data/u_dns.dat')
phi_sq = phi_variance_wall_function(y_plus, U_plus, u2_plus)
```

---

Reproducibility

The key numerical results reported in the paper are:

Quantity Value
\phi-variance wall function R^2 ( y^+ \le 30 ) 0.9792
Blended \phi_0 wall function R^2 (all y^+) 0.9904
Calibrated blending parameter A 14.61

To verify these values, run:

```bash
python code/validation_dns.py
```

The output will be printed to the console and saved in results/results.json.

---

Citation

If you use this code or the results of this paper, please cite:

```bibtex
@article{AbdelSattar2026phi,
  title   = {A \(\phi\)-Variance Wall Function and Spectral Scaling for Turbulent Channel Flow},
  author  = {Abdel-Sattar, Mahmoud F.},
  journal = {Journal of Turbulence},
  year    = {2026},
  note    = {Submitted}
}
```

A permanent DOI will be assigned upon publication.

---

License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Contact

Mahmoud F. Abdel-Sattar
Department of Astronomy and Meteorology
Faculty of Science, Al-Azhar University
Cairo, Egypt
Email: m.f.abdel-sattar@azhar.edu.eg
ORCID: 0009-0003-0224-7007

---

Acknowledgements

The author acknowledges the developers of the DNS database of Moser, Kim & Mansour (1999), which was used for validation in this study.

```
