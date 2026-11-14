#!/usr/bin/env python3
"""
================================================================================
A φ-Based Wall Function and Spectral Scaling for Turbulent Channel Flow:
Exploiting the Quadratic Relation ⟨φ²⟩ = φ₀² + ⟨φ'²⟩

Author: [Mahmoud F. Abdel-Sattar ]
Affiliation: [Department of Astronomy and MeteorologyFaculty of Science, Al-Azhar University, Cairo, Egypt]
Date: 2026
Reference: Moser, Kim & Mansour (1999), Phys. Fluids 11, 943–945.
================================================================================

This code reproduces the results presented in the paper:
1. A blended φ₀ wall function for mean velocity (viscous sublayer corrected).
2. A φ-variance wall function using PCHIP and the quadratic relation.
3. A spectral scaling E_φ(k) = E_u(k) / (1 + U^{+2}).

It compares the blended φ₀ wall function with Reichardt's classical law.
The code is self-contained and can be run in Google Colab or Jupyter.

Requirements:
    numpy, scipy, matplotlib

Usage:
    Run the script. If running in Colab, you will be prompted to upload
    the six DNS files (y_dns.dat, u_dns.dat, u2_dns.dat, v2_dns.dat,
    w2_dns.dat, uv_dns.dat). Otherwise, place them in a folder named
    'dns_data' in the current working directory.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os, json, warnings
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. PUBLICATION-QUALITY PLOT SETTINGS
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "text.usetex": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
})

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
def load_dns_from_upload():
    """
    Load DNS data either by uploading files (Colab) or from a local
    'dns_data' directory.

    Returns a dictionary with keys: y, u, u2, v2, w2, uv.
    The data are expected in wall units, with y in outer units (y/δ).
    """
    try:
        from google.colab import files
        print("Please upload the six DNS files.")
        uploaded = files.upload()
        os.makedirs("dns_data", exist_ok=True)
        for fname, content in uploaded.items():
            with open(os.path.join("dns_data", fname), "wb") as f:
                f.write(content)
    except ImportError:
        # Not in Colab; proceed with local directory
        pass

    base = ["y_dns", "u_dns", "u2_dns", "v2_dns", "w2_dns", "uv_dns"]
    suffixes = [".dat", ".dat.txt", ".txt"]
    files_found = {}
    for bn in base:
        for s in suffixes:
            p = os.path.join("dns_data", bn + s)
            if os.path.exists(p):
                files_found[bn] = p
                break
    if len(files_found) != 6:
        raise FileNotFoundError(
            "DNS files missing. Please ensure they are in 'dns_data/' or upload them."
        )
    data = {}
    for bn in base:
        key = bn.replace("_dns", "")
        data[key] = np.loadtxt(files_found[bn]).flatten()
    return data

# ==============================================================================
# 3. CLASSICAL WALL FUNCTION (REICHARDT)
# ==============================================================================
def reichardt_law(y_plus, kappa=0.41, B=5.5):
    """
    Reichardt's law of the wall (explicit).
    Valid over the entire wall-normal direction.
    """
    return (1.0 / kappa) * np.log(1.0 + kappa * y_plus) + \
           7.8 * (1.0 - np.exp(-y_plus / 11.0) - (y_plus / 11.0) * np.exp(-y_plus / 3.0))

# ==============================================================================
# 4. BLENDED φ₀ WALL FUNCTION FOR MEAN VELOCITY
# ==============================================================================
def blended_phi0_wall_function(y_plus, phi0_DNS, kappa=0.41, B=5.2, U_ref=1.0):
    """
    Compute the blended φ₀ wall function:
        φ₀(y⁺) = (1 - w) φ₀_visc + w φ₀_log,
    where
        φ₀_visc = arcsinh(y⁺ / U_ref),
        φ₀_log  = arcsinh( U_log / U_ref ),  U_log = (1/κ) ln(y⁺) + B,
        w = tanh( (y⁺ / A)² ).

    The blending parameter A is calibrated against DNS.

    Returns
    -------
    U_WF : ndarray
        Predicted mean velocity U⁺.
    A_opt : float
        Optimal blending parameter.
    """
    yp = np.maximum(y_plus, 1e-10)
    U_log = (1.0 / kappa) * np.log(yp) + B
    phi0_log = np.arcsinh(U_log / U_ref)
    phi0_visc = np.arcsinh(yp / U_ref)

    def error(A):
        w = np.tanh((yp / A) ** 2)
        phi0_blend = (1 - w) * phi0_visc + w * phi0_log
        return np.sum((phi0_blend - phi0_DNS) ** 2)

    res = minimize_scalar(error, bounds=(1.0, 50.0), method="bounded")
    A_opt = res.x

    w = np.tanh((yp / A_opt) ** 2)
    phi0_blend = (1 - w) * phi0_visc + w * phi0_log
    U_WF = U_ref * np.sinh(phi0_blend)
    return U_WF, A_opt

# ==============================================================================
# 5. φ-VARIANCE WALL FUNCTION
# ==============================================================================
def phi_variance_wall_function(y_plus, U_plus, u2_plus, U_ref=1.0):
    """
    Compute ⟨φ'²⟩ using the quadratic relation:
        ⟨φ'²⟩ = ⟨u'²⟩ / (U_ref² + U⁺²),
    where ⟨u'²⟩ is reconstructed via PCHIP from DNS anchors in the near-wall
    region (y⁺ ≤ 30).

    Parameters
    ----------
    y_plus : ndarray
        Wall-normal coordinate in wall units.
    U_plus : ndarray
        Mean velocity U⁺.
    u2_plus : ndarray
        DNS ⟨u'²⟩.
    U_ref : float, optional
        Reference velocity (default 1.0).

    Returns
    -------
    phi_sq : ndarray
        Wall function for ⟨φ'²⟩ (NaN for y⁺ > 30).
    """
    anchors_y = [0.0, 1.0, 5.0, 12.0, 30.0]
    anchors_u2 = [
        0.0,
        np.interp(1.0, y_plus, u2_plus),
        np.interp(5.0, y_plus, u2_plus),
        np.interp(12.0, y_plus, u2_plus),
        np.interp(30.0, y_plus, u2_plus),
    ]
    pchip = PchipInterpolator(anchors_y, anchors_u2)
    u2_model = pchip(np.clip(y_plus, 0, 30))
    u2_model[y_plus > 30] = np.nan
    u2_model[0] = 0.0
    phi_sq = u2_model / (U_ref ** 2 + U_plus ** 2)
    return phi_sq

# ==============================================================================
# 6. SPECTRAL SCALING
# ==============================================================================
def spectral_scaling(k, y_plus_pos, y_plus, U_plus, u2_plus, U_ref=1.0, kappa=0.41):
    """
    Compute the von Kármán model spectrum E_u(k) and the scaled E_φ(k):
        E_φ(k) = E_u(k) / (1 + U⁺²).

    Parameters
    ----------
    k : ndarray
        Wavenumber array.
    y_plus_pos : float
        y⁺ position at which to compute the spectrum.
    y_plus : ndarray
        DNS y⁺ array (for interpolation).
    U_plus : ndarray
        DNS U⁺ array.
    u2_plus : ndarray
        DNS ⟨u'²⟩ array.
    U_ref : float, optional
        Reference velocity (default 1.0).
    kappa : float, optional
        von Kármán constant (default 0.41).

    Returns
    -------
    E_u : ndarray
        Velocity spectrum.
    E_phi : ndarray
        φ-spectrum.
    """
    U_star = np.interp(y_plus_pos, y_plus, U_plus)
    u2_star = np.interp(y_plus_pos, y_plus, u2_plus)
    L = max(kappa * y_plus_pos, 1.0)  # mixing length proxy
    E_u = (2.0 / 3.0) * u2_star * L / (1.0 + (k * L) ** 2) ** (5.0 / 6.0)
    E_phi = E_u / (U_ref ** 2 + U_star ** 2)
    return E_u, E_phi

# ==============================================================================
# 7. MAIN ROUTINE
# ==============================================================================
def main():
    print("=" * 80)
    print(" φ-Based Wall Function and Spectral Scaling")
    print("=" * 80)

    Re_tau = 395
    U_ref = 1.0
    kappa = 0.41
    B = 5.2

    # Load DNS
    dns = load_dns_from_upload()
    y_plus = dns["y"] * Re_tau
    U_plus = dns["u"]
    u2_plus = dns["u2"]

    # DNS reference quantities
    phi0_DNS = np.arcsinh(U_plus / U_ref)
    phi_sq_DNS = u2_plus / (U_ref ** 2 + U_plus ** 2)

    # --------------------------------------------------------------------------
    # 7.1 Mean velocity wall function
    # --------------------------------------------------------------------------
    U_WF, A_opt = blended_phi0_wall_function(y_plus, phi0_DNS, kappa, B, U_ref)
    U_reichardt = reichardt_law(y_plus)

    R2_phi0 = 1 - np.sum((U_WF - U_plus) ** 2) / np.sum((U_plus - np.mean(U_plus)) ** 2)
    R2_reichardt = 1 - np.sum((U_reichardt - U_plus) ** 2) / np.sum((U_plus - np.mean(U_plus)) ** 2)

    print(f"\n[1] Mean velocity wall functions")
    print(f"    Blended φ₀: R² = {R2_phi0:.4f} (A = {A_opt:.3f})")
    print(f"    Reichardt:  R² = {R2_reichardt:.4f}")
    print(f"    Note: blended φ₀ is NOT superior to Reichardt (ΔR² = {R2_phi0 - R2_reichardt:.4f}).")

    # Plot mean velocity
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(y_plus, U_plus, "k-", lw=2.5, label="DNS")
    ax.semilogx(y_plus, U_WF, "r--", lw=2, label=f"Blended φ₀ (R²={R2_phi0:.3f})")
    ax.semilogx(y_plus, U_reichardt, "b:", lw=2, label=f"Reichardt (R²={R2_reichardt:.3f})")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_title("Mean velocity wall function comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("wall_functions.pdf", bbox_inches="tight")
    plt.savefig("wall_functions.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("    -> wall_functions.pdf")

    # --------------------------------------------------------------------------
    # 7.2 φ-variance wall function
    # --------------------------------------------------------------------------
    phi_sq_WF = phi_variance_wall_function(y_plus, U_plus, u2_plus, U_ref)
    mask_near = y_plus < 30
    R2_phi_sq = 1 - np.sum((phi_sq_WF[mask_near] - phi_sq_DNS[mask_near]) ** 2) / \
                    np.sum((phi_sq_DNS[mask_near] - np.mean(phi_sq_DNS[mask_near])) ** 2)
    print(f"\n[2] φ-variance wall function")
    print(f"    R² (y⁺ < 30) = {R2_phi_sq:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(y_plus[mask_near], phi_sq_DNS[mask_near], "k-", lw=2.5, label="DNS")
    ax.loglog(y_plus[mask_near], phi_sq_WF[mask_near], "r--", lw=2,
              label=rf"PCHIP model, R²={R2_phi_sq:.3f}")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$\langle\phi'^2\rangle$")
    ax.set_title(r"$\langle\phi'^2\rangle$ wall function (y⁺ < 30)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("phi_variance_wall_function.pdf", bbox_inches="tight")
    plt.savefig("phi_variance_wall_function.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("    -> phi_variance_wall_function.pdf")

    # --------------------------------------------------------------------------
    # 7.3 Spectral scaling
    # --------------------------------------------------------------------------
    k = np.logspace(-3, 2, 300)
    yp_positions = [5.0, 30.0, 100.0]
    labels = ["y⁺ = 5", "y⁺ = 30", "y⁺ = 100"]
    colors = ["blue", "red", "green"]

    fig_spec, axes_spec = plt.subplots(1, 2, figsize=(14, 5))
    for yp_pos, lab, col in zip(yp_positions, labels, colors):
        E_u, E_phi = spectral_scaling(k, yp_pos, y_plus, U_plus, u2_plus, U_ref, kappa)
        U_val = np.interp(yp_pos, y_plus, U_plus)
        axes_spec[0].loglog(k, E_u, lw=2, ls="-", color=col,
                            label=f"{lab} (U⁺={U_val:.1f})")
        axes_spec[1].loglog(k, E_phi, lw=2, ls="--", color=col,
                            label=f"{lab} (U⁺={U_val:.1f})")

    axes_spec[0].set_xlabel(r"$k$")
    axes_spec[0].set_ylabel(r"$E_u(k)$")
    axes_spec[0].set_title("Velocity spectrum (von Kármán model)")
    axes_spec[0].legend(fontsize=9)
    axes_spec[0].grid(alpha=0.3, which="both")

    axes_spec[1].set_xlabel(r"$k$")
    axes_spec[1].set_ylabel(r"$E_\phi(k)$")
    axes_spec[1].set_title(r"$\phi$-spectrum: $E_\phi = E_u/(1+U^{+2})$")
    axes_spec[1].legend(fontsize=9)
    axes_spec[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("spectral_scaling.pdf", bbox_inches="tight")
    plt.savefig("spectral_scaling.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("    -> spectral_scaling.pdf")

    # --------------------------------------------------------------------------
    # 7.4 Save results
    # --------------------------------------------------------------------------
    results = {
        "Re_tau": Re_tau,
        "A_opt": float(A_opt),
        "R2_blended_phi0": float(R2_phi0),
        "R2_reichardt": float(R2_reichardt),
        "R2_phi_sq_near_wall": float(R2_phi_sq),
        "note": "Blended φ₀ is not superior to Reichardt; the main contribution is the φ-variance wall function and spectral scaling.",
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[3] Results saved to results.json")
    print("=" * 80)

if __name__ == "__main__":
    main()