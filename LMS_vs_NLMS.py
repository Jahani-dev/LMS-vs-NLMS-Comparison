#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 23:45:40 2025

@author: Sahar Jahani


LMS vs NLMS Adaptive Filter Demo

This script compares the performance of Least Mean Squares (LMS) 
and Normalized Least Mean Squares (NLMS) adaptive filters 
in denoising a signal corrupted with Gaussian noise.

Author: Sahar Jahani
"""

import numpy as np
import matplotlib.pyplot as plt

# Signal setup
n = 300
t = np.linspace(0, 4 * np.pi, n)
true_signal = 2 * np.sin(t)
noise = np.random.normal(0, 0.5, n)
noisy_signal = true_signal + noise

# LMS parameters
M = 10
p = np.mean(noisy_signal**2)
mu_lms = 1 / (10 * p)  # Step size # Recommended stability criterion
  
wlms = np.zeros(M) # LMS filter weights
ylms = np.zeros(n) # LMS output signal
elms = np.zeros(n) # LMS error signal

# --------------------------
# LMS Adaptive Filter
# --------------------------
for i in range(M, n):
    x_lms = noisy_signal[i-M:i][::-1]
    ylms[i] = wlms.T @ x_lms
    elms[i] = true_signal[i] - ylms[i]
    wlms = wlms + mu_lms * elms[i] * x_lms


# NLMS parameters
M = 10               # Filter order
mu_nlms = 0.8        # Step size
p = np.mean(noisy_signal**2)
mu_lms = 1 / (10 * p)  # Step size # Recommended stability criterion
epsilon = 1e-6        # Small constant to avoid division by zero
wnlms = np.zeros(M)   # Initial weights
enlms = np.zeros(n)   # Error
ynlms = np.zeros(n)   # Filter output

    
# --------------------------
# NLMS Adaptive Filter
# --------------------------

for i in range(M, n):
    x_nlms = noisy_signal[i-M:i][::-1]               # Input vector
    ynlms[i] = wnlms.T @ x_nlms                      # Filter output
    enlms[i] = true_signal[i] - ynlms[i]             # Error
    norm = epsilon + np.dot(x_nlms, x_nlms)           # Normalization term
    wnlms = wnlms + (mu_nlms / norm) * enlms[i] * x_nlms  # Update weights

# Plot the results
plt.figure(figsize=(12, 5))
plt.plot(t, true_signal, label='True Signal', linewidth=2)
plt.plot(t, noisy_signal, label='Noisy Signal', color='gray')
plt.plot(t, ylms, label='LMS Output', linestyle='--')
plt.plot(t, ynlms, label='NLMS Output', linestyle='-.')
plt.xlabel('Time', fontsize=14, fontweight='bold')
plt.ylabel('Amplitude', fontsize=14, fontweight='bold')
plt.title('LMS vs NLMS Adaptive Filter: Noise Cancellation', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LMS_vs_NLMS_Denoising.png")
plt.show()

# MSE Comparison
mse_lms = np.mean(elms[M:] ** 2)
mse_nlms = np.mean(enlms[M:] ** 2)
print(f"LMS Mean Squared Error (MSE): {mse_lms:.4f}")
print(f"NLMS Mean Squared Error (MSE): {mse_nlms:.4f}")

# Weight convergence plot
plt.figure(figsize=(12, 5))
plt.plot(wlms, label='Final LMS Weights', marker='o')
plt.plot(wnlms, label='Final NLMS Weights', marker='x')
plt.xlabel('Tap Index', fontsize=14, fontweight='bold')
plt.ylabel('Weight Value', fontsize=14, fontweight='bold')
plt.title('Weight Convergence: LMS vs NLMS', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LMS_vs_NLMS_Weights.png")
plt.show()

# MSE over time
plt.figure(figsize=(12, 5))
plt.plot(elms**2, label='LMS MSE per Sample', alpha=0.7)
plt.plot(enlms**2, label='NLMS MSE per Sample', alpha=0.7)
plt.xlabel('Sample Index', fontsize=14, fontweight='bold')
plt.ylabel('Squared Error', fontsize=14, fontweight='bold')
plt.title('MSE Comparison Over Time', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("LMS_vs_NLMS_MSE.png")
plt.show()
