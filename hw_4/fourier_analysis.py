# HW4 - Fourier Analysis 

# Put this file in the same folder as your .fits file, then run python hw4_fourier.py
#
# This file uses the default epoch (3312 to 3339). To pick a different epoch, give the start and end times:
#
# python hw4_fourier.py --start 3452 --end 3519
#
# To use a different .fits file, add --file:
#
# python hw4_fourier.py --file tic0001234567.fits
#
# Each plot pops up in a window and also saved as .png files in the same folder.

import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# How many Fourier coefficients to keep in the inverse transform
N_values = [5, 20, 100, 500]


def fourier_analysis(filename, t_start, t_end):
    """
    Do a Fourier analysis on one epoch of a TESS light curve.

    Reads the light curve from a .fits file then you choose one epoch between
    t_start and t_end, takes the Fourier transform, and plots the power
    spectrum. Shows the lightcure with the inverse transform and fills in missing time
    steps with linear interpolation and redoes the analysis.

    Parameters: 
    filename :  Name of the .fits file as a string
    t_start : Start time of the epoch (days)
    t_end : End time of the epoch (days)

    Returns : Plots are shown on screen and saved as .png files.
    """
    #  Read the file
    hdul = fits.open(filename)
    times = hdul[1].data['times']
    fluxes = hdul[1].data['fluxes']
    ferrs = hdul[1].data['ferrs']

    # Plot the whole light curve -> help choose epoch
    plt.figure(figsize=(12, 4))
    plt.plot(times, fluxes, '.', markersize=1)
    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.title("Full light curve")
    plt.savefig("full_lightcurve.png")
    plt.show()

    #the epoch we chose
    keep = (times > t_start) & (times < t_end)
    t = times[keep]
    f = fluxes[keep]
    print("Points in this epoch:", len(t))

    # Plot just the epoch
    plt.figure(figsize=(12, 4))
    plt.plot(t, f, '.', markersize=1)
    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.title("Light curve for our epoch (" + str(t_start) + " to " + str(t_end) + ")")
    plt.savefig("epoch_lightcurve.png")
    plt.show()

    # Fourier transform and power spectrum
    dt = np.median(np.diff(t))              # time between observations (days)
    print("Time step:", dt * 24 * 60, "minutes")

    coeffs = np.fft.rfft(f - np.mean(f))    # Fourier coefficients
    freqs = np.fft.rfftfreq(len(f), dt)     # frequencies 
    power = np.abs(coeffs) ** 2             # power spectrum

    biggest = np.argmax(power)
    print("Strongest frequency:", freqs[biggest], "per day")
    print("That is a period of:", 1 / freqs[biggest], "days")

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, power)
    plt.yscale("log")
    plt.xlim(0, 20)
    plt.xlabel("Frequency (cycles per day)")
    plt.ylabel("Power")
    plt.title("Power spectrum")
    plt.savefig("power_spectrum.png")
    plt.show()

    #Inverse transform using N coefficients
    #keep the N biggest coefficients and set the rest to zero.
    print("Total number of coefficients:", len(coeffs))

    plt.figure(figsize=(12, 5))
    plt.plot(t, f, '.', color='gray', label="data")
    for N in N_values:
        small = np.zeros_like(coeffs)
        top_N = np.argsort(np.abs(coeffs))[-N:]   # positions of the N biggest
        small[top_N] = coeffs[top_N]
        f_back = np.fft.irfft(small, len(f)) + np.mean(f)
        plt.plot(t, f_back, label="N = " + str(N))
    plt.xlim(t[0], t[0] + 3)                      # zoom in on the first 3 days
    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.title("Inverse transform ")
    plt.legend()
    plt.savefig("inverse_transform.png")
    plt.show()

    # find the missing time steps and fill them by interpolation
    gaps = np.diff(t)
    n_missing = np.sum(np.round(gaps / dt) - 1)
    print("Number of missing time steps:", int(n_missing))

    t_even = np.arange(t[0], t[-1], dt)       # perfectly evenly spaced times
    f_even = np.interp(t_even, t, f)          # linear interpolation

    # Redo the Fourier analysis on the interpolated data
    coeffs2 = np.fft.rfft(f_even - np.mean(f_even))
    freqs2 = np.fft.rfftfreq(len(f_even), dt)
    power2 = np.abs(coeffs2) ** 2

    biggest2 = np.argmax(power2)
    print("After interpolation, strongest frequency:", freqs2[biggest2], "per day")
    print("That is a period of:", 1 / freqs2[biggest2], "days")

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, power, label="original")
    plt.plot(freqs2, power2, label="interpolated", alpha=0.7)
    plt.yscale("log")
    plt.xlim(0, 20)
    plt.xlabel("Frequency (cycles per day)")
    plt.ylabel("Power")
    plt.title("Power spectrum: original vs interpolated")
    plt.legend()
    plt.savefig("power_spectrum_compare.png")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(t_even, f_even, '.', color='gray', label="interpolated data")
    for N in N_values:
        small = np.zeros_like(coeffs2)
        top_N = np.argsort(np.abs(coeffs2))[-N:]
        small[top_N] = coeffs2[top_N]
        f_back = np.fft.irfft(small, len(f_even)) + np.mean(f_even)
        plt.plot(t_even, f_back, label="N = " + str(N))
    plt.xlim(t_even[0], t_even[0] + 3)
    plt.xlabel("Time (days)")
    plt.ylabel("Flux")
    plt.title("Inverse transform (interpolated data)")
    plt.legend()
    plt.savefig("inverse_transform_interpolated.png")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fourier analysis of a TESS light curve")
    parser.add_argument("--start", type=float, default=3312, help="start time of the epoch")
    parser.add_argument("--end", type=float, default=3339, help="end time of the epoch")
    parser.add_argument("--file", default="tic0007854182.fits", help="name of the .fits file")
    args = parser.parse_args()

    fourier_analysis(args.file, args.start, args.end)