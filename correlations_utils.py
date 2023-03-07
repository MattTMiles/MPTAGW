#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 21:01:48 2023

@author: dreardon
"""

import numpy as np
from scipy.signal import correlate
from scipy.stats import gaussian_kde
import scipy.special as ss
import healpy as hp
import bilby

def acor(arr):
    arr -= np.mean(arr)
    auto_correlation = correlate(arr, arr, mode='full')
    auto_correlation = auto_correlation[auto_correlation.size//2:]
    auto_correlation /= auto_correlation[0]
    indices = np.where(auto_correlation<0.5)[0]
    if len(indices)>0:
        return indices[0]
    else:
        return 0

def hd_orf(theta):
    """Hellings & Downs spatial correlation function."""
    omc2 = (1 - np.cos(theta * np.pi/180)) / 2
    orf = (1/2) - (1/4) * omc2 + (3/2) * omc2 * np.log(omc2)
    orf[theta == 0] = 0.5
    return orf

def signalResponse_fast(ptheta_a, pphi_a, gwtheta_a, gwphi_a):
    """
    Create the signal response matrix FAST
    """

    # Create a meshgrid for both phi and theta directions
    gwphi, pphi = np.meshgrid(gwphi_a, pphi_a)
    gwtheta, ptheta = np.meshgrid(gwtheta_a, ptheta_a)

    return createSignalResponse(pphi, ptheta, gwphi, gwtheta)

def createSignalResponse(pphi, ptheta, gwphi, gwtheta):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.
    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param gwphi:   Phi of GW propagation direction
    @param gwtheta: Theta of GW propagation direction
    @return:    Signal response matrix of Earth-term
    """
    Fp = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True)
    Fc = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=False)

    # Pixel maps are lumped together, polarization pixels are neighbours
    F = np.zeros((Fp.shape[0], 2 * Fp.shape[1]))
    F[:, 0::2] = Fp
    F[:, 1::2] = Fc

    return F

def createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True, norm=True):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.
    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param gwphi:   Phi of GW propagation direction
    @param gwtheta: Theta of GW propagation direction
    @param plus:    Whether or not this is the plus-polarization
    @param norm:    Normalise the correlations to equal Jenet et. al (2005)
    @return:    Signal response matrix of Earth-term
    """
    # Create the unit-direction vectors. First dimension
    # will be collapsed later. Sign convention of Gair et al. (2014)
    Omega = np.array([-np.sin(gwtheta) * np.cos(gwphi), -np.sin(gwtheta) * np.sin(gwphi), -np.cos(gwtheta)])

    mhat = np.array([-np.sin(gwphi), np.cos(gwphi), np.zeros(gwphi.shape)])
    nhat = np.array([-np.cos(gwphi) * np.cos(gwtheta), -np.cos(gwtheta) * np.sin(gwphi), np.sin(gwtheta)])

    p = np.array([np.cos(pphi) * np.sin(ptheta), np.sin(pphi) * np.sin(ptheta), np.cos(ptheta)])

    # There is a factor of 3/2 difference between the Hellings & Downs
    # integral, and the one presented in Jenet et al. (2005; also used by Gair
    # et al. 2014). This factor 'normalises' the correlation matrix.
    npixels = Omega.shape[2]
    if norm:
        # Add extra factor of 3/2
        c = np.sqrt(1.5) / np.sqrt(npixels)
    else:
        c = 1.0 / np.sqrt(npixels)

    # Calculate the Fplus or Fcross antenna pattern. Definitions as in Gair et
    # al. (2014), with right-handed coordinate system
    if plus:
        # The sum over axis=0 represents an inner-product
        Fsig = (
            0.5 * c * (np.sum(nhat * p, axis=0) ** 2 - np.sum(mhat * p, axis=0) ** 2) / (1 - np.sum(Omega * p, axis=0))
        )
    else:
        # The sum over axis=0 represents an inner-product
        Fsig = c * np.sum(mhat * p, axis=0) * np.sum(nhat * p, axis=0) / (1 - np.sum(Omega * p, axis=0))

    return Fsig

def almFromClm(clm):
    """
    Given an array of clm values, return an array of complex alm valuex
    Note: There is a bug in healpy for the negative m values. This function
    just takes the imaginary part of the abs(m) alm index.
    """
    maxl = int(np.sqrt(len(clm))) - 1

    nalm = hp.Alm.getsize(maxl)
    alm = np.zeros((nalm), dtype=np.complex128)

    clmindex = 0
    for ll in range(0, maxl + 1):
        for mm in range(-ll, ll + 1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))

            if mm == 0:
                alm[almindex] += clm[clmindex]
            elif mm < 0:
                alm[almindex] -= 1j * clm[clmindex] / np.sqrt(2)
            elif mm > 0:
                alm[almindex] += clm[clmindex] / np.sqrt(2)

            clmindex += 1

    return alm

def mapFromClm_fast(clm, nside):
    """
    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside
    @param clm:     Array of C_{lm} values (inc. 0,0 element)
    @param nside:   Nside of the healpix pixelation
    return:     Healpix pixels
    Use Healpix spherical harmonics for computational efficiency
    """
    maxl = int(np.sqrt(len(clm))) - 1
    alm = almFromClm(clm)

    h = hp.alm2map(alm, nside, maxl, verbose=False)

    return h

def real_sph_harm(mm, ll, phi, theta):
    """
    The real-valued spherical harmonics.
    """
    if mm > 0:
        ans = (1.0 / np.sqrt(2)) * (ss.sph_harm(mm, ll, phi, theta) + ((-1) ** mm) * ss.sph_harm(-mm, ll, phi, theta))
    elif mm == 0:
        ans = ss.sph_harm(0, ll, phi, theta)
    elif mm < 0:
        ans = (1.0 / (np.sqrt(2) * complex(0.0, 1))) * (
            ss.sph_harm(-mm, ll, phi, theta) - ((-1) ** mm) * ss.sph_harm(mm, ll, phi, theta)
        )

    return ans.real

def getCov(clm, nside, F_e):
    """
    Given a vector of clm values, construct the covariance matrix
    @param clm:     Array with Clm values
    @param nside:   Healpix nside resolution
    @param F_e:     Signal response matrix
    @return:    Cross-pulsar correlation for this array of clm values
    """
    # Create a sky-map (power)
    # Use mapFromClm to compare to real_sph_harm. Fast uses Healpix
    # sh00 = mapFromClm(clm, nside)
    sh00 = mapFromClm_fast(clm, nside)

    # Double the power (one for each polarization)
    sh = np.array([sh00, sh00]).T.flatten()

    # Create the cross-pulsar covariance
    hdcov_F = np.dot(F_e * sh, F_e.T)

    # The pulsar term is added (only diagonals: uncorrelated)
    return hdcov_F + np.diag(np.diag(hdcov_F))

def anis_basis(psr_locs, lmax, nside=32):
    """
    Calculate the correlation basis matrices using the pixel-space
    transormations
    @param psr_locs:    Location of the pulsars [phi, theta]
    @param lmax:        Maximum l to go up to
    @param nside:       What nside to use in the pixelation [32]
    Note: GW directions are in direction of GW propagation
    """
    pphi = psr_locs[:, 0]
    ptheta = psr_locs[:, 1]

    # Create the pixels
    npixels = hp.nside2npix(nside)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    gwtheta = pixels[0]
    gwphi = pixels[1]

    # Create the signal response matrix
    F_e = signalResponse_fast(ptheta, pphi, gwtheta, gwphi)

    # Loop over all (l,m)
    basis = []
    nclm = (lmax + 1) ** 2
    clmindex = 0
    for ll in range(0, lmax + 1):
        for mm in range(-ll, ll + 1):
            clm = np.zeros(nclm)
            clm[clmindex] = 1.0

            basis.append(getCov(clm, nside, F_e))
            clmindex += 1

    return np.array(basis)


def anis_orf(pos1, pos2, params, **kwargs):
    """Anisotropic GWB spatial correlation function."""

    anis_basis = kwargs["anis_basis"]
    psrs_pos = kwargs["psrs_pos"]
    lmax = kwargs["lmax"]

    psr1_index = [ii for ii in range(len(psrs_pos)) if np.all(psrs_pos[ii] == pos1)][0]
    psr2_index = [ii for ii in range(len(psrs_pos)) if np.all(psrs_pos[ii] == pos2)][0]

    clm = np.zeros((lmax + 1) ** 2)
    clm[0] = 2.0 * np.sqrt(np.pi)
    if lmax > 0:
        clm[1:] = params

    return sum(clm[ii] * basis for ii, basis in enumerate(anis_basis[: (lmax + 1) ** 2, psr1_index, psr2_index]))

def monopole(theta):
    """Monopole spatial correlation function."""
    return np.ones(np.shape(theta))

def dipole(theta):
    """Dipole spatial correlation function."""
    return np.cos(theta * np.pi/180)

def plot_violin(ax, pos, chain, width=22.5, colour='darkgreen', alpha=1, edgecolour=None, linewidth=0, ylabel='Correlation coefficient'):

    ax.set_xlabel('Sky separation (deg)')
    ax.set_ylabel(ylabel)

    violin_dict = ax.violinplot(chain, positions = [pos], widths=width, showextrema=False, ) #, showextrema = True, showmeans = False, showmedians = False)
    for violin_body in violin_dict['bodies']:
        violin_body.set_alpha(alpha)
        violin_body.set_facecolor(colour)
        violin_body.set_edgecolor(edgecolour)
        violin_body.set_linewidth(linewidth)

    return ax

def get_pairs(psrnames):

    ipair = 1
    pairs = {}

    for i in range(0, len(psrnames)):
        for j in range(0, len(psrnames)):
            if j >= i:
                continue
            pairs[str(ipair)] = [psrnames[i], psrnames[j]]
            ipair += 1

    return pairs

def get_psrnames():
    psrnames = ['J0030+0451',
                'J0125-2327',
                'J0437-4715',
                'J0613-0200',
                'J0614-3329',
                'J0711-6830',
                'J0900-3144',
                'J1017-7156',
                'J1022+1001',
                'J1024-0719',
                'J1045-4509',
                'J1125-6014',
                'J1446-4701',
                'J1545-4550',
                'J1600-3053',
                'J1603-7202',
                'J1643-1224',
                'J1713+0747',
                'J1730-2304',
                'J1744-1134',
                'J1832-0836',
                'J1857+0943',
                'J1902-5105',
                'J1909-3744',
                'J1933-6211',
                'J1939+2134',
                'J2124-3358',
                'J2129-5721',
                'J2145-0750',
                'J2241-5236']
    return psrnames













