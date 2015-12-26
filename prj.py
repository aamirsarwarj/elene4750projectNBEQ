#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Module: nbeq.py
#
# Rashad Barghouti (UNI:rb3074)
# AM Sarwar Jahan (UNI:aj2599)
# ELEN E4750, Fall 2015
#------------------------------------------------------------------------------

# System imports
import sys
import os
import time
import pyopencl as cl
import pyopencl.array
#from pyfft.cl import Plan
import numpy as np
from scipy import signal
# To avoid "backend has already been chosen error," set GPU platform before
# import pyplot
if 'rb3074' in os.environ['HOME']:
    GPU = 'TESSERACT_K40'
    import matplotlib as mpl
    mpl.use('agg')
else:
    GPU = 'HOME_QUADRO'
import matplotlib.pyplot as plt

# Local defs and imports
PLOT_MAGNITUDE_SPECTRA = True
PLOT_POWER_SPECTRA = True
SHOW_PLOTS = False
PRINT_KERNELS = False

def _main():

    #**************************************************************************
    #************************ Numpy Modem Processing **************************
    #                    (Move to own module/class later)
    #**************************************************************************
    #**************************************************************************

    constellQAM4  = np.array([-1-1j, -1+1j,  1-1j,  1+1j], dtype=np.complex64)
    constellQAM16 = np.array([-3-3j, -3-1j, -3+3j, -3+1j,
                              -1-3j, -1-1j, -1+3j, -1+1j,
                               3-3j,  3-1j,  3+3j,  3+1j,
                               1-3j,  1-1j,  1+3j,  1+1j], dtype=np.complex64)

    SNRs = [30, 25, 20, 17, 14, 11, 8]

    # Define the channel as a 3-tap notch filter that produces a deep fade at
    # center frequencies and an attenuated 2nd path
    channel = np.array([1, 0, 0.5]).astype(np.float32)

    #####################
    #### Transmitter ####
    #####################

    snrdB = 10
    frmlen = 500
    constell = constellQAM4
    bitspersym = 4 if constell.size == 16 else 2

    # Generate user data and corresponding QAM frame
    txfrm, txdata, txdatabinary = QAMmodulate(frmlen, constell)

    # Fitler Tx frame with channel FIR and compute output signal energy/symbol
    # Note: returned data type from lfilter() is double (complex128), so need
    # to downcast to complex64. Otherwise GPU ops will be on smaller array size
    # than desired
    chfrm = signal.lfilter(channel, [1, 0, 0], txfrm).astype(np.complex64)
    chEs = np.mean(np.abs(chfrm)**2)
    chEsdBm = 10*np.log10(chEs)

    # Add AWGN to produce the received frame; N0 is the equivalent lowpass
    # noise PSD
    rxfrm, N0 = awgn(chfrm, snrdB, chEs)
    rxEs = np.mean(np.abs(rxfrm)**2)
    rxEsdBm = 10.0*np.log10(rxEs)

    np.set_printoptions(precision=2)
    print 'Avg symbol energy before AWGN: chEs: {:.3f} dBm'.format(chEsdBm)
    print 'Avg. received symbol energy: rxEs: {:.3f} dBm'.format(rxEsdBm)
    print 'One-sided noise PSD (N0): {:.3f}'.format(N0)

    ##################
    #### Receiver ####
    ##################

    # Compute channel and equalizers' FFTs and plot their magnitude spectra
    fftlen = 2**np.int32(np.log2(frmlen*2-1)+1)
    chfft = np.fft.fft(channel, n=fftlen)
    ZFfft = np.reciprocal(chfft)
    MMSEfft = np.reciprocal(chfft+N0)

    # Filter the received data 
    rxfrmfft = np.fft.fft(rxfrm, fftlen)
    MMSEoutfft = rxfrmfft * MMSEfft
    ZFoutfft = rxfrmfft * ZFfft
    MMSEout = np.fft.ifft(MMSEoutfft)
    ZFout = np.fft.ifft(ZFoutfft)
    '''
    print 'rxfrm ', rxfrm.shape
    print 'rxfrmfft ', rxfrmfft
    print 'fftlen ', fftlen
    print 'MMSEfft ' , MMSEfft.shape
    print 'MMSEoutfft ', MMSEoutfft.shape
    '''
    ocl_result = opencl_compute(rxfrm, fftlen, MMSEfft, ZFfft, MMSEoutfft, ZFoutfft, rxfrmfft)
    print 'ocl_result = ', ocl_result
    '''
    #print 'rxfrmfft = ', rxfrmfft
    #print 'MMSEfft = ', MMSEfft
    print 'MMSEoutfft = ', MMSEoutfft
    print 'mul_result' , mul_result
    print mul_result.shape
    '''
    # Demodulate received symbols to recover user data
    MMSEdata, MMSEdatabinary = QAMdemodulate(MMSEout[:frmlen], constell)
    ZFdata, ZFdatabinary = QAMdemodulate(ZFout[:frmlen], constell)

    # Calculate bit error rates (BER).
    nbits = np.float32(len(txdatabinary))
    MMSEbiterrors = np.sum(np.abs(txdatabinary-MMSEdatabinary))
    ZFbiterrors = np.sum(np.abs(txdatabinary-ZFdatabinary))
    print 'SNR: {}dB - MMSE Bit Error Rate (BER) = {}/{} = {:.2f}%'.format(
            snrdB, MMSEbiterrors, nbits, (MMSEbiterrors/nbits)*100)
    print 'SNR: {}dB - ZF Bit Error Rate (BER): {}/{} = {:.2f}%'.format(
            snrdB, ZFbiterrors, nbits, (ZFbiterrors/nbits)*100)


    #**************************************************************************
    #************************** GPU Processing ********************************
    #**************************************************************************

    # Set up platform
    #ctx, cq = init_ocl_runtime()

    ## Create read-only buffers and copy equalizers' coefficients to them
    #mf = cl.mem_flags
    #d_ZFfft = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ZFfft)
    #d_MMSEfft = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MMSEfft)
    #cl.enqueue_copy(cq, d_ZFfft, ZFfft)
    #cl.enqueue_copy(cq, d_MMSEfft, MMSEfft)

    ## Read kernels and create a program object
    #with open("kernels.cl", "r") as fp:
    #    kernels_src = fp.read()
    #    if PRINT_KERNELS == True:
    #        print kernels_src

    #gpu = cl.Program(ctx, kernels_src).build()

    ## For GPU testing only
    ##for i in xrange(rxfrm.size):
    ##    rxfrm[i] = complex(np.float32(i), np.float32(i+1))

    ## Create device input and output buffers
    #d_rxfrm = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rxfrm)
    #d_obuf = cl.Buffer(ctx, mf.WRITE_ONLY, size=rxfrm.nbytes)

    ## Allocate local memory here instead of inside the kernel to avoid
    ## "demotion' by the compiler. (See MyNotes.docx.)
    ## 
    #d_lmem = cl.LocalMemory(frmlen*np.dtype('complex64').itemsize)

    ## Write PTX code to file. Note: prg.binaries() returns a list object
    ## containing the lines of the PTX source
    #ptx = gpu.binaries
    #with open("gpu.ptx", "wb") as fp:
    #    fp.write(''.join(ptx))

    ## Run GPU kernel
    #walltime = time.clock()
    #evt = gpu.r4DITfft(cq, (16, 1), (16, 1), d_rxfrm, d_obuf, d_lmem)
    #evt.wait()
    #d_out = np.empty_like(rxfrm)
    #cl.enqueue_copy(cq, d_out, d_obuf)
    #walltime = time.clock() - walltime
    #np.set_printoptions(precision=3, linewidth=80, threshold=rxfrm.size)
    #print 'Equal: ', np.allclose(rxfrm, d_out)
    #gtime = 1e-9*(evt.profile.end-evt.profile.start)
    #print 'Wall time: {:.3f}, GPU time: {:.3f} msec'.format(walltime, 1e3*gtime)
    #print 'rxfrm:\n', rxfrm
    #print 'd_out:\n', d_out

    #************************* Done GPU Processing ****************************
    #**************************************************************************

    # Plot various spectra for report
    plots_generated = False
    if PLOT_MAGNITUDE_SPECTRA == True:
        plot_magnitude_spectra(chfft, ZFfft, MMSEfft)
        plt.gcf()
        fname = 'mag' + str(snrdB) + 'dB.png'
        plt.savefig(fname)
        plots_generated = True

    if PLOT_POWER_SPECTRA == True:
        plot_power_spectra(txfrm, rxfrm, ZFout, MMSEout, fftlen)
        fname = 'psds' + str(snrdB) + 'dB.png'
        plt.savefig(fname)
        plots_generated = True

    # plt.show() call is blocking, so keep it as last statement in program 
    if plots_generated == True and SHOW_PLOTS == True:
        plt.show()
#------------------------------------------------------------------------------
# init_ocl_runtime(pltname='NVIDIA CUDA')
#   Sets up OpenCL runtime.
# Returns
#    Tuple: (context, queue)
#------------------------------------------------------------------------------

def opencl_compute(rxfrm, fftlen, MMSEfft, ZFfft, MMSEoutfft, ZFoutfft, rxfft):

    #Selecting OpenCL platform;
    NAME = 'NVIDIA CUDA'
    platforms = cl.get_platforms()
    devs = None
    for platform in platforms:
        if platform.name == NAME:
                devs = platform.get_devices()

    #Create context on found list of platforms
    ctx = cl.Context(devs)
    queue = cl.CommandQueue(ctx,
        properties = cl.command_queue_properties.PROFILING_ENABLE)

    #Setup memory flags
    #Input Buffers
    mf=cl.mem_flags
    rxfft_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rxfft)
    rxfrm_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rxfrm)
    MMSEfft_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MMSEfft)
    ZFfft_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ZFfft)
    
    #Output Buffers. 
    MMSEoutfft_buf = cl.Buffer(ctx, mf.WRITE_ONLY, MMSEoutfft.nbytes)
    ZFoutfft_buf = cl.Buffer(ctx, mf.WRITE_ONLY, ZFoutfft.nbytes)
    MMSEout_buf = cl.Buffer(ctx, mf.WRITE_ONLY, MMSEfft.nbytes)
    ZFout_buf = cl.Buffer(ctx, mf.WRITE_ONLY, ZFfft.nbytes)

    
    #rxfft = np.fft.fft(rxframe, fftlen)

    prg = cl.Program(ctx, """
//Declarations
#define USE_MAD 1

#if CONFIG_USE_DOUBLE

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
// double
typedef double real_t;
typedef double2 real2_t;
#define FFT_PI 3.14159265358979323846
#define FFT_SQRT_1_2 0.70710678118654752440

#else

// float
typedef float real_t;
typedef float2 real2_t;
#define FFT_PI       3.14159265359f
#define FFT_SQRT_1_2 0.707106781187f
#endif

#include <pyopencl-complex.h>

// Return A*B
real2_t mul(real2_t a, real2_t b)
{
#if USE_MAD
  return (real2_t)(mad(a.x, b.x, -a.y * b.y), mad(a.x, b.y, a.y * b.x)); // mad
#else
  return (real2_t)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); // no mad
#endif
}

// Return A * exp(K*ALPHA*i)
real2_t twiddle(real2_t a, int k, real_t alpha)
{
  real_t cs,sn;
  sn = sincos((real_t)k * alpha, &cs);
  return mul(a, (real2_t)(cs, sn));
}

// In-place DFT-2, output is (a,b). Arguments must be variables.
#define DFT2(a,b) { real2_t tmp = a - b; a += b; b = tmp; }

__kernel void fftRadix2Kernel(__global const real2_t * x,__global real2_t * y,int p)
{
  int t = get_global_size(0); // thread count
  int i = get_global_id(0);   // thread index
  int k = i & (p - 1);            // index in input sequence, in 0..P-1
  int j = ((i - k) << 1) + k;     // output index
  real_t alpha = -FFT_PI * (real_t)k / (real_t)p;
  
  // Read and twiddle input
  x += i;
  real2_t u0 = x[0];
  real2_t u1 = twiddle(x[t], 1, alpha);

  // In-place DFT-2
  DFT2(u0, u1);

  // Write output
  y += j;
  y[0] = u0;
  y[p] = u1;
}

__kernel void multiply(__global const cfloat_t *rxfft, __global const cfloat_t *in2, __global cfloat_t *out){
int gid = get_global_id(0);
out[gid] = cfloat_mul(rxfft[gid], in2[gid]);

}

""").build()

    fft_result = np.empty((fftlen,), dtype= rxfrm.dtype)
    prg.fftRadix2Kernel(queue, (fftlen, ), None, rxfrm_buf, rxfft_buf, np.uint32(fftlen))
    cl.enqueue_copy(queue, fft_result, rxfft_buf)

    MMSEmul_result = np.empty((fftlen, ), dtype= rxfrm.dtype)
    prg.multiply(queue, (fftlen,), None, rxfft_buf, MMSEfft_buf, MMSEoutfft_buf)
    cl.enqueue_copy(queue, MMSEmul_result, MMSEoutfft_buf)

    ZFmul_result = np.empty((fftlen, ), dtype= rxfrm.dtype)
    prg.multiply(queue, (fftlen,), None, rxfft_buf, ZFfft_buf, ZFoutfft_buf)
    cl.enqueue_copy(queue, ZFmul_result, ZFoutfft_buf)

    #MMSEout_ocl = np.fft.ifft(MMSEmul_result)
    #ZFout_ocl = np.fft.ifft(ZFmul_result)   
   # print 'fftlen: ', fftlen       
    #print np.allclose(mul_result, MMSEoutfft)
    print 'fft =', fft_result
   # print 'MMSEmul_result ', MMSEmul_result
   # print 'MMSEout_ocl' , MMSEout_ocl
    return (MMSEmul_result, ZFmul_result)

def init_ocl_runtime(pltname='NVIDIA CUDA'):

    platforms = cl.get_platforms()
    devs = None
    for platform in platforms:
        if platform.name == pltname:
            devs = platform.get_devices()

    # Set up command queue and enable GPU profiling
    context = cl.Context(devs)
    queue = cl.CommandQueue(context,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    return (context, queue)

#------------------------------------------------------------------------------
# QAMmodulate(frmlen, constell)
#   This function performs QAM modulation by generating random user data and
#   mapping to symbols in the given constellation.
# Inputs:
#   frmlen: Output frame size
#   constell: QAM constellation array
# Returns:
#   1. QAM-modulated data in a complex64 array 
#   2. User data in decimal and binary int8 arrays. (Binary bits must have an
#      unsigned integer type so that overflow is avoided in receiver when
#      computing bit errors. The operation abs(y-x) will overflow when
#      y = uint8(1) and x = uint8(0).
#------------------------------------------------------------------------------
def QAMmodulate(frmlen, constell):

    txdata = np.random.randint(constell.size, size=frmlen).astype(np.int8)
    txfrm = constell[txdata]

    return txfrm, txdata, dec2bin(txdata)

#------------------------------------------------------------------------------
# QAMdemodulate(rxfrm, constell)
#   This function performs QAM demodulation using the given constellation.
#
# Inputs:
#   rxfrm: Received frame array
#   constell: QAM constellation array
#
# Returns:
#   Demodulated bits in an int8 array. (Bits must have an unsigned integer type
#   so that overflow is avoided in receiver when computing bit errors. The
#   operation abs(y-x) will overflow when y = uint8(1) and x = uint8(0).
#------------------------------------------------------------------------------
def QAMdemodulate(rxfrm, constell):

    # A demodulated value is the index of the constellation symbol to which
    # it's closest
    demoddata = np.array(map(lambda i: np.argmin(np.abs(rxfrm[i]-constell)),
                    xrange(rxfrm.size))).astype(np.int8)

    return demoddata, dec2bin(demoddata)

#------------------------------------------------------------------------------
# def dec2bin(ddata, M=16)
#   Converts an array of decimal symbol values into a binary stream
#
#   Inputs:
#       ddata: Integer array of decimal data
#       M: QAM constellation size. Either 16 (default) or 4
#
#   Returns:
#       Array of binary data with the same type as the input.
#------------------------------------------------------------------------------
def dec2bin(decdata, M=16):

    # Init # of symbols packed in a byte
    ns = 2
    if M == 4: ns = 4

    # Record input type
    dt = decdata.dtype
    decdata = decdata.astype(np.uint8)
    bindata = np.unpackbits(decdata).reshape(decdata.size, 8)
    arylist = np.hsplit(bindata, ns)

    # Return output in same type as input
    return arylist[ns-1].flatten().astype(dt)
#------------------------------------------------------------------------------
# plot_magnitude_spectra(chfft, ZFfft, MMSEfft)
#   This routine plots the magnitude spectra of the input frequency responses
#------------------------------------------------------------------------------
def plot_magnitude_spectra(chfft, ZFfft, MMSEfft):

    # Since channel is a real sequence, its magnitude response symetric, and we
    # need only plot half the data
    fftlen = chfft.size/2
    freqband = (np.arange(fftlen, dtype=np.float32)/fftlen)*1000

    # Create a new figure
    plt.figure()

    plt.title('Magnitude Responses of Channel and Equalizers') 
    #plt.title('Channel Magnitude Response') 
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('Magnitude Response')
    plt.plot(freqband, np.abs(chfft[:fftlen]), 'b', label='Channel', lw=3)
    plt.plot(freqband, np.abs(ZFfft[:fftlen]), 'r', lw=2,
            label='ZF Equalizer')
    plt.plot(freqband, np.abs(MMSEfft[:fftlen]), 'g', lw=2, 
            label='MMSE Equalizer')
    plt.legend(loc=2, fontsize='medium')
    plt.grid(True)

    return

#------------------------------------------------------------------------------
# plot_power_spectra(qampsd, rxfrm, ZFout, MMSEout)
#   This routine computes and plots the power spectral densities for the
#   vectors passed in the input arguments
#------------------------------------------------------------------------------
def plot_power_spectra(txfrm, rxfrm, ZFout, MMSEout, fftlen):

    # Create new figure
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.05, top=0.95, hspace=0.40)

    # Since all input data are complex, the output PSDs will be two-sided, and
    # we need only examine one of them
    len = fftlen/2
    freqband = (np.arange(len, dtype=np.float32)/len)*1000

    f, qampsd = signal.welch(txfrm, nfft=fftlen)
    plt.subplot(411)
    plt.title('Spectrum of Transmitted QAM Symbols (Channel Input)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.plot(freqband, qampsd[:len])
    plt.grid(True)

    f, rxpsd = signal.welch(rxfrm, nfft=fftlen)
    plt.subplot(412)
    plt.title('Spectrum of Channel Output (Received Data)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.plot(freqband, rxpsd[:len])
    plt.grid(True)

    f, ZFoutpsd = signal.welch(ZFout, nfft=fftlen)
    plt.subplot(413)
    plt.title('Spectrum of ZF Equalizer Output')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.plot(freqband, ZFoutpsd[:len])
    plt.grid(True)

    f, MMSEoutpsd = signal.welch(MMSEout, nfft=fftlen)
    plt.subplot(414)
    plt.title('Spectrum of MMSE Equalizer Output')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.plot(freqband, MMSEoutpsd[:len])
    plt.grid(True)

    return
#------------------------------------------------------------------------------
# awgn(idata, snrdB, Es=None)
#   This routine computes Additive White Gaussian Noise (AWGN) based on the
#   desired output SNR and adds it to the input data.
#
# Input:
#   idata: Input frame data
#   snrdB: Desired output frame SNR in dB
#   Es: Average signal energy per symbol; (if None, value will be computed)
#
# Return:
#   odata = idata + noise 
#   N0: one-sided AWGN PSD (see notes below)
#
# Algorithm:
#   Relationship between Es/N0 and snrdB is: Es/N0 = 10log10(Tsym/Tsamp)+snrdB.
#   Since no oversampling is done (i.e., Tsym=Tsamp), Es/N0 = snrdB.
#
#       Calculations:
#           Es = sum(abs(idata)**2)/len(idata)
#           snrLinear = 10**(snrdB/10)
#           N0lp = Es/snrLinear
#           nreal = sqrt(N0lp/2)*randn(len(idata))
#           nimag = sqrt(N0lp/2)*randn(len(idata))*1.j
#           ndata = nreal + nimage
#           return idata+ndata, N0lp
#   Notes:
#       (1) For a WGN PSD of N0/2, the lowpass-equivalent PSD is 2*N0, i.e., 4x
#       the double-sided value (see Proakis 5th ed., p.81). To produce a
#       complex AWGN sequence, the real and imaginary parts are each generated
#       with variance=N0, since for i.i.d. complex samples xn = xn.r + xn.i*j,
#       var(xn) = var(xn.r) + var(xn.i). 
#
#       (2) EsN0 (dB) = EbN0 (dB) + 10log10(k), where k = # bits/symbol. k
#       is usually a combination of bits/modulation symbol + any redundancy
#       bits introduced by subsequent coding.
#------------------------------------------------------------------------------
def awgn(idata, snrdB, Es=None):

    # If not already done, calculate average energy
    if Es == None:
        Es = np.mean(np.abs(idata)**2)

    N0 = Es/10**(snrdB/10.0)/2.0
    ndata = np.sqrt(N0) * np.random.randn(idata.size).astype(idata.dtype) + \
            np.sqrt(N0) * np.random.randn(idata.size).astype(idata.dtype)*1.j

    # Get variance of generated samples. The value should be 2*N0 computed
    # above
    #print 'awgn(): LP N0 {:.2f}\n'.format(np.var(ndata))

    return (idata+ndata, N0)

#------------------------------------------------------------------------------
# Define the entry point to the program
#------------------------------------------------------------------------------
if __name__ == '__main__':
    _main()
