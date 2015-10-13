import scipy as s
import matplotlib.pyplot as pl
import scipy.fftpack
import scipy.signal

# functions assume that inputs are real


def cconv(x1,x2):
    X1 = s.fftpack.fft(x1)
    X2 = s.fftpack.fft(x2)
    return s.fftpack.ifft(X1*X2) 

# Lag/smoothing windows (use smooth if convolving with SDF or lag if multiplying
# ACVS) 
# bandwidth = 1.85*Fs/m
# d.o.f. = 3.71*N/(m*1.94)   (for Hann window)
def parzen(N,m,Fs,wType = 'smooth'):
    w = s.zeros(2*N-1)
    t = s.arange(0,m+1)
    w[0:m/2+1] = 1 - 6*(t[0:m/2+1]/m)**2 + 6*(t[0:m/2+1]/m)**3
    w[m/2+1:m] = 2*(1-t[m/2+1:m]/m)**3;
    flip = w[::-1]
    w[-m:] = flip[-m-1:-1]
    if wType == 'lag':
        return w
    else:
        return 1/Fs*s.real(s.fftpack.fft(w))
 
# Lag/smoothing windows (use smooth if convolving with SDF or lag if multiplying
# ACVS) 
# bandwidth = Fs/m   
# d.o.f. = 2*N/(m*1.94)   (for Hann window)
def daniell(N,m,Fs,wType = 'smooth'):
    w = s.zeros(2*N-1)
    t = s.arange(0,N)
    w[:N] = s.sin(s.pi*t/m)/(s.pi*t/m)
    
    flip = w[::-1]
    w[N:] = flip[N-1:-1]
    w[0] = 1
    if wType == 'lag':
        return w
    else:
        return 1/Fs*s.real(s.fftpack.fft(w))

# Lag/smoothing windows (use smooth if convolving with SDF or lag if multiplying
# ACVS) 
# bandwidth = 0.80*Fs/m
# d.o.f. = 1.60*N/(m*1.94)   (for Hann window) 
def gaussian(N,m,Fs,wType = 'smooth'):
    w = s.zeros(2*N-1)
    t = s.arange(0,N)
    w[:N] = s.exp(-t**2/m**2)
    flip = w[::-1]
    w[N:] = flip[N-1:-1]
    if wType == 'lag':
        return w
    else:
        return 1/Fs*s.real(s.fftpack.fft(w))

# Lag/smoothing windows (use smooth if convolving with SDF or lag if multiplying
# ACVS) 
# bandwidth = 1.5*Fs/m
# d.o.f. = 3*N/(m*1.94)   (for Hann window)         
def bartlett(N,m,Fs,wType = 'smooth'):
    w = s.zeros(2*N-1)
    t = s.arange(0,N)
    w[:m] = 1-t[:m]/m
    flip = w[::-1]
    w[-m:] = flip[-m-1:-1]
    if wType == 'lag':
        return w
    else:
        return 1/Fs*s.real(s.fftpack.fft(w))

# Bandwidth for data window convolved w/ lag window (input windows shouldn't be normalized)
def lagwindow_bw(window,lag_window,Fs):
    w_norm = window/s.sqrt(s.sum(window**2))
    l_shift = s.roll(lag_window,len(window)-1)
    return Fs/s.sum((l_shift*s.signal.correlate(w_norm,w_norm))**2)
    
# Estimates spectral density using smoothing window. Estimate consists of direct
# spectral estimate convolved with a smoothing window.  Use to decrease variance
# of estimate.  See smoothing window functions for bandwith/d.o.f.
# (100-2*p)% CI(dB) = 10*log10[v/Q_v(p)-v/Q_v(1-p)]     
def lag_window_PSD(x, Fs, window, smooth_window, scale_by_freq = True, two_sided = False, detrend = True):
    N = len(x)
    # Remove mean and pad to make length 2*N-1
    if detrend:
        X = s.hstack((x-s.mean(x),s.zeros(N-1)))
    else:
        X = s.hstack((x,s.zeros(N-1)))
        
    # Pad data window to make it length 2*N-1
    W = s.hstack((window,s.zeros(N-1)))
    
    # Calculate two-sided periodogram (necessary for convolution in freq. domain)
    SDF = periodogram(X,Fs,W,scale_by_freq,two_sided = True,detrend = False)  
    
    # Convolve in freq. domain (ideally shouldn't need abs, but doesn't quite 
    # return real-valued function - non symmetric due to rounding errors?)
    S = abs(cconv(SDF,smooth_window/s.sum(smooth_window)))
    
    if two_sided:
        return S
    else:
        return S[0:N]

# Bandwidth for data window (input window shouldn't be normalized)
def datawindow_bw(window,Fs):
    w_norm = window/s.sqrt(s.sum(window**2))
    return Fs/s.sum(s.signal.correlate(w_norm,w_norm)**2)
    
# Basic estimator of spectral density function if window is rectangular but can be biased 
# (look for non-constant variance when plotted on dB scale - should be about 20dB througout,
# compare to direct estimate). Becomes direct spectral estimate when window is not rectangular.
# bandwidth = data window bandwidth (see data window bandwidth functions)
# d.o.f. = 2
# (100-2*p)% CI(dB) = 10*log10[log(p)/log(1-p)]     
def periodogram(x, Fs, window, scale_by_freq = True, two_sided = False, detrend = True):
    # Remove mean
    if detrend:
        m = s.mean(x)
        t = s.fftpack.fft((x-m)*window)
    else:
        t = s.fftpack.fft(x*window)
    
    # Return spectral density or just spectrum    
    if scale_by_freq:
        S = s.absolute(t)**2/s.sum(window**2)/Fs
    else:
        S = s.absolute(t)**2/s.sum(window**2)
    if two_sided:
        return S
    else:
        return S[:len(x)/2+1]

# Basic estimator of cross-spectral density between two different time series.  Since inputs are
# non-symmetric, will be complex valued.  See notes for periodogram.
def cross_periodogram(x1, x2, Fs, window, scale_by_freq = True, two_sided = False, detrend = True):
    # Remove mean
    if detrend:
        t1 = s.fftpack.fft((x1-s.mean(x1))*window)
        t2 = s.fftpack.fft((x2-s.mean(x2))*window)
    else:
        t1 = s.fftpack.fft(x1*window)
        t2 = s.fftpack.fft(x2*window)
        
    # Return cross spectral density or just spectrum    
    if scale_by_freq:
        csd = t1*s.conj(t2)/s.sum(window**2)/Fs
    else:
        csd = t1*s.conj(t2)/s.sum(window**2)
    if two_sided:
        return csd
    else:
        return csd[:len(x1)/2+1]

# Magnitude-squared coherence using welch's method for spectral density and cross-spectral density.
# Note that nDT should not be same as the length of the two time series, or the function will 
# return unity.
def welch_coherence(x1, x2, Fs, nDT, nOverlap, window, two_sided = False, detrend = True):
    S1 = welch_PSD(x1,Fs,nDT,nOverlap,window,two_sided = False, detrend = True)
    S2 = welch_PSD(x2,Fs,nDT,nOverlap,window,two_sided = False, detrend = True) 
    S12 = welch_CSD(x1,x2,Fs,nDT,nOverlap,window,two_sided = False, detrend = True)
    return abs(S12**2)/(S1*S2)

# Uses welch's method to determine cross-spectral density between two different time series.        
def welch_CSD(x1, x2, Fs, nDT, nOverlap, window, scale_by_freq = True, two_sided = False,pad = True,detrend = True):
    N = len(x1) 
    # Remove mean
    if detrend:
        d1 = x1 - s.mean(x1)
        d2 = x2 - s.mean(x2)
    else:
        d1 = x1
        d2 = x2
        
    # Pad inputs to get an integer number of overlapping segments
    rem = (N-nDT)%(nDT-nOverlap)
    if nOverlap % 1 != 0:
        nOverlap = round(nOverlap)
    if rem == 0:
        pads = []
    else:
        pads = s.zeros((nDT-nOverlap)-rem)
    padded1 = s.hstack((d1,pads))
    padded2 = s.hstack((d2,pads))
    nSeg = (len(padded1)-nOverlap)/(nDT-nOverlap)

    # Calculate periodogram/direct estimate of each segment
    start = 0
    end = nDT
    CSD = []
    if pad:
        win = s.hstack((window,s.zeros(len(x1)-nDT)))        
        for i in range(int(nSeg)):
            sig1 = s.hstack((padded1[start:end],s.zeros(len(x1)-nDT)))
            sig2 = s.hstack((padded2[start:end],s.zeros(len(x1)-nDT)))
            CSD.append(cross_periodogram(sig1,sig2,Fs,win,scale_by_freq,two_sided,detrend = False))
            start += nDT-nOverlap
            end = start+nDT
        if two_sided:
            return s.mean(CSD,axis = 0)
        else:
            return s.mean(CSD,axis = 0)[:N/2+1]
    else:
        for i in range(int(nSeg)):
            CSD.append(cross_periodogram(padded1[start:end],padded2[start:end],Fs,window,scale_by_freq,two_sided))
            start += nDT-nOverlap
            end = start+nDT
        if two_sided:
            return s.mean(CSD,axis = 0)
        else:
            return s.mean(CSD,axis = 0)[:nDT/2+1]  

# Uses welch's overlapping segmentation averaging method to determine spectral density
# bandwidth = data window bandwidth (see data window bandwidth functions)
# d.o.f. = 2
# (100-2*p)% CI(dB) = 10*log10[log(p)/log(1-p)]         
def welch_PSD(x, Fs, nDT, nOverlap, window, scale_by_freq = True, two_sided = False,pad = True,detrend = True):
    N = len(x) 
    # Remove mean
    if detrend:
        d = x - s.mean(x)
    else:
        d = x
        
    # Pad input to get an integer number of overlapping segments
    rem = (N-nDT)%(nDT-nOverlap)
    if nOverlap % 1 != 0:
        nOverlap = round(nOverlap)
    if rem == 0:
        pads = []
    else:
        pads = s.zeros((nDT-nOverlap)-rem)
    padded = s.hstack((d,pads))
    nSeg = (len(padded)-nOverlap)/(nDT-nOverlap)

    # Calculate periodogrom/direct estimate of each segment
    start = 0
    end = nDT
    PSD = []
    if pad:
        win = s.hstack((window,s.zeros(len(x)-nDT)))        
        for i in range(int(nSeg)):
            sig = s.hstack((padded[start:end],s.zeros(len(x)-nDT)))
            PSD.append(periodogram(sig,Fs,win,scale_by_freq,two_sided,detrend = False))
            start += nDT-nOverlap
            end = start+nDT
        if two_sided:
            return s.mean(PSD,axis = 0)
        else:
            return s.mean(PSD,axis = 0)[:N/2+1]
    else:
        for i in range(int(nSeg)):
            PSD.append(periodogram(padded[start:end],Fs,window,scale_by_freq,two_sided))
            start += nDT-nOverlap
            end = start+nDT
        if two_sided:
            return s.mean(PSD,axis = 0)
        else:
            return s.mean(PSD,axis = 0)[:nDT/2+1]          

# Fits an autoregressive model to estimate the spectral density. Method can be
# Yule-Walker or Burg (typically performs better)
def AR_PSD(x,Fs,order,method = 'burg',pad = True, two_sided = False):
    m = s.mean(x)
    N = len(x)
    if order >= N:
        print('Order is too long')
        return 0
     
    # Burg method 
    if method == 'burg':
        d = x-m
        coefs = s.zeros(order)
        sig = 1/N*s.sum(d**2)
        
        # Levinson-Durbin recursion
        for k in range(1,order+1):
            ef = s.zeros(N)
            er = s.zeros(N)
            for t in range(k-1,N):
                ef[t] = d[t] - s.sum(coefs[0:k-1]*d[::-1][N-t:N-t+(k-1)])
            for t in range(k,N+1):
                er[t-k] = d[t-k] - s.sum(coefs[0:k-1]*d[t-(k-1):t])
            A = s.sum(ef[k::]**2+er[:N-k]**2)
            B = 2*s.sum(ef[k::]*er[:N-k])
            coefs[k-1] = B/A
            temp = s.copy(coefs)
            for j in range(1,k):
                coefs[j-1] = temp[j-1]-coefs[k-1]*temp[k-j-1]
            sig = sig *(1-coefs[k-1]**2)
            
    # Yule-Walker method
    elif method == 'yule-walker':
        # Estimate ACVS    
        sp = s.zeros(2*N-1)
        for k in range(N):
            sp[k+N-1] = 1/N*sum((x[k::]-m)*(x[0:N-k]-m))
        temp = sp[N::]
        sp[0:N-1] = temp[::-1]
        
        # Levinson-Durbin recursion
        coefs = s.zeros(order)
        coefs[0] = sp[N]/sp[N-1]
        sig = sp[N-1]*(1-coefs[0]**2)
        for k in range(2,order+1):
            coefs[k-1] = (sp[N-1+k]-sum(coefs[0:k-1]*sp[N-k:N-1]))/sig
            temp = s.copy(coefs)
            for j in range(1,k):
                coefs[j-1] = temp[j-1]-coefs[k-1]*temp[k-j-1]
            sig = sig *(1-coefs[k-1]**2)
    else:
        print('method not understood')
        return -1
        
    # Return Spectral Density Function
    if pad:
        coefs = s.hstack((0,coefs,s.zeros(N-order-1)))
        S = (sig/Fs)/(abs(1-s.fftpack.fft(coefs)))**2
        if two_sided:
            return S
        else:
            return S[0:N/2+1]
    else:
        coefs = s.hstack((0,coefs))
        S = (sig/Fs)/(abs(1-s.fftpack.fft(coefs)))**2
        if two_sided:
            return S
        else:
            return S[0:order/2+1]        

# Creates set of sine tapers for use in multitaper estimation
# bandwidth = (nTapers+1)/(N+1)*Fs
def sine_tapers(N,nTapers):    
    t = s.arange(0,N)
    h = s.zeros((nTapers,N))
    for k in range(nTapers):
        h[k,:] = (2/(N+1))*s.sin((k+1)*s.pi*(t+1)/(N+1))
    return h  
 
# Creates a set of slepian (discrete prolate spheroidal sequence) for use in
# multitaper estimation.  W = Fs*(2,3,4)/N 
# see slepian_bw for bandwidth  
def slepian_tapers(N,W,Fs):
    M = s.zeros((N,N))
    t1 = s.arange(0,N)
    for t2 in range(N):
        M[t2,:] = s.sin(2*s.pi*W*(t2-t1))/(s.pi*(t2-t1)) 
        M[t2,t2] = 2*W
    w,vr = s.linalg.eig(M)
    g = s.argsort(w)[::-1] 
    Kmax = 2*N*W/Fs-1
    return s.real(s.transpose(vr[:,g[:Kmax]]))

# Bandwidth for family of slepian tapers    
def slepian_bw(windows,Fs):
    l = s.size(windows,axis = 0)
    N = s.size(windows,axis = 1)
    avg = s.zeros(2*N-1)
    for i in range(l):
        w_norm = windows[i,:]/s.sqrt(s.sum(windows[i,:]**2))
        avg += s.signal.correlate(w_norm,w_norm)**2        
    return Fs/s.sum(avg/l)
    
# Estimates spectral density using multitapering
# bandwidth = see tapers
# d.o.f =  2/(sum(weights**2)) = 2*N if weights are equal
def multitaper_PSD(x,Fs,tapers,weights = 'equal',two_sided = False):
    nTapers = s.size(tapers,axis = 0)
    PSD = []
    if weights == 'equal':
        weights = s.ones(nTapers)/nTapers
    if len(weights) != nTapers:
        print('ERROR: Length of weight vector should be same as number of tapers')
        return 0
    else:
        for k in range(nTapers):
            PSD.append(weights[k]*periodogram(x,Fs,tapers[k,:],two_sided = two_sided))
    return s.sum(PSD,axis = 0)       


