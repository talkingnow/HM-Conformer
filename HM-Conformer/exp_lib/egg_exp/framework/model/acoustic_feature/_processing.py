import sys
import random
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import torchaudio as ta

def trimf(x, params):
    """
    trimf: similar to Matlab definition
    https://www.mathworks.com/help/fuzzy/trimf.html?s_tid=srchtitle
    
    """
    assert len(params) == 3, 'requires params to be a list of 3 elements' 

    a = params[0]
    b = params[1]
    c = params[2]

    assert a <= b and b <= c, 'trimp(x, [a, b, c]) requires a<=b<=c' 

    y = torch.zeros_like(x, dtype=torch.float32)

    if a < b:
        index = torch.logical_and(a < x, x < b)
        y[index] = (x[index] - a) / (b - a)

    if b < c:    
        index = torch.logical_and(b < x, x < c)              
        y[index] = (c - x[index]) / (c - b)

    y[x == b] = 1

    return y 
    
def delta(x):
    length = x.shape[1]
    output = torch.zeros_like(x)
    x_temp = torch_nn_func.pad(x.unsqueeze(1), (0, 0, 1, 1), 'replicate').squeeze(1)
    output = -1 * x_temp[:, 0:length] + x_temp[:,2:]
    return output

def rfft_wrapper(x, onesided=True, inverse=False):
    # compatiblity with torch fft API 
    if hasattr(torch, "rfft"):
        # for torch < 1.8.0, rfft is the API to use
        # torch 1.7.0 complains about this API, but it is OK to use
        if not inverse:
            # FFT
            return torch.rfft(x, 1, onesided=onesided)
        else:
            # inverse FFT
            return torch.irfft(x, 1, onesided=onesided)
    else:
        # for torch > 1.8.0, fft.rfft is the API to use
        if not inverse:
            # FFT
            if onesided:
                data = torch.fft.rfft(x)
            else:
                data = torch.fft.fft(x)
            return torch.stack([data.real, data.imag], dim=-1)
        else:
            # It requires complex-tensor
            real_image = torch.chunk(x, 2, dim=1)
            x = torch.complex(real_image[0].squeeze(-1), 
                              real_image[1].squeeze(-1))
            if onesided:
                return torch.fft.irfft(x)
            else:
                return torch.fft.ifft(x)
            
def dct1(x):
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return rfft_wrapper(
        torch.cat([x, x.flip([1])[:, 1:-1]], dim=1))[:, :, 0].view(*x_shape)


def idct1(X):
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = rfft_wrapper(v, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi/(2*N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, 
                     device=X.device)[None, :]*np.pi/(2*N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = rfft_wrapper(V, onesided=False, inverse=True)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

class FrequencyMasking:
    """Tensor based"""
    def __init__(self, p, max):
        self.p = p
        
        # masking range [0, max]
        self.frq_masking = ta.transforms.FrequencyMasking(freq_mask_param=max)

    def __call__(self, spec):
        if self.p < random.random():
            return spec
        
        spec = self.frq_masking(spec)
        
        return spec

class LinearDCT(torch_nn.Linear):
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!

class Melspec(object):
    def __init__(self, sf=16000, fl=400, fs=80, fftl=1024, mfbsize=80, 
                 melmin=0, melmax=None, ver=1):
        self.ver = ver
        # sampling rate
        self.sf = sf
        # frame length 
        self.fl = fl
        # frame shift
        self.fs = fs
        # fft length
        self.fftl = fftl
        # mfbsize
        self.mfbsize = mfbsize
        # mel.min frequency (in Hz)
        self.melmin = melmin
        # mel.max frequency (in Hz)
        if melmax is None:
            self.melmax = sf/2
        else:
            self.melmax = melmax
        
        # windows
        self.window = np.square(np.blackman(self.fl).astype(np.float32))
        winpower = np.sqrt(np.sum(self.window))

        if self.ver == 2:
            self.window = np.blackman(self.fl).astype(np.float32) / winpower
        else:
            self.window = self.window / winpower
        
        # create mel-filter bank
        self.melfb = self._melfbank(self.melmin, self.melmax)
        
        # eps = 1.0E-12
        self.eps = 1.0E-12
        return
    
    def _freq2mel(self, freq):
        return 1127.01048 * np.log(freq / 700.0 + 1.0)

    def _mel2freq(self, mel):
        return (np.exp(mel / 1127.01048) - 1.0) * 700.0
        
    def _melfbank(self, melmin, melmax):
        linear_freq = 1000.0
        mfbsize = self.mfbsize - 1

        bFreq = np.linspace(0, self.sf / 2.0, self.fftl//2 + 1, 
                            dtype=np.float32)
        
        minMel = self._freq2mel(melmin)
        maxMel = self._freq2mel(melmax)
        
        iFreq = self._mel2freq(np.linspace(minMel, maxMel, mfbsize + 2, 
                                           dtype=np.float32))
        
        linear_dim = np.where(iFreq<linear_freq)[0].size
        iFreq[:linear_dim+1] = np.linspace(iFreq[0], iFreq[linear_dim], 
                                           linear_dim+1)

        diff = np.diff(iFreq)
        so = np.subtract.outer(iFreq, bFreq)
        lower = -so[:mfbsize] / np.expand_dims(diff[:mfbsize], 1)
        upper = so[2:] / np.expand_dims(diff[1:], 1)
        fb = np.maximum(0, np.minimum(lower, upper))

        enorm = 2.0 / (iFreq[2:mfbsize+2] - iFreq[:mfbsize])
        fb *= enorm[:, np.newaxis]

        fb0 = np.hstack([np.array(2.0*(self.fftl//2)/self.sf, np.float32), 
                         np.zeros(self.fftl//2, np.float32)])
        fb = np.vstack([fb0, fb])

        return fb
    
class LFCC(torch_nn.Module):
    def __init__(self, fl, fs, fn, sr, filter_num, 
                 with_energy=False, with_emphasis=True,
                 with_delta=True, flag_for_LFB=False,
                 num_coef=None, min_freq=0, max_freq=1):
        super(LFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num
        self.num_coef = num_coef

        # decide the range of frequency bins
        if min_freq >= 0 and min_freq < max_freq and max_freq <= 1:
            self.min_freq_bin = int(min_freq * (fn//2+1))
            self.max_freq_bin = int(max_freq * (fn//2+1))
            self.num_fft_bins = self.max_freq_bin - self.min_freq_bin 
        else:
            print("LFCC cannot work with min_freq {:f} and max_freq {:}".format(
                min_freq, max_freq))
            sys.exit(1)
        
        # build the triangle filter bank
        f = (sr / 2) * torch.linspace(min_freq, max_freq, self.num_fft_bins)
        filter_bands = torch.linspace(min(f), max(f), filter_num+2)
        
        filter_bank = torch.zeros([self.num_fft_bins, filter_num])
        for idx in range(filter_num):
            filter_bank[:, idx] = trimf(
                f, [filter_bands[idx], 
                    filter_bands[idx+1], 
                    filter_bands[idx+2]])
        self.lfcc_fb = torch_nn.Parameter(filter_bank, requires_grad=False)

        # DCT as a linear transformation layer
        self.l_dct = LinearDCT(filter_num, 'dct', norm='ortho')

        # opts
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.flag_for_LFB = flag_for_LFB
        if self.num_coef is None:
            self.num_coef = filter_num

        # Add a buf to store window coefficients
        #  
        self.window_buf = None
        return
    
    def forward(self, x):
        """
        Param
            x: 2D Tensor(batchm, length) 
        
        Return
            lfcc_output: 3D Tensor(batch, frame, dim)
        """
        # pre-emphsis 
        if self.with_emphasis:
            # to avoid side effect
            x_copy = torch.zeros_like(x) + x
            x_copy[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
        else:
            x_copy = x
        
        if self.window_buf is None:
            self.window_buf = torch.hamming_window(self.fl).to(x.device)

        # STFT
        #x_stft = torch.stft(x_copy, self.fn, self.fs, self.fl, 
        #                    window=torch.hamming_window(self.fl).to(x.device), 
        #                    onesided=True, pad_mode="constant")
        x_stft = torch.stft(x_copy, self.fn, self.fs, self.fl, window=self.window_buf, onesided=True, pad_mode='constant', return_complex=False)
        
        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()
        
        if self.min_freq_bin > 0 or self.max_freq_bin < (self.fn//2+1):
            sp_amp = sp_amp[:, :, self.min_freq_bin:self.max_freq_bin]
        
        # filter bank
        fb_feature = torch.log10(torch.matmul(sp_amp, self.lfcc_fb) + 
                                 torch.finfo(torch.float32).eps)
        
        # DCT (if necessary, remove DCT)
        lfcc = self.l_dct(fb_feature) if not self.flag_for_LFB else fb_feature
        
        # Truncate the output of l_dct when necessary
        if not self.flag_for_LFB and self.num_coef != self.filter_num:
            lfcc = lfcc[:, :, :self.num_coef]
            

        # Add energy 
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(power_spec.sum(axis=2)+ 
                                 torch.finfo(torch.float32).eps)
            lfcc[:, :, 0] = energy

        # Add delta coefficients
        if self.with_delta:
            lfcc_delta = delta(lfcc)
            lfcc_delta_delta = delta(lfcc_delta)
            lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
        else:
            lfcc_output = lfcc

        # done
        return lfcc_output

class LFB(LFCC):
    def __init__(self, fl, fs, fn, sr, filter_num, 
                 with_energy=False, with_emphasis=True,
                 with_delta=False):
        super(LFB, self).__init__(fl, fs, fn, sr, filter_num, with_energy,
                                  with_emphasis, with_delta, flag_for_LFB=True)
        return
    
    def forward(self, x):
        """
        Param
            x: 2D Tensor(batchm, length) 
        
        Return
            lfb_output: 3D Tensor(batch, frame, dim)
        """
        return super(LFB, self).forward(x)

class Spectrogram(torch_nn.Module):
    """ Spectrogram front-end
    """
    def __init__(self, fl, fs, fn, sr, 
                 with_emphasis=True, with_delta=False, in_db=False):
        super(Spectrogram, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr

        # opts
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.in_db = in_db

        # buf to store window coefficients
        self.window_buf = None
        return
    
    def forward(self, x):
        """
        Param
            x: 2D Tensor(batchm, length) 
        
        Return
            lfcc_output: 3D Tensor(batch, frame, dim)
        """
        # pre-emphsis 
        if self.with_emphasis:
            x[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
        
        if self.window_buf is None:
            self.window_buf = torch.hamming_window(self.fl).to(x.device)

        # STFT      
        x_stft = torch.stft(x, self.fn, self.fs, self.fl, window=self.window_buf, onesided=True, pad_mode='constant', return_complex=False)
        

        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()
        
        if self.in_db:
            sp_amp = torch.log10(sp_amp + torch.finfo(torch.float32).eps)

        # Add delta coefficients
        if self.with_delta:
            sp_delta = delta(sp_amp)
            sp_delta_delta = delta(sp_delta)
            sp_output = torch.cat((sp_amp, sp_delta, sp_delta_delta), 2)
        else:
            sp_output = sp_amp

        # done
        return sp_amp

class MFCC(torch_nn.Module):
    """ Based on asvspoof.org baseline Matlab code.
    Difference: with_energy is added to set the first dimension as energy
    """
    def __init__(self, fl, fs, fn, sr, filter_num, 
                 with_energy=False, with_emphasis=True,
                 with_delta=True, flag_for_MelSpec=False,
                 num_coef=None, min_freq=0, max_freq=1):
        super(MFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num
        self.num_coef = num_coef

        # decide the range of frequency bins
        if min_freq >= 0 and min_freq < max_freq and max_freq <= 1:
            pass
        else:
            print("MFCC cannot work with min_freq {:f} and max_freq {:}".format(
                min_freq, max_freq))
            sys.exit(1)
        
        # opts
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.flag_for_MelSpec = flag_for_MelSpec
        if self.num_coef is None:
            self.num_coef = filter_num
            
        # get filter bank
        tmp_config = Melspec(sr, fl, fs, fn, filter_num, 
                                       sr/2*min_freq, sr/2*max_freq)
        filter_bank = torch.tensor(tmp_config.melfb.T, dtype=torch.float32)
        self.mel_fb = torch_nn.Parameter(filter_bank, requires_grad=False)

        # DCT as a linear transformation layer
        if not self.flag_for_MelSpec:
            self.l_dct = LinearDCT(filter_num, 'dct', norm='ortho')
        else:
            self.l_dct = None


        # Add a buf to store window coefficients
        #  
        self.window_buf = None
        return
    
    def forward(self, x):
        """
        Param
            x: 2D Tensor (batch, length)
            
        Return
            lfcc_output: 3D Tensor(batch, frame, dim)
        output:
        """
        # pre-emphsis 
        if self.with_emphasis:
            # to avoid side effect
            x_copy = torch.zeros_like(x) + x
            x_copy[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
        else:
            x_copy = x
        
        if self.window_buf is None:
            self.window_buf = torch.hamming_window(self.fl).to(x.device)

        # STFT
        x_stft = torch.stft(x_copy, self.fn, self.fs, self.fl, window=self.window_buf, onesided=True, pad_mode='constant', return_complex=False)

        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()
        
        # filter bank
        fb_feature = torch.log10(torch.matmul(sp_amp, self.mel_fb) + 
                                 torch.finfo(torch.float32).eps)
        
        # DCT (if necessary, remove DCT)
        if not self.flag_for_MelSpec:
            output = self.l_dct(fb_feature) 
        else:
            output = fb_feature
        
        # Truncate the output of l_dct when necessary
        if not self.flag_for_MelSpec and self.num_coef != self.filter_num:
            output = output[:, :, :self.num_coef]

        # Add energy 
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(power_spec.sum(axis=2)+ 
                                 torch.finfo(torch.float32).eps)
            output[:, :, 0] = energy

        # Add delta coefficients
        if self.with_delta:
            output_delta = delta(output)
            output_delta_delta = delta(output_delta)
            output = torch.cat((output, output_delta, output_delta_delta), 2)
        else:
            pass

        # done
        return output
