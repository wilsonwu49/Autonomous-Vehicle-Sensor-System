import os # for later...
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio_ffmpeg

C = 299_792_458.0  # m/s

# IWR6843AOP virtual array
# positions are in units of lambda/2 = 2.5mm is a unit. checl antGeometry0/1 and antPhaseRot in demo config
# positions 0 -11 for 12 virtual array
# virtual antenna 0 = TX0 RX0= X[0] (1 unit left) Y[0] (1 unit down)
# virtual antenna 1 = TX0 RX1 = X[1] (1 unit left) Y[1] (0 unit down) etc.
X = np.array([-1, -1, 0, 0, -3, -3, -2, -2, -1, -1, 0, 0])   # azimuth index
Y = np.array([-1, 0, 0, -1, -3, -2, -2, -3, -3, -2, -2, -3])  # elevation index
AOP_ROT = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])  # per-channel 180 deg flips

# eventually i would just want to parse the cfg file
def parse_args():
    p = argparse.ArgumentParser(description="Range + Dopp plots")
    p.add_argument("file", nargs="?", default="adc_data.bin")
    g = p.add_argument_group("capture config")
    g.add_argument("--fs-ksps", type=float, default=10000, help="digOutSampleRate, ksps")
    g.add_argument("--samples", type=int, default=256, help="numAdcSamples per chirp")
    g.add_argument("--slope", type=float, default=60, help="freqSlopeConst, MHz/us")
    g.add_argument("--idle-us", type=float, default=100, help="idleTime, us")
    g.add_argument("--ramp-end-us", type=float, default=60, help="rampEndTime, us")
    g.add_argument("--start-ghz", type=float, default=60.0, help="startFreq, GHz")
    g.add_argument("--chirps", type=int, default=122,
                   help="chirps per frame")
    g.add_argument("--rx", type=int, default=1, help="number of RX ")
    g.add_argument("--tx", type=int, default=1, help="number of TX ")
    g.add_argument("--frame-period-ms", type=float, default=40, help="framePeriodicity, ms")
    
    g = p.add_argument_group("file layout")
    g.add_argument("--iq-order", choices=["qi", "iq", "lanes", "lane1"], default="lane1",
               help="qi: Q0 I0 Q1 I1 (Robert's), iq: I0 Q0 I1 Q1, lanes: I0 I1 Q0 Q1, "
                    "lane1: only LVDS lane 1 has data, every other word is 0")
    g.add_argument("--rx-layout", choices=["noninterleaved", "interleaved"], default="noninterleaved",
                   help="noninterleaved: per chirp, all samples of RX0 then RX1. etc "
                        "interleaved: per chirp, sample0 of RX0..RX3, sample1 of RX0."
                        ".RX3 etc")
    g.add_argument("--frames", type=int, default=493, help="frames to read")
    g.add_argument("--skip", type=int, default=0, help="frames to skip at the start of the file")
    return p.parse_args()

def derive(a):
    d = {}
    d["slope_hz_s"] = a.slope * 1e12               # MHz/us -> Hz/s
    d["fs_hz"] = a.fs_ksps * 1e3                   # ksps -> Hz
    d["T_c"] = (a.idle_us + a.ramp_end_us) * 1e-6  # one chirp period, s
    d["T_f"] = a.frame_period_ms * 1e-3            # frame period, s
    d["lam"] = C / (a.start_ghz * 1e9)             # wavelength, m

    if a.chirps % a.tx:
        raise SystemExit(f"--chirps ({a.chirps}) must be divisible by --tx ({a.tx})")
    d["loops"] = a.chirps // a.tx                  # num of chirps each tx sends per frame. doppler fft 
    d["T_slow"] = d["T_c"] * a.tx                  # time between chirps of the same TX

    d["range_per_bin"] = C * d["fs_hz"] / (2 * d["slope_hz_s"] * a.samples) # fft bins r fs/n apart
    d["vel_per_bin"] = d["lam"] / (2 * d["loops"] * d["T_slow"])
    d["v_max"] = d["lam"] / (4 * d["T_slow"])
    d["bandwidth"] = d["slope_hz_s"] * a.samples / d["fs_hz"]

    active = a.chirps * d["T_c"]
    if active > d["T_f"]:
        print(f"WARNING: chirps take {active*1e3:.2f} ms but frame period is "
              f"{a.frame_period_ms} ms; check --frame-period-ms")
    if a.samples / d["fs_hz"] > a.ramp_end_us * 1e-6:
        print("WARNING: ADC sampling time exceeds rampEndTime. check --samples / --fs-ksps")
    return d

#read file
def load_cube(a):
    """Return complex cube shaped (frame, ADC sample, Rx, chirp)."""
    frametot = a.samples * a.chirps * a.rx          # complex samples per frame
    bytes_per_complex = 4    

    # i wazs trying to check what i was reading
    # raw = np.fromfile("adc_data.bin", dtype=np.int16)
    # print("words:", raw.size, " min/max:", raw.min(), raw.max(), " std:", raw.std())
    # print("expected words per frame:", 2 * 256 * 128 * 4)
    # print("frames in file:", raw.size / (2 * 256 * 128 * 4))
    # plt.plot(raw[:2048]); plt.title("first 2048 raw words"); plt.show()  
                        
    pad = 2 if a.iq_order == "lane1" else 1          # file has 2x the words when lane 2 is empty
    raw = np.fromfile(
        a.file,
        dtype=np.uint16,
        count=2 * frametot * a.frames * pad,
        offset=a.skip * bytes_per_complex * frametot * pad,
    ).astype(np.float32)

    if raw.size != 2 * frametot * a.frames * pad:
        raise RuntimeError(f"File shorter than expected: got {raw.size // (2 * frametot * pad)} "
                           f"full frames, wanted {a.frames}")

    raw[raw >= 2**15] -= 2**16

    if a.iq_order == "lane1": #lol 
        p = raw[0::2].reshape(-1, 2)             
        iq = p[:, 0] + 1j * p[:, 1]                  
    elif a.iq_order == "qi":                          
        p = raw.reshape(-1, 2)
        iq = p[:, 1] + 1j * p[:, 0]
    elif a.iq_order == "iq":                       
        p = raw.reshape(-1, 2)
        iq = p[:, 0] + 1j * p[:, 1]
    else:                                          
        p = raw.reshape(-1, 4)
        iq = np.empty(p.shape[0] * 2, dtype=np.complex64)
        iq[0::2] = p[:, 0] + 1j * p[:, 2]
        iq[1::2] = p[:, 1] + 1j * p[:, 3]
    iq = iq.astype(np.complex64)

    #  data cube = (frame, ADC, Rx, chirp) 
    # c-order reshape lists axes slowest-first 
    if a.rx_layout == "noninterleaved":            
        cube = iq.reshape(a.frames, a.chirps, a.rx, a.samples).transpose(0, 3, 2, 1)
    else:                                           
        cube = iq.reshape(a.frames, a.chirps, a.samples, a.rx).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(cube)

# RANGE STUFF!! hamming window, fft along sample axis! 
def range_fft(data, a):
    """(frame, ADC, Rx, chirp) ->( range bin, Rx, chirp, frame)"""
    wt = np.hamming(a.samples).astype(np.float32)   
    data_2 = np.fft.fft(data * wt[None, :, None, None], n=a.samples, axis=1)
    return np.moveaxis(data_2, 0, -1)

def to_virtual(data_2, ntx, rot=AOP_ROT, cal=None):
    """
    (range, rx, chirp, frame) -> (range, virt, loop, frame), with virt = tx*nrx + rx.
    # virt is vritual antwnna, and loop is how many times each tx fired per frame(chirps in a fram /number of tx used)
    this assumes chirp m = loop*ntx + t and that the tx order = t0 t1 t2 
    applies the antenna phase rotation, plus an optional complex calibration vector (len = nvirt). 
    """
    nk, nrx, nch, nf = data_2.shape
    loops = nch // ntx
    x = data_2.reshape(nk, nrx, loops, ntx, nf)          # [k, r, loop, t, f]
    x = x.transpose(0, 3, 1, 2, 4).reshape(nk, ntx * nrx, loops, nf)  # [k, t*nrx+r, loop, f]
    # the phase application according to AOP_ROT for each virtual antenna
    w = rot.astype(np.complex64) if rot is not None else np.ones(ntx * nrx, np.complex64)
    if cal is not None:
        w = w * cal
    return x * w[None, :, None, None]
#TO DO:
def covariance_per_range():
    pass

def steering_vector():
    pass

# velocity STUFF!!!, also hamming window, fft along chirp axis. 
# the bigger the spaces between chrips / if we use frames, the lower our max velocity.
# however, a longer total integration time fives finer velocity resolution
def chirp_doppler(data_2, a, d):
    """
    doppler FFT along the chirp axis, per frame (range bin, Rx, chirp, frame) 
    """
    nk, nrx, _, nf = data_2.shape
    x = data_2.reshape(nk, nrx, d["loops"], a.tx, nf)           # [k, r, loop, t, f]
    x = x.transpose(0, 1, 3, 2, 4).reshape(nk, nrx * a.tx, d["loops"], nf)  # [k, virt, loop, f]       
    x = x - x.mean(axis=2, keepdims=True) # removes static # subtract eacg range bins mean across chrips. removes statioonary object noise + DC bs
    wd = np.hamming(d["loops"]).astype(np.float32)               
    Y = np.fft.fftshift(np.fft.fft(x * wd[None, None, :, None], axis=2), axes=2)
    return Y
    # i also need to account for the  Tx>1 velocity dep phase for AoA

#helpers 
def db_norm(x, global_max=None):
    m = np.abs(x).max() if global_max is None else global_max
    return 20 * np.log10(np.abs(x) / m + 1e-12)

def pow_db(p, ref):
    return 10 * np.log10(p / ref + 1e-12)


