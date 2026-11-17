# pipeline: range fft along sample axis : range bins -> FFT along chirp axis (velocity) = doppler ->
# AoA estimate: compare phase diff from the four antennas (beamforming, capon, music? )
# , CFAR -> translate range, azimuth/elevation + velocity to coordinates
#
# values straight from the .cfg to make accurate plots
# sampling rate in ksps, samples per chirp, slope of chirp, chirp period, start ghz,  
# number of chirps, number of rx used, number of tx used, number of frames in data

# if we have multiple tx, they take turns sending chirps
# chirp 0 tx0, etc. 

import os # for later...
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio_ffmpeg

C = 299_792_458.0  # m/s

# frame period: number of chirps * chirp period  
#chirp period: idle us + ramp_endus

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

    g = p.add_argument_group("plotting")
    g.add_argument("--frame", type=int, default=0, help="frame shown in the static plots")
    g.add_argument("--remove-static", action="store_true",
                   help="subtract per-range-bin mean across chirps for dopp")
    g.add_argument("--animate", action="store_true", help="play the frames as a video")
    g.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time, ")
    g.add_argument("--save", default=None, help="save the animation (.gif or .mp4) instead of showing it")
    g.add_argument("--no-static", action="store_true", help="skip the static plots")
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
        print("WARNING: ADC sampling time exceeds rampEndTime; check --samples / --fs-ksps")
    return d

#print out everything to crosscheck
def describe(a, d):
    print(f"bandwidth sampled  {d['bandwidth']/1e6:8.1f} MHz")
    print(f"range / bin        {d['range_per_bin']*100:8.2f} cm   (max {d['range_per_bin']*a.samples:.1f} m)")
    print(f"chirp period       {d['T_c']*1e6:8.1f} us   (same-TX spacing {d['T_slow']*1e6:.1f} us)")
    print(f"velocity / bin     {d['vel_per_bin']:8.3f} m/s  (max +/-{d['v_max']:.2f} m/s)")
    print(f"frame period       {d['T_f']*1e3:8.1f} ms")

# read file
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

# velocity STUFF!!!, also hamming window, fft along chirp axis. 
# the bigger the spaces between chrips / if we use frames, the lower our max velocity.
# however, a longer total integration time fives finer velocity resolution
def chirp_doppler(data_2, a, d):
    """
    doppler FFT along the chirp axis, per frame (range bin, Rx, chirp, frame) -> rd: (range bin, Doppler bin, frame),
    power summed over all virtual antennas (TX x RX)!! idk if thats correct. we lose phase from this ?
    2ith tx > 1 the chirp axis is first split into (loop, tx): chirp m = loop*tx + t.
    """
    nk, nrx, _, nf = data_2.shape
    x = data_2.reshape(nk, nrx, d["loops"], a.tx, nf)           # [k, r, loop, t, f]
    x = x.transpose(0, 1, 3, 2, 4).reshape(nk, nrx * a.tx, d["loops"], nf)  # [k, virt, loop, f]
    if a.remove_static:                                         # subtract eacg range bins mean across chrips. removes statioonary object noise + DC bs
        x = x - x.mean(axis=2, keepdims=True)
    wd = np.hamming(d["loops"]).astype(np.float32)               
    Y = np.fft.fftshift(np.fft.fft(x * wd[None, None, :, None], axis=2), axes=2)
    return np.sum(np.abs(Y) ** 2, axis=1)                       # noncoherent integration. we  lose phase through this so i have to change iy latet for AoA
# i also need to account for the  Tx>1 velocity dep phase for AoA


#helpers 
def db_norm(x, global_max=None):
    m = np.abs(x).max() if global_max is None else global_max
    return 20 * np.log10(np.abs(x) / m + 1e-12)

def pow_db(p, ref):
    return 10 * np.log10(p / ref + 1e-12)


# roberts RTI and frame doppler for comparison with chirp doppler
def static_plots(data_2, rd, rAx, vAx, a, d):
    f = a.frame
    t = (a.skip + f) * d["T_f"]

    # range profile: one frame, first Rx, first chirp
    prof = data_2[:, 0, 0, f]
    plt.figure()
    plt.plot(rAx, db_norm(prof))
    plt.xlabel("Range (m)")
    plt.ylabel("Normalized power (dB)")
    plt.title(f"Range profile (frame {a.skip + f}, t = {t:.3f} s, Rx 0, chirp 0)")

    # all range bins, receiver 0, chirp 0, all frames
    # RTI = range time intensity = 256-point ffts of all frames with a fixed chirp and receiver
    rti = data_2[:, 0, 0, :]                     
    plt.figure()
    plt.imshow(
        db_norm(rti),
        aspect="auto",
        origin="lower",                          
        extent=[0, rti.shape[1], rAx[0], rAx[-1]],
        vmin=-60, vmax=0,                         
    )
    plt.xlabel("Chirp index")
    plt.ylabel("Range (m)")
    cb = plt.colorbar()
    cb.set_label("Normalized power (dB)")
    plt.title("RTI")

    # frame doppler
    doppler_map = np.fft.fftshift(np.fft.fft(rti, n=rti.shape[1], axis=1), axes=1)
    plt.figure()
    plt.imshow(
        db_norm(doppler_map),
        aspect="auto",
        origin="lower",
        extent=[0, doppler_map.shape[1], rAx[0], rAx[-1]],
        vmin=-60, vmax=0,
    )

    plt.xlabel("Doppler bin")
    plt.ylabel("Range (m)")
    cb = plt.colorbar()
    cb.set_label("Normalized power (dB)")
    plt.title("Range-Doppler")

    # chirp doppler
    plt.figure()
    plt.imshow(
        pow_db(rd[:, :, f], rd[:, :, f].max()),
        aspect="auto",
        origin="lower",
        extent=[vAx[0], vAx[-1], rAx[0], rAx[-1]],
        vmin=-60, vmax=0,
    )

    plt.xlabel("Radial velocity (m/s)")
    plt.ylabel("Range (m)")
    cb = plt.colorbar()
    cb.set_label("Normalized power (dB)")
    plt.title(f"Range-Doppler, frame {a.skip + f}, t = {t:.3f} s, power summed over antennas")

# this is the fake real time graping where we stel through frames to show change over time
# for the raneg perfile, i average over chirps and rx
def animate(data_2, rd, rAx, vAx, a, d):
    nf = data_2.shape[3]
    prof = np.mean(np.abs(data_2) ** 2, axis=(1, 2))           # (range bin, frame)
    prof_ref = prof.max()                                       # global max -> comparable frames
    rd_ref = rd.max()

    fig, (ax_r, ax_rd) = plt.subplots(1, 2, figsize=(12, 5))
    line, = ax_r.plot(rAx, pow_db(prof[:, 0], prof_ref))
    ax_r.set_xlim(rAx[0], rAx[-1])
    ax_r.set_ylim(-60, 2)
    ax_r.set_xlabel("Range (m)")
    ax_r.set_ylabel("Power (dB, rel. capture max)")
    ax_r.set_title("Range profile (mean over chirps, Rx)")
    ax_r.grid(alpha=0.3)

    im = ax_rd.imshow(pow_db(rd[:, :, 0], rd_ref), aspect="auto", origin="lower",
                      extent=[vAx[0], vAx[-1], rAx[0], rAx[-1]], vmin=-60, vmax=0)
    ax_rd.set_xlabel("Radial velocity (m/s)")
    ax_rd.set_ylabel("Range (m)")
    ax_rd.set_title("Range-Doppler (chirp axis)")
    fig.colorbar(im, ax=ax_rd, label="Power (dB, rel. capture max)")
    head = fig.suptitle("")

    def update(f):
        frame_no = a.skip + f
        t = frame_no * d["T_f"]             
        line.set_ydata(pow_db(prof[:, f], prof_ref))
        im.set_data(pow_db(rd[:, :, f], rd_ref))
        head.set_text(f"frame {frame_no}   t = {t:.3f} s") # the time is not acc rn bc idk the frame periodicity
        return line, im, head

    update(0)
    interval_ms = d["T_f"] * 1e3 / a.speed
    ani = FuncAnimation(fig, update, frames=nf, interval=interval_ms, blit=False, repeat=True)
    if a.save:
        fps = max(1, round(1000 / interval_ms))
        writer = "pillow" if a.save.lower().endswith(".gif") else "ffmpeg"
        ani.save(a.save, writer=writer, fps=fps)
        print(f"saved {nf} frames -> {a.save}")
    return ani


def main():
    a = parse_args()
    d = derive(a)
    describe(a, d)

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    print("ffmpeg:", plt.rcParams["animation.ffmpeg_path"])

    w = np.fromfile("adc_data_oscillating_source_10_29.bin", dtype=np.int16, count=4_000_000).astype(float)
    for i in range(4):
        print(f"word {i}: mean {w[i::4].mean():8.1f}  std {w[i::4].std():8.1f}")
    z_even = w[0::4] + 1j*w[2::4]
    z_odd  = w[1::4] + 1j*w[3::4]
    print("corr between even/odd samples:", abs(np.vdot(z_even, z_odd)) / (np.linalg.norm(z_even)*np.linalg.norm(z_odd)))

    data = load_cube(a)
    print("data shape (Frame, ADC sample, Rx, chirp):", data.shape)

    data_2 = range_fft(data, a)
    print("data_2 shape (Range bin, Rx, chirp, Frame):", data_2.shape)

    rd = chirp_doppler(data_2, a, d)
    print("rd shape (Range bin, Doppler bin, Frame):", rd.shape)

    rAx = np.arange(a.samples) * d["range_per_bin"]
    vAx = (np.arange(d["loops"]) - d["loops"] // 2) * d["vel_per_bin"]

    if not a.no_static:
        if not 0 <= a.frame < a.frames:
            raise SystemExit(f"--frame must be in [0, {a.frames})")
        static_plots(data_2, rd, rAx, vAx, a, d)

    ani = animate(data_2, rd, rAx, vAx, a, d) if a.animate else None
    if not a.save:
        plt.show()


if __name__ == "__main__":
    main()