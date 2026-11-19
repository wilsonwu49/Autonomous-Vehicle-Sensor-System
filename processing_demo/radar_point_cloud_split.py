import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===== Settings =====
filename = "background_Raw_0.bin"
num_samples = 256                  # ADC samples per chirp
num_chirps = 32                    # chirps per frame
num_rx = 4                         # receive antennas
meters_per_bin = 0.0505            # range resolution
edge_bins = 5                      # range bins ignored at each end (TX->RX leakage)
remove_clutter = False             # True = subtract static background (only moving things remain)
pad = 32                           # angle FFT size (zero padding)
flipped_rx = [1, 3]                # channels that are 180 degrees out of phase on the board
rx_row = np.array([1, 1, 0, 0])    # where each RX sits in the 2x2 square
rx_col = np.array([1, 0, 1, 0])
first_frame = 0
fps = 20
profile_doppler = 0                # Doppler bin shown in the range profile panel (0 = static)

# CFAR settings, same meaning as the chip's cfarCfg (one per direction)
# mode: "CA" = average of both sides, "CASO" = smaller side, "CAGO" = bigger side
range_cfar = dict(mode="CASO", win=8, guard=4, threshold_db=15, cyclic=False)
doppler_cfar = dict(mode="CA", win=4, guard=2, threshold_db=15, cyclic=True)

bytes_per_frame = num_samples * num_chirps * num_rx * 4
num_frames = os.path.getsize(filename) // bytes_per_frame - first_frame
# print("frames in file:", num_frames)

# Angle FFT bin -> sin(angle). Same idea as range bin -> meters.
sin_axis = 2 * np.fft.fftshift(np.fft.fftfreq(pad))
ranges = np.arange(num_samples) * meters_per_bin


# File stores Q, I, Q, I... as int16. Output: complex (chirp, rx, sample).
def load_frame(n):
    raw = np.fromfile(filename, dtype=np.int16,
                      count=bytes_per_frame // 2,
                      offset=bytes_per_frame * (first_frame + n))
    samples = raw[1::2] + 1j * raw[0::2]
    return samples.reshape(num_chirps, num_rx, num_samples)


# =====================================================================
# PART 1: DETECT POINTS
# calibrate -> range FFT -> (clutter removal) -> Doppler FFT
#           -> detection matrix -> CFAR detect
# =====================================================================

# Undo the channels that are 180 degrees flipped on the board
def calibrate(frame):
    frame = frame.copy()
    frame[:, flipped_rx, :] *= -1
    return frame


# Sample axis -> range bins
def range_fft(frame):
    return np.fft.fft(frame * np.hamming(num_samples), axis=2)


# For each range bin and antenna, subtract the average over the chirps.
# Anything that doesn't change chirp to chirp (walls, leakage) cancels.
def clutter_removal(rfft):
    return rfft - rfft.mean(axis=0, keepdims=True)


# Chirp axis -> Doppler bins, with 0 (static) moved to the middle
def doppler_fft(rfft):
    windowed = rfft * np.hanning(num_chirps)[:, None, None]
    return np.fft.fftshift(np.fft.fft(windowed, axis=0), axes=0)


# Chip style: log magnitude of each channel, averaged over channels.
# Output: (doppler, range) in dB. Phase is gone here, so the angle step uses rd.
def detection_matrix(rd):
    return np.mean(20 * np.log10(np.abs(rd)), axis=1)


# Noise estimate for every cell along one line (a Doppler row or a range column).
# Average 'win' training cells on each side, skipping 'guard' cells next to the
# cell, then combine the two sides by mode. cyclic = wrap around the ends.
def cfar_noise(line, mode, win, guard, cyclic, **_):
    n = len(line)
    noise = np.full(n, np.inf)
    reach = win + guard
    padded = np.concatenate([line[-reach:], line, line[:reach]]) if cyclic else line
    shift = reach if cyclic else 0
    cells = range(n) if cyclic else range(reach, n - reach)
    for i in cells:
        j = i + shift
        left = padded[j - guard - win:j - guard].mean()
        right = padded[j + guard + 1:j + guard + 1 + win].mean()
        noise[i] = {"CA": (left + right) / 2, "CASO": min(left, right), "CAGO": max(left, right)}[mode]
    return noise


# A cell is a detection only if it passes both directions:
#   range:   above range noise + threshold, and bigger than its left/right range neighbors
#   Doppler: above Doppler noise + threshold, and bigger than its up/down Doppler neighbors
# Returns (doppler bin, range bin, snr), snr = dB above the local range noise.
def detect(power_db):
    range_noise = np.array([cfar_noise(power_db[d], **range_cfar) for d in range(num_chirps)])
    doppler_noise = np.array([cfar_noise(power_db[:, b], **doppler_cfar) for b in range(num_samples)]).T
    detections = []
    for d in range(num_chirps):
        for b in range(edge_bins, num_samples - edge_bins):
            cell = power_db[d, b]
            up, down = power_db[(d - 1) % num_chirps, b], power_db[(d + 1) % num_chirps, b]
            if cell < range_noise[d, b] + range_cfar["threshold_db"]:
                continue
            if cell < power_db[d, b - 1] or cell < power_db[d, b + 1]:
                continue
            if cell < doppler_noise[d, b] + doppler_cfar["threshold_db"]:
                continue
            if cell < up or cell < down:
                continue
            detections.append((d, b, cell - range_noise[d, b]))
    return detections


# =====================================================================
# PART 2: ANGLE FOR EACH POINT
# 4 antenna values -> 2x2 square -> 2D FFT -> brightest bin = direction
# =====================================================================

def angle_spectrum(rd, d, b):
    grid = np.zeros((2, 2), dtype=complex)
    grid[rx_row, rx_col] = rd[d, :, b]
    return np.abs(np.fft.fftshift(np.fft.fft2(grid, s=(pad, pad))))


def estimate_angle(rd, d, b):
    spectrum = angle_spectrum(rd, d, b)
    row, col = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    el = np.arcsin(sin_axis[row])
    az = np.arcsin(np.clip(sin_axis[col] / np.cos(el), -1, 1))
    return az, el


# =====================================================================
# PART 3: XYZ FOR EACH POINT
# range + angles -> x (sideways), y (forward), z (up)
# =====================================================================

def to_xyz(r, az, el):
    x = r * np.cos(el) * np.sin(az)
    y = r * np.cos(el) * np.cos(az)
    z = r * np.sin(el)
    return x, y, z


# Parts 1-3 chained: one frame in, (N, 5) points + detection matrix out.
# Columns: x, y, z, doppler bin, snr (dB)
def point_cloud(frame):
    rfft = range_fft(calibrate(frame))
    if remove_clutter:
        rfft = clutter_removal(rfft)
    rd = doppler_fft(rfft)
    power_db = detection_matrix(rd)

    points = []
    for d, b, snr in detect(power_db):
        az, el = estimate_angle(rd, d, b)
        x, y, z = to_xyz(b * meters_per_bin, az, el)
        points.append([x, y, z, d - num_chirps // 2, snr])
    # print(np.array(points).round(2))
    return np.array(points).reshape(-1, 5), power_db


# =====================================================================
# PART 4: OBJECT DETECTION (later)
# clustering points into objects, then tracking objects across frames
# =====================================================================


# =====================================================================
# PLOT: 5 panels
# top row: point cloud (top-down, side, 3D), colored by SNR
# bottom row: range-Doppler map with detections, range profile with CFAR threshold
# =====================================================================

fig = plt.figure(figsize=(16, 9))
layout = fig.add_gridspec(2, 6)
color = dict(cmap="Blues", vmin=0, vmax=50, edgecolors="k")
counter = fig.suptitle("", fontsize=14)

ax_top = fig.add_subplot(layout[0, 0:2])
top = ax_top.scatter([], [], c=[], **color)
ax_top.plot(0, 0, "k^", markersize=10)
ax_top.set_xlabel("Sideways x (m)")
ax_top.set_ylabel("Forward y (m)")
ax_top.set_title("Top-down")
ax_top.set_xlim(-8, 8)
ax_top.set_ylim(0, 13)
ax_top.set_aspect("equal")
fig.colorbar(top, ax=ax_top, label="SNR (dB)")

ax_side = fig.add_subplot(layout[0, 2:4])
side = ax_side.scatter([], [], c=[], **color)
ax_side.plot(0, 0, "k^", markersize=10)
ax_side.set_xlabel("Forward y (m)")
ax_side.set_ylabel("Up z (m)")
ax_side.set_title("Side view")
ax_side.set_xlim(0, 13)
ax_side.set_ylim(-5, 5)

ax_3d = fig.add_subplot(layout[0, 4:6], projection="3d")
ax_3d.set_xlabel("x (m)")
ax_3d.set_ylabel("y (m)")
ax_3d.set_zlabel("z (m)")
ax_3d.set_title("3D")
ax_3d.set_xlim(-8, 8)
ax_3d.set_ylim(0, 13)
ax_3d.set_zlim(-5, 5)

# Range-Doppler map: the detection matrix CFAR searched, red circles = what it kept
ax_rd = fig.add_subplot(layout[1, 0:3])
heat = ax_rd.imshow(np.zeros((num_samples, num_chirps)), origin="lower", aspect="auto", cmap="Blues",
                    extent=[-num_chirps / 2 - 0.5, num_chirps / 2 - 0.5, 0, ranges[-1]])
circles = ax_rd.scatter([], [], s=120, facecolors="none", edgecolors="red", linewidths=1.5)
ax_rd.set_xlabel("Doppler bin (0 = static)")
ax_rd.set_ylabel("Range (m)")
ax_rd.set_title("Range-Doppler map (red = detections)")
fig.colorbar(heat, ax=ax_rd, label="dB")

# Range profile: one Doppler row of the detection matrix and its CFAR threshold.
# A detection is where the signal pokes above the dashed line.
ax_prof = fig.add_subplot(layout[1, 3:6])
signal_line, = ax_prof.plot(ranges, np.zeros(num_samples), label="detection matrix")
threshold_line, = ax_prof.plot(ranges, np.zeros(num_samples), "--", label="range CFAR threshold")
hits = ax_prof.scatter([], [], marker="x", s=80, color="red", zorder=3, label="detections")
ax_prof.set_xlim(0, ranges[-1])
ax_prof.set_xlabel("Range (m)")
ax_prof.set_ylabel("dB")
ax_prof.set_title(f"Range profile, Doppler bin {profile_doppler}")
ax_prof.legend(loc="upper right")
ax_prof.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])


# =====================================================================
# VIDEO: run the full chain on frame n and redraw all 5 panels
# =====================================================================

def update(n):
    points, power_db = point_cloud(load_frame(n))
    # print(f"frame {n}: {len(points)} points")
    r = np.linalg.norm(points[:, :3], axis=1)

    top.set_offsets(points[:, [0, 1]])
    top.set_array(points[:, 4])
    side.set_offsets(points[:, [1, 2]])
    side.set_array(points[:, 4])

    for old in ax_3d.collections:
        old.remove()
    ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 4], **color)

    heat.set_data(power_db.T)
    heat.set_clim(power_db.max() - 50, power_db.max())
    circles.set_offsets(np.column_stack([points[:, 3], r]))

    row = power_db[profile_doppler + num_chirps // 2]
    threshold = cfar_noise(row, **range_cfar) + range_cfar["threshold_db"]
    signal_line.set_ydata(row)
    threshold_line.set_ydata(np.where(np.isinf(threshold), np.nan, threshold))
    on_row = points[:, 3] == profile_doppler
    bins = np.round(r[on_row] / meters_per_bin).astype(int)
    hits.set_offsets(np.column_stack([r[on_row], row[bins]]))
    ax_prof.set_ylim(row.min() - 5, row.max() + 5)

    counter.set_text(f"Frame {n + 1} / {num_frames}   ·   {len(points)} points")


anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps)
# anim.save("radar_point_cloud.mp4", fps=fps)   # needs ffmpeg
plt.show()
