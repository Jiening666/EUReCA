"""
xim_fast.py — drop-in, vectorised replacement for ReadXim.read_xim.

The reference decoder in ReadXim.py reconstructs the compressed pixel buffer
with a pure-Python per-pixel loop (~10 s per Halcyon 384x3072 frame).  The XIM
compression stores, for every pixel after the first row, a difference against
the raster predictor

        pred[i] = P[i-1] + P[i-W] - P[i-W-1]          (left + top - topleft)

so that   P[i] = diff[i] + P[i-1] + P[i-W] - P[i-W-1].

Let g[i] = P[i] - P[i-1].  Substituting gives  g[i] = g[i-W] + diff[i], i.e. a
column-wise cumulative sum; then P = raster cumulative sum of g.  Both the
variable-width diff decode and the reconstruction therefore vectorise to a
handful of numpy ops, turning ~10 s/frame into a few ms/frame.

Produces the same dict/keys as ReadXim.read_xim for the fields downstream code
uses ('pixel_data', 'properties', dimensions).  Falls back to the reference
reader for the rare uncompressed case.
"""

import struct
import warnings

import numpy as np

import ReadXim

_WIDTH_BYTES = np.array([1, 2, 4, 4], dtype=np.int64)  # lut code -> byte width


def read_xim(filename, read_pixel_data=True):
    with open(filename, "rb") as f:
        img = {"file_name": filename}
        img["file_format_identifier"] = f.read(8).decode("ascii")
        (img["file_format_version"], img["image_width"], img["image_height"],
         img["bits_per_pixel"], img["bytes_per_pixel"],
         img["compression_indicator"]) = struct.unpack("<iiiiii", f.read(24))

        W, H = img["image_width"], img["image_height"]

        if img["compression_indicator"] != 1:
            # Uncompressed: defer to the reference implementation (cheap path).
            return ReadXim.read_xim(filename, read_pixel_data)

        # ---- compressed ----
        lut_size = struct.unpack("<i", f.read(4))[0]
        raw_bytes = np.fromfile(f, dtype=np.uint8, count=lut_size)
        vals = np.empty(lut_size * 4, dtype=np.uint8)
        vals[0::4] = raw_bytes & 0x03
        vals[1::4] = (raw_bytes >> 2) & 0x03
        vals[2::4] = (raw_bytes >> 4) & 0x03
        vals[3::4] = (raw_bytes >> 6) & 0x03

        cbuf_size = struct.unpack("<i", f.read(4))[0]

        if read_pixel_data and H > 0 and W > 0:
            cbuf = np.frombuffer(f.read(cbuf_size), dtype=np.uint8)
            _decode_into(img, cbuf, vals, W, H)
        else:
            f.seek(cbuf_size, 1)
            img["pixel_data"] = np.zeros((H, W),
                                         dtype=np.uint16 if img["bytes_per_pixel"] == 2
                                         else np.int32)

        img["uncompressed_pixel_buffer_size"] = struct.unpack("<i", f.read(4))[0]

        _read_histogram(f, img)
        _read_properties(f, img)
    return img


def _decode_into(img, cbuf, vals, W, H):
    n_pix = W * H
    n_raw = W + 1
    raw = cbuf[: n_raw * 4].view("<i4").astype(np.int32)

    n_rest = n_pix - n_raw
    codes = vals[:n_rest].astype(np.int64)
    wb = _WIDTH_BYTES[codes]
    # byte offset of each diff within the post-raw region
    off = np.zeros(n_rest, dtype=np.int64)
    if n_rest > 1:
        np.cumsum(wb[:-1], out=off[1:])

    buf = cbuf[n_raw * 4:]
    diffs = np.zeros(n_rest, dtype=np.int32)

    m1 = codes == 0
    if m1.any():
        o = off[m1]
        diffs[m1] = buf[o].view(np.int8).astype(np.int32)
    m2 = codes == 1
    if m2.any():
        o = off[m2]
        u = buf[o].astype(np.uint16) | (buf[o + 1].astype(np.uint16) << 8)
        diffs[m2] = u.view(np.int16).astype(np.int32)
    m4 = codes >= 2
    if m4.any():
        o = off[m4]
        u = (buf[o].astype(np.uint32)
             | (buf[o + 1].astype(np.uint32) << 8)
             | (buf[o + 2].astype(np.uint32) << 16)
             | (buf[o + 3].astype(np.uint32) << 24))
        diffs[m4] = u.view(np.int32)

    # innovations: raw first-row diffs, then decoded diffs
    innov = np.empty(n_pix, dtype=np.int32)
    innov[0] = raw[0]
    innov[1:n_raw] = raw[1:n_raw] - raw[: n_raw - 1]
    innov[n_raw:] = diffs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        g = np.cumsum(innov.reshape(H, W), axis=0, dtype=np.int32)  # g[i]=g[i-W]+innov
        # Column 0 has an extra raw pixel (pos=W): the g-recurrence for col 0
        # starts at row 1, so row 0's value (raw[0]) must not propagate down it.
        g[1:, 0] -= np.int32(raw[0])
        P = np.cumsum(g.reshape(-1), dtype=np.int32)                # P[i]=P[i-1]+g[i]

    # XIM 16-bit detector data is UNSIGNED. Casting to signed int16 wraps any
    # value > 32767 (e.g. HighGain bowtie flood peaks ~34289) to negative,
    # which downstream flat-fielding then guards to 1.0 -> spurious black band.
    if img["bytes_per_pixel"] == 2:
        P = P.astype(np.uint16)
    img["pixel_data"] = P.reshape(H, W)


# --- histogram + properties: identical layout to ReadXim, decoded inline ---

def _read_histogram(f, img):
    num_bins = struct.unpack("<i", f.read(4))[0]
    if num_bins > 0:
        img["histogram"] = {
            "number_of_bins": num_bins,
            "data": np.fromfile(f, dtype=np.int32, count=num_bins),
        }


def _read_properties(f, img):
    num_props = struct.unpack("<i", f.read(4))[0]
    props = {}
    for _ in range(max(num_props, 0)):
        try:
            name = f.read(struct.unpack("<i", f.read(4))[0]).decode("ascii")
            ptype = struct.unpack("<i", f.read(4))[0]
            if ptype == 0:
                v = struct.unpack("<i", f.read(4))[0]
            elif ptype == 1:
                v = struct.unpack("<d", f.read(8))[0]
            elif ptype == 2:
                v = f.read(struct.unpack("<i", f.read(4))[0]).decode("ascii")
            elif ptype == 4:
                n = struct.unpack("<i", f.read(4))[0] // 8
                v = np.fromfile(f, dtype=np.float64, count=n)
            elif ptype == 5:
                n = struct.unpack("<i", f.read(4))[0] // 4
                v = np.fromfile(f, dtype=np.int32, count=n)
            else:
                break
            props[name] = v
        except struct.error:
            break
    img["properties"] = {k: props[k] for k in sorted(props)}
