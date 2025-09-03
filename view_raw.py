import av
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from av import logging as avlog
import io
import cv2

def unpack_yuyv(frame, H, W, swap = False):
    assert frame.format.name == "yuyv422", f"Got {frame.format.name}, not yuyv422"
    yuy2 = frame.to_ndarray(format="yuyv422")      # shape (H, W, 2) or similar
    b = yuy2.reshape(-1)                            # flat byte view

    # Macropixels: [Y0, U, Y1, V] → reconstruct two 16-bit pixels
    y0 = b[0::4].astype(np.uint16)
    u  = b[1::4].astype(np.uint16)
    y1 = b[2::4].astype(np.uint16)
    v  = b[3::4].astype(np.uint16)

    p0 = (y0 << 8) | u
    p1 = (y1 << 8) | v

    mono16 = np.empty(W * H, dtype=np.uint16)
    mono16[0::2] = p0
    mono16[1::2] = p1
    mono16 = mono16.reshape(H, W)

    # Byte-swap (MSB/LSB reverse) as you requested
    if swap is True:
        mono16_swapped = mono16.byteswap()
        return mono16_swapped
    else:
        return mono16

device_name = "FX3"
W, H, FPS = 640, 480, 30

container = av.open(
    format="dshow",
    file=f"video={device_name}",
    options={"video_size": f"{W}x{H}", "framerate": str(FPS), "pixel_format": "yuyv422"}
)

decoder = container.decode(video=0)

# prepare Matplotlib window
plt.ion()
fig, ax = plt.subplots(figsize=(12,8))
im = ax.imshow([[0]], cmap='gray', vmin=0, vmax=2**16-1)
ax.axis("off")

try:
    for frame in decoder:
        # YUY2 → (H, W) uint16
        mono16_swapped = unpack_yuyv(frame, H, W, swap=False)
        
        # update the displayed image
        im.set_data(mono16_swapped)
        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    pass
finally:
    container.close()
    plt.ioff()
