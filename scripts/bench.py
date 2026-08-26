import time
import cupy as cp

src = r'''
extern "C" __global__
void mandelbrot_f(
    unsigned char* out,
    int width, int height,
    float min_re, float min_im,
    float scale_re, float scale_im,
    int max_iter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float cr = min_re + (float)x * scale_re;
    float ci = min_im + (float)y * scale_im;
    float zr = 0.0f, zi = 0.0f;
    int n = 0;
    while (n < max_iter && (zr*zr + zi*zi) <= 4.0f) {
        float tmp = zr*zr - zi*zi + cr;
        zi = 2.0f * zr * zi + ci;
        zr = tmp;
        n++;
    }
    float t = (float)n;
    if (n < max_iter) {
        float logz = 0.5f * logf(zr*zr + zi*zi);
        t = t - log2f(logz / logf(2.0f));
    }
    unsigned char v = (unsigned char)(255.0f * t / (float)max_iter);
    int idx = 3 * (y * width + x);
    out[idx] = v; out[idx+1] = v; out[idx+2] = v;
}
'''

kernel = cp.RawKernel(src, 'mandelbrot_f')

for W, H, label in [(1920, 1080, "1080p"), (3840, 2160, "4K")]:
    out = cp.zeros((H, W, 3), dtype=cp.uint8)
    grid = ((W + 15) // 16, (H + 15) // 16)
    block = (16, 16)
    args = (out, W, H, -2.5, -1.0, 3.5 / W, 2.0 / H, 512)
    for i in range(4):
        kernel(grid, block, args)
    cp.cuda.Stream.null.synchronize()
    times = []
    for i in range(10):
        t0 = time.perf_counter()
        kernel(grid, block, args)
        cp.cuda.Stream.null.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"{label} fp32: avg {sum(times)/len(times):.2f} ms, min {min(times):.2f} ms")
