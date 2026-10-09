import numpy as np

def instrument_convolve(energies: np.ndarray, spec: np.ndarray, sigma_instr: float) -> np.ndarray:
    de = energies[1] - energies[0]
    nsig = sigma_instr / de
    size = len(energies)
    idx = np.arange(size)
    center = size // 2
    kernel = np.exp(-0.5 * ((idx - center) / nsig) ** 2)
    kernel /= np.sum(kernel)
    convolved = np.real(np.fft.ifft(np.fft.fft(spec) * np.fft.fft(np.fft.ifftshift(kernel))))
    return convolved
