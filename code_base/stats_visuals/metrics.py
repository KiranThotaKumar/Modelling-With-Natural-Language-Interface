import numpy as np


def rmse(y_true, y_pred):
	return np.sqrt(np.mean((y_true - y_pred) ** 2))

def correlation(y_true, y_pred):
	if np.std(y_true) == 0 or np.std(y_pred) == 0:
		return 0.0
	return np.corrcoef(y_true, y_pred)[0, 1]

def residuals(y_true, y_pred):
	return 0

def mae(y_true, y_pred):
	return np.mean(np.abs(y_true - y_pred))

def chi2(y_true, y_pred, errors, dof, nparams):	
    resid = y_true - y_pred
    chi2 = np.sum((resid / errors)**2)
    dof = len(y_true) - nparams
    reduced_chi2 = chi2 / dof
    print("reduced chi2:", reduced_chi2)
    return reduced_chi2

def simple_ess(self, samples_param):
    # very simple: use integrated autocorrelation estimate from numpy (better: emcee.autocorr)
    # Here we use a rough heuristic: ESS ≈ N / (1 + 2 * sum_rhos)
    import statsmodels.api as sm
    acf = sm.tsa.acf(samples_param, nlags=200, fft=True)
    # keep positive part
    rhos = acf[1:]
    pos = rhos[rhos > 0]
    ess = samples_param.size / (1 + 2*np.sum(pos))
    return ess
        
def ensure_shape_time_nqubits(arr):
    """
    Checks if array is (T, n_qubits). If it is (n_qubits, T), transposes it.
    Assumes T > 10 and 1 <= n_qubits <= 7.
    """
    # Ensure it's a numpy array for shape and transpose support
    arr = np.asanyarray(arr)
    
    # Get current shape (rows, cols)
    rows, cols = arr.shape
    
    # Logic: If the first dimension is small (n_qubits) and the second is large (T), 
    # then it is currently in (n_qubits, T) format and needs a transpose.
    if (1 <= rows <= 7) and (cols > 10):
        print(f"Transposing from ({rows}, {cols}) to ({cols}, {rows})")
        return arr.T
    
    # Otherwise, assume it is already in (T, n_qubits) or handles other cases
    return arr
