import numpy as np
import matplotlib.pyplot as plt

def plot_hist(hist, bins, err=None, ax=None, has_overflow=True, no_plot=False, **kwds):
    ax = plt.gca() if ax is None and not no_plot else ax
    if has_overflow:
        hist, err = _remove_overflow(hist, err)

    left, right = bins[:-1], bins[1:]
    midpoints = (left + right) / 2.0
    x = np.array([left, right]).T.flatten()
    y = np.array([hist, hist]).T.flatten()
    
    if not no_plot:
        line, = ax.plot(x, y, **kwds)
        if err is not None:
            if 'color' in kwds:
                del kwds['color']
            ax.errorbar(midpoints, hist, yerr=err, fmt='none',
                        color=line.get_color(), **kwds)
        ax.set_xlim(bins[0], bins[-1])
    return x, y, err

def _remove_overflow(hist, err):
    hist = hist[1:-1]
    if err is not None:
        if len(err) == len(hist) + 2: # +2 since we removed the overflow
            err = err[1:-1]
        elif len(err) == 2: # For asymmetric errors
            err = [err[0][1:-1], err[1][1:-1]]
        #TODO: What if len(hist) == 2 ?
        else:
            raise ValueError('err has wrong shape')
    return hist, err

##################
# Minuit helpers #
##################
def hist2args(hist):
    N = len(hist)
    hist_val = {'bin_{}'.format(i+1): hist[i] for i in range(N)}
    hist_err = {'error_bin_{}'.format(i+1): np.sqrt(np.abs(hist[i])) for i in range(N)}
    return dict(hist_val.items() + hist_err.items())

def vals2hist(minuit):
    N = minuit.narg
    vals = np.array([minuit.values[minuit.pos2var[i]]
                     for i in range(N)])
    errors = np.array([minuit.errors[minuit.pos2var[i]]
                       for i in range(N)])
    return vals, errors

def cov2array(minuit):
    N = minuit.narg
    cov = minuit.covariance
    return np.array([[cov[(minuit.pos2var[i], minuit.pos2var[j])] for j in range(N)]
                      for i in range(N)])