import numpy as np
from scipy.stats import norm, cauchy,laplace, stats
from scipy.optimize import minimize, root_scalar
from statsmodels.robust.scale import mad
import warnings


def beta_cauchy(x):
    """
    Find the function beta for the mixed normal prior with Cauchy
    tails.  It is assumed that the noise variance is equal to one.
    
    Parameters:
    x - a real value or vector
    """
    
    phix = norm.pdf(x)
    j = (x != 0)
    beta = x
    beta = np.where(j == False, -1/2, beta)
    beta[j] = (norm.pdf(0) / phix[j] - 1) / (x[j] ** 2) - 1
    
    return beta



def beta_laplace(x, s=1, a=0.5):
    """
    The function beta for the Laplace prior given parameter a and s (sd)
    
    x - the value or vector of data values
    s - the value or vector of standard deviations; if vector, must have the same length as x
    a - the scale parameter of the Laplace distribution
    """
    x = np.abs(x)
    xpa = x/s + s*a
    xma = x/s - s*a
    rat1 = 1/xpa
    xpa < 35
    rat1[xpa < 35]
    if isinstance(rat1, (int, float, str, bool)) and xpa < 35:
        rat1 = norm.cdf(-xpa) / norm.pdf(xpa)
    elif isinstance(rat1, (int, float, str, bool)) and xpa > 35:
        rat1 = rat1
    else:
        rat1[xpa < 35] = norm.cdf(-xpa[xpa < 35]) / norm.pdf(xpa[xpa < 35])
    rat2 = 1/np.abs(xma)
    if isinstance(xma, (int, float, str, bool)) and xma > 35:
        xma = 35
    elif isinstance(xma, (int, float, str, bool)) and xma < 35:
        xma = xma
    else:
        xma[xma > 35] = 35
    if isinstance(rat1, (int, float, str, bool)) and xma > -35:
        rat2 = norm.cdf(xma) / norm.pdf(xma)
    elif isinstance(rat1, (int, float, str, bool)) and xma < -35:
        rat2 = rat2
    else:
        rat2[xma > -35] = norm.cdf(xma[xma > -35]) / norm.pdf(xma[xma > -35])
    beta = (a * s) / 2 * (rat1 + rat2) - 1
    
    return beta

def ebayesthresh(x, prior="laplace", a=0.5, bayesfac=False, sdev=None, verbose=False, threshrule="median", universalthresh=True, stabadjustment=None):
    pr = prior[0:1]

    if sdev is None:
        sdev = mad(x, center=0)
        stabadjustment_condition = True
    elif len(np.atleast_1d(sdev)) == 1:
        if stabadjustment is not None:
            raise ValueError("Argument stabadjustment is not applicable when variances are homogeneous.")
        if np.isnan(sdev):
            sdev = mad(x, center=0)
        stabadjustment_condition = True
    else:
        if pr == "c":
            raise ValueError("Standard deviation has to be homogeneous for Cauchy prior.")
        if len(sdev) != len(x):
            raise ValueError("Standard deviation has to be homogeneous or have the same length as observations.")
        if stabadjustment is None:
            stabadjustment = False
        stabadjustment_condition = stabadjustment

    if stabadjustment_condition:
        m_sdev = np.mean(sdev)
        s = sdev / m_sdev
        x = x / m_sdev
    else:
        s = sdev

    if (pr == "l") and np.isnan(a):
        pp = wandafromx(x, s, universalthresh)
        w = pp['w']
        a = pp['a']
    else:
        w = wfromx(x, s, prior=prior, a=a, universalthresh=universalthresh)

    if pr != "m" or verbose:
        tt = tfromw(w, s, prior=prior, bayesfac=bayesfac, a=a)[0]
        if stabadjustment_condition:
            tcor = tt * m_sdev
        else:
            tcor = tt

    if threshrule == "median":
        muhat = postmed(x, s, w, prior=prior, a=a)
    elif threshrule == "mean":
        muhat = postmean(x, s, w, prior=prior, a=a)
    elif threshrule == "hard":
        muhat = threshld(x, tt)
    elif threshrule == "soft":
        muhat = threshld(x, tt, hard=False)
    elif threshrule == "none":
        muhat = None
    else:
        raise ValueError(f"Unknown threshold rule: {threshrule}")

    if stabadjustment_condition:
        muhat = muhat * m_sdev

    if not verbose:
        return muhat
    else:
        retlist = {
            'muhat': muhat,
            'x': x,
            'threshold.sdevscale': tt,
            'threshold.origscale': tcor,
            'prior': prior,
            'w': w,
            'a': a,
            'bayesfac': bayesfac,
            'sdev': sdev,
            'threshrule': threshrule
        }
        if pr == "c":
            del retlist['a']
        if threshrule == "none":
            del retlist['muhat']
        return retlist

    
    
def ebayesthresh_wavelet_dwt(x_dwt, vscale="independent", smooth_levels=float('inf'), 
                             prior="laplace", a=0.5, bayesfac=False, 
                             threshrule="median"):
    nlevs = len(x_dwt) - 1
    slevs = min(nlevs, smooth_levels)
    
    if isinstance(vscale, str):
        vs = vscale[0].lower()
        if vs == "i":
            vscale = mad(x_dwt[0], center=0)
        if vs == "l":
            vscale = None
    
    for j in range(slevs):
        x_dwt[j] = ebayesthresh(x_dwt[j], prior=prior, a=a, bayesfac=bayesfac, 
                                sdev=vscale, verbose=False, threshrule=threshrule)
    
    return x_dwt

def ebayesthresh_wavelet_splus(x_dwt, vscale="independent", smooth_levels=float('inf'), 
                                prior="laplace", a=0.5, bayesfac=False, threshrule="median"):
    nlevs = len(x_dwt)
    slevs = min(nlevs, smooth_levels)
    
    if isinstance(vscale, str):
        vs = vscale[0].lower()
        if vs == "i":
            vscale = mad(x_dwt[-1])  # Use the last level for vscale
        elif vs == "l":
            vscale = None
    
    for j in range(nlevs - slevs + 1, nlevs + 1):
        x_dwt[j - 1] = ebayesthresh(x_dwt[j - 1], prior=prior, a=a, bayesfac=bayesfac, 
                                    sdev=vscale, verbose=False, threshrule=threshrule)
    
    return x_dwt

def ebayesthresh_wavelet_wd(x_wd, vscale="independent", smooth_levels=float('inf'), 
                             prior="laplace", a=0.5, bayesfac=False, threshrule="median"):
    nlevs = x_wd.nlevels
    slevs = min(nlevs - 1, smooth_levels)
    
    if isinstance(vscale, str):
        vs = vscale[0].lower()
        if vs == "i":
            vscale = mad(x_wd[-1].d)  # Use the last level for vscale
        elif vs == "l":
            vscale = None
    
    for j in range(nlevs - slevs, nlevs - 1):
        x_wd.d[j] = ebayesthresh(x_wd.d[j], prior=prior, a=a, bayesfac=bayesfac, 
                                  sdev=vscale, verbose=False, threshrule=threshrule)
    
    return x_wd

def cauchy_medzero(x, z, w):
    hh = z - x
    dnhh = norm.pdf(hh)
    yleft = norm.cdf(hh) - z * dnhh + ((z * x - 1) * dnhh * norm.cdf(-x)) / norm.pdf(x)
    yright2 = 1 + np.exp(-z**2 / 2) * (z**2 * (1 / w - 1) - 1)
    return yright2 / 2 - yleft


def cauchy_threshzero(z, w):
    if isinstance(z, (list)):
        z = np.array(z)
    y = norm.cdf(z) - z * norm.pdf(z) - 1/2 - (z**2 * np.exp(-z**2/2) * (1/w - 1))/2
    return y


def isotone(x, wt=None, increasing=False):
    """
    Find the weighted least squares isotone fit to the sequence x,
    the weights given by the sequence wt. If increasing == True,
    the curve is set to be increasing, otherwise to be decreasing.
    The vector ip contains the indices on the original scale of the
    breaks in the regression at each stage.

    Parameters:
        x (list or numpy.ndarray): Input sequence.
        wt (list or numpy.ndarray, optional): Weights for the sequence x. Defaults to None.
        increasing (bool, optional): If True, the curve is set to be increasing, otherwise decreasing.
            Defaults to False.

    Returns:
        list: Isotonic fit to the input sequence x.
    """
    nn = len(x)

    if nn == 1:
        x = x

    if not increasing:
        x = -x

    ip = np.arange(1, nn+1)
    dx = np.diff(x)
    nx = len(x)

    while nx > 1 and np.min(dx) < 0:
        # Find all local minima and maxima
        jmax = np.arange(nx)[(np.concatenate((dx <= 0, [False])) & np.concatenate(([True], dx > 0)))]
        jmin = np.arange(nx)[(np.concatenate((dx > 0, [True])) & np.concatenate(([False], dx <= 0)))]

        for jb in range(len(jmax)):
            ind = np.arange(jmax[jb], jmin[jb]+1)
            wtn = np.sum(wt[ind])
            x[jmax[jb]] = np.sum(wt[ind] * x[ind]) / wtn
            wt[jmax[jb]] = wtn
            x[jmax[jb]+1:jmin[jb]+1] = np.nan

        # Clean up within iteration, eliminating the parts of sequences that
        # were set to NA
        ind = ~np.isnan(x)
        x = x[ind]
        wt = wt[ind]
        ip = ip[ind]
        dx = np.diff(x)
        nx = len(x)

    # Final cleanup: reconstruct z at all points by repeating the pooled
    # values the appropriate number of times
    jj = np.zeros(nn, dtype=int)
    jj[ip - 1] = 1
    z = x[np.cumsum(jj) - 1]

    if not increasing:
        z = -z

    return z.tolist()


def laplace_threshzero(x, s=1, w=0.5, a=0.5):
    """
    This function needs to be zeroed to find the threshold using Laplace prior.

    Parameters:
    x (float): Input value
    s (float): Standard deviation (default: 1)
    w (float): Mean (default: 0.5)
    a (float): Input value where a < 20 (default: 0.5)
    """
    a = min(a, 20)
    
    if isinstance(x, list):
        z = []
        for elem in x:
            xma = elem / s - s * a
            z_add = norm.cdf(xma) - (1 / a) * (1 / s * norm.pdf(xma)) * (1 / w + beta_laplace(elem, s, a))
            z.append(z_add)
    
    else:
        xma = x / s - s * a
        z = norm.cdf(xma) - (1 / a) * (1 / s * norm.pdf(xma)) * (1 / w + beta_laplace(x, s, a))
    
    return z

def negloglik_laplace(xpar, xx, ss, tlo, thi):
    """
    Marginal negative log likelihood function for Laplace prior. 
    Constraints for thresholds need to be passed externally.

    Parameters:
    xpar : array-like, shape (2,)
        Vector of two parameters:
        xpar[0]: a value between [0, 1] which will be adjusted to range of w 
        xpar[1]: inverse scale (rate) parameter ("a")
    xx : array-like
        Data
    ss : array-like
        Vector of standard deviations
    tlo : array-like
        Lower bound of thresholds
    thi : array-like
        Upper bound of thresholds

    Returns:
    neg_loglik : float
        Negative log likelihood
    """
    a = xpar[1]
    
    # Calculate the range of w given a, using negative monotonicity
    # between w and t
    wlo = wfromt(thi, ss, a=a)
    whi = wfromt(tlo, ss, a=a)
    wlo = np.max(wlo)
    whi = np.min(whi)
    
    loglik = np.sum(np.log(1 + (xpar[0] * (whi - wlo) + wlo) *
                           beta_laplace(xx, ss, a)))
    
    return -loglik

def postmean(x, s=1, w=0.5, prior="laplace", a=0.5):
    """
    Find the posterior mean for the appropriate prior for given x, s (sd), w, and a.
    """
    pr = prior[0:1]
    if pr == "l":
        mutilde = postmean_laplace(x, s, w, a=a)
    elif pr == "c":
        if np.any(s != 1):
            raise ValueError("Only standard deviation of 1 is allowed for Cauchy prior.")
        mutilde = postmean_cauchy(x, w)
    else:
        raise ValueError("Unknown prior type.")
    return mutilde

def postmean_cauchy(x, w):
    """
    Find the posterior mean for the quasi-Cauchy prior with mixing
    weight w given data x, which may be a scalar or a vector.
    """
    muhat = x  # Ensure x is a numpy array
    ind = (x == 0)
    x = x[~ind]  # Remove zeros from x
    ex = np.exp(-x**2/2)
    z = w * (x - (2 * (1 - ex))/x)
    z = z / (w * (1 - ex) + (1 - w) * ex * x**2)
    muhat[~ind] = z
    return muhat


def postmean_laplace(x, s=1, w=0.5, a=0.5):
    """
    Find the posterior mean for the double exponential prior for
    given x, s (sd), w, and a.

    Args:
        x (float or numpy array): Input data.
        s (float): Standard deviation.
        w (float): Parameter.
        a (float): Parameter.

    Returns:
        float or numpy array: Posterior mean.
    """
    # Only allow a < 20 for input value.
    a = min(a, 20)
    
    # First find the probability of being non-zero
    w_post = wpost_laplace(w, x, s, a)
    
    # Now find the posterior mean conditional on being non-zero
    sx = np.sign(x)
    x = np.abs(x)
    xpa = x/s + s*a
    xma = x/s - s*a
    xpa[xpa > 35] = 35
    xma[xma < -35] = -35
    
    cp1 = norm.cdf(xma)
    cp2 = norm.cdf(-xpa)
    ef = np.exp(np.minimum(2 * a * x, 100))
    postmean_cond = x - a * s**2 * (2 * cp1 / (cp1 + ef * cp2) - 1)
    
    # Calculate posterior mean and return
    return sx * w_post * postmean_cond

def postmed(x, s=1, w=0.5, prior="laplace", a=0.5):
    """
    Find the posterior median for the appropriate prior for given x, s (sd), w, and a.

    Parameters:
        x (array-like): Observations.
        s (float): Standard deviation.
        w (float): Weight parameter.
        prior (str): Type of prior ("laplace" or "cauchy").
        a (float): Parameter a.

    Returns:
        array-like: Posterior median estimates.
    """
    pr = prior[0:1]
    if pr == "l":
        muhat = postmed_laplace(x, s, w, a)
    elif pr == "c":
        if np.any(s != 1):
            raise ValueError("Only standard deviation of 1 is allowed for Cauchy prior.")
        muhat = postmed_cauchy(x, w)
    else:
        raise ValueError(f"Unknown prior: {prior}")
    return muhat

def postmed_cauchy(x, w):
    """
    Find the posterior median of the Cauchy prior with mixing weight w,
    pointwise for each of the data points x
    """
    nx = len(x)
    zest = np.full(nx, np.nan)
    w = np.full(nx, w)
    ax = np.abs(x)
    j = (ax < 20)
    zest[~j] = ax[~j] - 2 / ax[~j]
    
    if np.sum(j) > 0:
        zest[j] = vecbinsolv(zf=np.zeros(np.sum(j)), fun=cauchy_medzero,
                             tlo=0, thi=np.max(ax[j]), z=ax[j], w=w[j])
                             
    zest[zest < 1e-7] = 0
    zest = np.sign(x) * zest
    
    return zest

def postmed_laplace(x, s=1, w=0.5, a=0.5):
    """
    Find the posterior median for the Laplace prior for given x (observations), s (sd), w, and a.
    
    Parameters:
        x (array-like): Observations.
        s (float): Standard deviation.
        w (float): Weight parameter.
        a (float): Parameter a.
        
    Returns:
        array-like: Posterior median estimates.
    """
    # Only allow a < 20 for input value
    a = min(a, 20)
    
    # Work with the absolute value of x, and for x > 25 use the approximation
    # to dnorm(x-a)*beta_laplace(x, a)
    sx = np.sign(x)
    x = np.abs(x)
    xma = x / s - s * a
    zz = 1 / a * (1 / s * norm.pdf(xma)) * (1 / w + beta_laplace(x, s, a))
    zz[xma > 25] = 0.5
    mucor = norm.ppf(np.minimum(zz, 1))
    muhat = sx * np.maximum(0, xma - mucor) * s
    
    return muhat

def threshld(x, t, hard=True):
    """
    Threshold the data x using threshold t.
    If hard=True, use hard thresholding.
    If hard=False, use soft thresholding.
    """
    if hard:
        z = x * (np.abs(x) >= t)
    else:
        z = np.sign(x) * np.maximum(0, np.abs(x) - t)
    return z



def wandafromx(x, s=1, universalthresh=True):
    """
    Find the marginal max lik estimators of w and a given standard  deviation s,
    using a bivariate optimization; If universalthresh=TRUE, the thresholds will 
    be upper bounded by universal threshold adjusted by standard deviation. 
    The threshold is constrained to lie between 0 and sqrt ( 2 log (n)) *   s. 
    Otherwise, threshold can take any nonnegative value;  If running R, 
    the routine optim is used; in S-PLUS the routine is nlminb.
    """
    # Range for thresholds
    if universalthresh:
        thi = np.sqrt(2 * np.log(len(x))) * s
    else:
        thi = np.inf

    if isinstance(s, int):
        tlo = np.zeros(len(str(s)))
    else:
        tlo = np.zeros(len(s))
    lo = np.array([0, 0.04])
    hi = np.array([1, 3])
    startpar = np.array([0.5, 0.5])

    if 'optim' in globals():
        result = minimize(negloglik_laplace, startpar, method='L-BFGS-B', bounds=[(lo[0], hi[0]), (lo[1], hi[1])], args=(x, s, tlo, thi))
        uu = result.x
    else:
        result = minimize(negloglik_laplace, startpar, bounds=[(lo[0], hi[0]), (lo[1], hi[1])], args=(x, s, tlo, thi))
        uu = result.x

    a = uu[1]
    wlo = wfromt(thi, s, a=a)
    whi = wfromt(tlo, s, a=a)
    wlo = np.max(wlo)
    whi = np.min(whi)
    w = uu[0] * (whi - wlo) + wlo
    return {'w': w, 'a': a}

def mad(x, center=None, constant=1.4826, na_rm=False, low=False, high=False):
    # NA 제거 옵션 처리
    if na_rm:
        x = x[~np.isnan(x)]

    # center 값 설정
    if center is None:
        center = np.median(x)

    # low 또는 high median 설정
    if low:
        center = np.percentile(x, 25)
    elif high:
        center = np.percentile(x, 75)

    # Median Absolute Deviation 계산
    mad_value = np.median(np.abs(x - center)) * constant
    return mad_value



def wfromt(tt, s=1, prior="laplace", a=0.5):
    """
    Find the weight that has posterior median threshold tt, 
    given s (sd) and a.
    
    tt - Threshold value or vector of values.
    s - A single value or a vector of standard deviations if the Laplace prior is used. If
    a vector, must have the same length as tt. Ignored if Cauchy prior is used.
    prior - Specification of prior to be used; can be "cauchy" or "laplace".
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
    """
    pr = prior[0:1]
    if pr == "l":
        tma = tt / s - s * a
        wi = 1 / np.abs(tma)
        if isinstance(wi, (int, float, str, bool)) and tma > -35:
            wi = norm.cdf(tma) / norm.pdf(tma)
        elif isinstance(wi, (int, float, str, bool)) and tma < -35:
            wi = wi
        else:
            wi[tma > -35] = norm.cdf(tma[tma > -35])/norm.pdf(tma[tma > -35])
        wi = a * s * wi - beta_laplace(tt, s, a)
        
    if pr == "c":
        dnz = norm.pdf(tt)
        wi = 1 + (norm.cdf(tt) - tt * dnz - 1/2) / (np.sqrt(np.pi/2) * dnz * tt**2)
        if isinstance(wi, np.ndarray):
            for i in range(len(wi)):
                if not np.isfinite(wi[i]):
                    wi[i] = 1
        else:
            if not np.isfinite(wi):
                wi = 1
    return 1 / wi


def wfromx(x, s=1, prior="laplace", a=0.5, universalthresh=True):
    """
    Given the vector of data x and s (sd),
    find the value of w that zeroes S(w) in the
    range by successive bisection, carrying out nits harmonic bisections
    of the original interval between wlo and 1.  
  
    x - Vector of data.
    s - A single value or a vector of standard deviations if the Laplace prior is used. If
    a vector, must have the same length as x. Ignored if Cauchy prior is used.
    prior - Specification of prior to be used; can be "cauchy" or "laplace".
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
    universalthresh - If universalthresh = TRUE, the thresholds will be upper bounded by universal
    threshold; otherwise, the thresholds can take any non-negative values.
    """
    pr = prior[0:1]

    if pr == "c":
        s = 1

    if universalthresh:
        tuniv = np.sqrt(2 * np.log(len(x))) * s
        wlo = wfromt(tuniv, s, prior, a)
        wlo = np.max(wlo)
    else:
        wlo = 0

    if pr == "l":
        beta = beta_laplace(x, s, a)
    elif pr == "c":
        beta = beta_cauchy(x)

    whi = 1
    beta = np.minimum(beta, 1e20)

    shi = np.sum(beta / (1 + beta))
    if shi >= 0:
        shi =  1

    slo = np.sum(beta / (1 + wlo * beta))
    if slo <= 0:
        slo = wlo

    for _ in range(1,31):
        wmid = np.sqrt(wlo * whi)
        smid = np.sum(beta / (1 + wmid * beta))
        if smid == 0:
            smid = wmid
        if smid > 0:
            wlo = wmid
        else:
            whi = wmid
            
    return np.sqrt(wlo * whi)



def wmonfromx(xd, prior="laplace", a=0.5,  tol=1e-08, maxits=20):
    """
    Find the monotone marginal maximum likelihood estimate of the
    mixing weights for the Laplace prior with parameter a.  It is
    assumed that the noise variance is equal to one.
    Find the beta values and the minimum weight
    Current version allows for standard deviation of 1 only.
    xd - A vector of data.
    prior - Specification of the prior to be used; can be cauchy or laplace.
    a - Scale parameter in prior if prior="laplace". Ignored if prior="cauchy".
    tol - Absolute tolerance to within which estimates are calculated.
    maxits - Maximum number of weighted least squares iterations within the calculation.
    """
    pr = prior[0:1]
    nx = len(xd)
    wmin = wfromt(np.sqrt(2 * np.log(len(xd))), prior=prior, a=a)
    winit = 1
    if pr == "l":
        beta = beta_laplace(xd, a=a)
    if pr == "c":
        beta = beta_cauchy(xd)
    """
    now conduct iterated weighted least squares isotone regression
    """
    w = np.repeat(winit, len(beta))
    for j in range(maxits):
        aa = w + 1 / beta
        ps = w + aa
        ww = 1 / aa ** 2
        wnew = isotone(ps, ww, increasing=False)
        wnew = np.maximum(wmin, wnew)
        wnew = np.minimum(1, wnew)
        zinc = np.max(np.abs(np.diff(wnew)))
        w = wnew
        if zinc < tol:
            return w

    # warnings.filterwarnings("More iterations required to achieve convergence")
    return w


def vecbinsolv(zf, fun, tlo, thi, nits=30, **kwargs):
    """
    Given a monotone function fun, and a vector of values zf find a vector of numbers t such that f(t) = zf.
    The solution is constrained to lie on the interval (tlo, thi).
    
    The function fun may be a vector of increasing functions.
    
    Present version is inefficient because separate calculations are done for each element of z, 
    and because bisections are done even if the solution is outside the range supplied.
    
    It is important that fun should work for vector arguments. 
    Additional arguments to fun can be passed through *args and **kwargs.
    
    Works by successive bisection, carrying out nits harmonic bisections of the interval between tlo and thi.
    
    zf- the right hand side of the equation(s) to be solved

    fun - an increasing function of a scalar argument, or a vector of such functions

    tlo - lower limit of interval over which the solution is sought

    thi-upper limit of interval over which the solution is sought

    nits-number of binary subdivisions carried out
    """
    s = kwargs.get('s', 0) 
    w = kwargs.get('w', 0) 
    a = kwargs.get('a', 0) 
    if isinstance(zf, int):
        nz = len(str(zf))
    else:
        nz = len(zf)
    
    if isinstance(tlo, (int, float, np.int64)):
        tlo = np.array([tlo] * nz)
    if not isinstance(tlo, (int, float, np.int64)) and len(tlo) != nz:
        raise ValueError("Lower constraint has to be homogeneous or has the same length as #functions.")
    if isinstance(thi, (int, float, np.int64)):
        thi = np.array([thi] * nz)
    if not isinstance(thi, (int, float, np.int64)) and len(thi) != nz:
        raise ValueError("Upper constraint has to be homogeneous or has the same length as #functions.")

    # carry out nits bisections
    for _ in range(nits):
        tmid = np.array([(lo + hi) / 2 for lo, hi in zip(tlo, thi)])
        if fun == cauchy_threshzero:
            fmid = fun(z=tmid, w=w)
        elif fun == laplace_threshzero:
            fmid = fun(x=tmid, s=s, w=w, a=a)
        elif fun == beta_cauchy:
            fmid = fun(tmid)
        elif fun == beta_laplace:
            fmid = fun(tmid, s=s, a=a)
        else:
            fmid = fun(tmid, **kwargs)
        if isinstance(fmid, (list,np.ndarray)) and isinstance(zf, (list,np.ndarray)):
            indt = [f <= z for f, z in zip(fmid, zf)]
        else: 
            indt = fmid <= zf
        tlo = [tm if ind else lo for tm, lo, ind in zip(tmid, tlo, indt)]
        thi = [tm if not ind else hi for tm, hi, ind in zip(tmid, thi, indt)]
        
    tsol = [(lo + hi) / 2 for lo, hi in zip(tlo, thi)]
    
    return tsol


def tfromw(w, s=1, prior="laplace", bayesfac=False, a=0.5):
    """
    
    This function finds the threshold or threshold vector corresponding to the given weight vector w and standard deviation s under the specified prior distribution.
    If bayesfac is set to True, it finds the Bayes factor thresholds; otherwise, it finds the posterior median thresholds.
    If the Laplace prior distribution is used, a specifies the value of the inverse scale (i.e., rate) parameter where a < 20.

    Parameters:
    w (array-like): Weight vector
    s (float): Standard deviation (default: 1)
    prior (str): Prior distribution (default: "laplace")
    bayesfac (bool): Whether to find the Bayes factor thresholds (default: False)
    a (float): Input value where a < 20 (default: 0.5)

    Returns:
    array-like: Threshold or threshold vector
    """
    pr = prior[0:1]
    if bayesfac:
        z = 1 / w - 2
        if pr == "l":
            if isinstance(s, (int, float, str, bool)) and len(w) >= len(str(s)):
                zz = z
            elif isinstance(s, (int, float, str, bool)) and len(w) <len(str(s)):
                zz = [z] * len(str(s))
            elif len(w) >= len(s):
                zz = z
            elif len(w) < len(str(s)):
                zz = [z] * len(s)
            tt = vecbinsolv(zz, beta_laplace, 0, 10, 30, s=s, w=w, a=a)
        elif pr == "c":
            tt = vecbinsolv(z, beta_cauchy, 0, 10, 30, w=w)
    else:
        z = 0
        if pr == "l":
            if isinstance(s, (int, float, str, bool)) and not isinstance(w, (int, float, str, bool)):
                zz = np.array([0] * max(len(str(s)), len(w)))
            elif not isinstance(s, (int, float, str, bool)) and isinstance(w, (int, float, str, bool)):
                zz = np.array([0] * max(len(s), len(str(w))))
            elif isinstance(s, (int, float, str, bool)) and isinstance(w, (int, float, str, bool)):
                zz = np.array([0] * max(len(str(s)), len(str(w))))
            else:
                zz = [0] * max(len(s), len(w))
            tt = vecbinsolv(zz, laplace_threshzero, 0, s * (25 + s * a), 30, s=s, w=w, a=a)
        elif pr == "c":
            tt = vecbinsolv(z, cauchy_threshzero, 0, 10, 30, w=w)
    return tt


def tfromx(x, s=1, prior="laplace", bayesfac=False, a=0.5, universalthresh=True):
    """
    Given the data x, the prior, and any other parameters, find the
    threshold corresponding to the marginal maximum likelihood
    estimator of the mixing weight.
    
    x - Vector of data.
    s - A single value or a vector of standard deviations if the Laplace prior is used. If
    a vector, must have the same length as x. Ignored if Cauchy prior is used.
    prior - Specification of prior to be used; can be "cauchy" or "laplace".
    bayesfac - Specifies whether Bayes factor threshold should be used instead of posterior
    median threshold.
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
    universalthresh - If universalthresh = TRUE, the thresholds will be upper bounded by universal
    threshold; otherwise, the thresholds can take any non-negative values.
    """
    pr = prior[0:1]
    if pr == "c":
        s = 1
    if pr == "l" and np.isnan(a):
        wa = wandafromx(x, s, universalthresh)
        w = wa['w']
        a = wa['a']
    else:
        w = wfromx(x, s, prior=prior, a=a)
    return tfromw(w, s, prior=prior, bayesfac=bayesfac, a=a)[0]

def wpost_laplace(w, x, s=1, a=0.5):
    # Calculate the posterior weight for non-zero effect
    laplace_beta = beta_laplace(x, s, a)
    return 1 - (1 - w) / (1 + w * laplace_beta)

def zetafromx(xd, cs, pilo=None, prior="laplace", a=0.5):
    """
    Given a sequence xd, a vector of scale factors cs, and a lower limit pilo,
    find the marginal maximum likelihood estimate of the parameter zeta
    such that the prior probability is of the form median(pilo, zeta*cs, 1).

    If pilo is None, then it is calculated according to the sample size
    to correspond to the universal threshold.

    Find the beta values and the minimum weight if necessary.

    Current version allows for standard deviation of 1 only.
    
    xd - A vector of data.
    cs - A vector of scale factors, of the same length as xd.
    pilo The lower limit for the estimated weights. If pilo=NA it is calculated according
    to the sample size to be the weight corresponding to the universal threshold √ 2 log n.
    prior Specification of prior to be used conditional on the mean being nonzero; can be
    cauchy or laplace.
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used. If, on entry,
    a=NA and prior="laplace", then the scale parameter will also be estimated by
    marginal maximum likelihood. If a is not specified then the default value 0.5
    will be used.
    """
    pr = prior[0:1]
    nx = len(xd)
    if pilo is None:
        pilo = wfromt(np.sqrt(2 * np.log(len(xd))), prior=prior, a=a)
    if pr == "l":
        beta = beta_laplace(xd, a=a)
    elif pr == "c":
        beta = beta_cauchy(xd)
    
    # Find jump points zj in the derivative of log likelihood as a function of z,
    # and other preliminary calculations
    zs1 = pilo / cs
    zs2 = 1 / cs
    zj = np.sort(np.unique(np.concatenate((zs1, zs2))))
    cb = cs * beta
    mz = len(zj)
    zlmax = None

    # Find left and right derivatives at each zj and check which are local minima
    lmin = np.zeros(mz, dtype=bool)
    for j in range(1, mz - 1):
        ze = zj[j]
        cbil = cb[(ze > zs1) & (ze <= zs2)]
        ld = np.sum(cbil / (1 + ze * cbil))
        if ld <= 0:
            cbir = cb[(ze >= zs1) & (ze < zs2)]
            rd = np.sum(cbir / (1 + ze * cbir))
            lmin[j] = rd >= 0
    
    # Deal with the two end points in turn, finding right deriv at lower end
    # and left deriv at upper.
    cbir = cb[zj[0] == zs1]
    rd = np.sum(cbir / (1 + zj[0] * cbir))
    if rd > 0:
        lmin[0] = True
    else:
        zlmax = zj[0]
    
    cbil = cb[zj[mz - 1] == zs2]
    ld = np.sum(cbil / (1 + zj[mz - 1] * cbil))
    if ld < 0:
        lmin[mz - 1] = True
    else:
        zlmax = zj[mz - 1]

    # Flag all local minima and do a binary search between them to find the local maxima
    zlmin = zj[lmin]
    nlmin = len(zlmin)
    for j in range(1, nlmin):
        zlo = zlmin[j - 1]
        zhi = zlmin[j]
        ze = (zlo + zhi) / 2
        zstep = (zhi - zlo) / 2
        for nit in range(10):
            cbi = cb[(ze >= zs1) & (ze <= zs2)]
            likd = np.sum(cbi / (1 + ze * cbi))
            zstep /= 2
            ze += zstep * np.sign(likd)
        zlmax = np.append(zlmax, ze)
    
    # Evaluate all local maxima and find global max;
    # use smaller value if there is an exact tie for the global maximum.
    nlmax = len(zlmax)
    zm = np.empty(nlmax)
    for j in range(nlmax):
        pz = np.maximum(zs1, np.minimum(zlmax[j], zs2))
        zm[j] = np.sum(np.log(1 + cb * pz))
    zeta = zlmax[zm == np.max(zm)]
    zeta = np.min(zeta)
    w = np.minimum(1, np.maximum(zeta * cs, pilo))
    return {"zeta": zeta, "w": w, "cs": cs, "pilo": pilo}
