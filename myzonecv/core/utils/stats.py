import scipy.stats as stats


def TruncNorm(lower=-1, upper=1, mu=0, sigma=1):
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    TN = stats.truncnorm(a, b, loc=mu, scale=sigma)
    return TN
