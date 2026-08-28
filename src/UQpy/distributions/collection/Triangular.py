from UQpy.distributions.baseclass import DistributionContinuous1D
import scipy.stats as stats

class Triangular(DistributionContinuous1D):
    def __init__(self, c, loc=0., scale=1.):
        super().__init__(c=c, loc=loc, scale=scale, ordered_parameters=("c", "loc", "scale"))
        self._construct_from_scipy(scipy_name=stats.triang)