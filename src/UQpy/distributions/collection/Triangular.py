from UQpy.distributions.baseclass import DistributionContinuous1D
import scipy.stats as stats

class Triangular(DistributionContinuous1D):
    def __init__(self, c, loc, scale):
        super().__init__(c=c, loc=loc, scale=scale, ordered_parameters=("loc", "scale"))
        self._construct_from_scipy(scipy_name=stats.triang)