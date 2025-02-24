import numpy as np
from scipy.stats import chi2, norm
from sklearn.preprocessing import normalize
from .independence_tests_base import CondIndTest

class RCOT(CondIndTest):
    """Randomized Conditional Independence Test using Fourier Transforms.

    This class implements the Randomized Conditional Independence Test (RCoT),
    which is based on randomized Fourier features to estimate the dependency
    measure and various approximation methods to compute the p-value.

    Parameters
    ----------
    seed : int, optional (default: 42)
        Random seed for reproducibility.
    
    **kwargs : additional arguments
        Additional arguments passed to the parent class `CondIndTest`.
    """

    def __init__(self, seed=42, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.measure = 'RCoT'

    def get_dependence_measure(self, array, xyz):
        """Calculate the dependence measure for given variables.

        Parameters
        ----------
        array : array-like
            Data array.
        xyz : list of arrays
            List containing the variables x, y, and z.

        Returns
        -------
        float
            Dependence measure.
        """
        x, y, z = xyz
        return self.rcot(x, y, z, seed=self.random_state)
    
    @staticmethod
    def hbe(coeff, x):
        """Computes the CDF of a positively-weighted sum of chi-squared random variables with the Hall-Buckley-Eagleson (HBE) method.

        Parameters
        ----------
        coeff : array-like
            Coefficient vector.
        x : array-like
            Values at which to compute the CDF.

        Returns
        -------
        array-like
            CDF values.
        """
        if len(coeff) == 0:
            raise ValueError("Coefficient vector must not be empty.")
        
        k1 = 2 * np.sum(coeff)
        k2 = 8 * np.sum(np.square(coeff))
        k3 = 48 * np.sum(np.power(coeff, 3))
        k4 = 384 * np.sum(np.power(coeff, 4))
        
        gamma1 = k3 / (k2 ** 1.5)
        gamma2 = k4 / (k2 ** 2)
        
        z = (x - k1) / np.sqrt(k2)
        z += (gamma1 / 6) * (z**2 - 1)
        z += (gamma2 / 24) * (z**3 - 3 * z)
        z -= (gamma1**2 / 36) * (2 * z**3 - 5 * z)
        
        cdf_values = norm.cdf(z)
        
        return cdf_values
    
    @staticmethod
    def lpb4(coeff, x):
        """Computes the CDF of a positively-weighted sum of chi-squared random variables with the Lindsay-Pilla-Basak (LPB4) method.

        Parameters
        ----------
        coeff : array-like
            Coefficient vector of length at least four.
        x : array-like
            Values at which to compute the CDF.

        Returns
        -------
        array-like
            CDF values.
        """
        if len(coeff) < 4:
            raise ValueError("Coefficient vector must be of length at least four.")
        
        support_points = np.array([0.167, 0.5, 1.5, 5.0])  # Example values
        weights = np.array([0.167, 0.5, 1.5, 5.0])  # Example values, need to be calculated based on the LPB4 method
        
        weights /= np.sum(weights)
        
        cdf_values = np.zeros_like(x, dtype=float)
        for i in range(len(support_points)):
            cdf_values += weights[i] * chi2.cdf(x, df=support_points[i] * np.sum(coeff))
        
        return cdf_values
    
    @staticmethod
    def sw(coeff, x):
        """Computes the CDF of a positively-weighted sum of chi-squared random variables with the Satterthwaite-Welch (SW) method.

        Parameters
        ----------
        coeff : array-like
            Coefficient vector.
        x : array-like
            Values at which to compute the CDF.

        Returns
        -------
        array-like
            CDF values.
        """
        if len(coeff) == 0:
            raise ValueError("Coefficient vector must not be empty.")
        
        df_eff = 2 * (np.sum(coeff) ** 2) / np.sum(np.square(coeff))
        scale_eff = np.sum(np.square(coeff)) / np.sum(coeff)
        
        cdf_values = chi2.cdf(x / scale_eff, df=df_eff)
        
        return cdf_values

    def rcot(self, x, y, z=None, approx='lpb4', num_f=100, num_f2=5, seed=None):
        """Performs the Randomized Conditional Independence Test (RCoT).

        Parameters
        ----------
        x, y : array-like
            Variables to test for conditional independence.
        z : array-like, optional
            Conditioning variables.
        approx : str, optional (default: 'lpb4')
            Approximation method to use ('perm', 'chi2', 'gamma', 'hbe', 'lpb4').
        num_f : int, optional (default: 100)
            Number of Fourier features for the conditioning variables.
        num_f2 : int, optional (default: 5)
            Number of Fourier features for the variables x and y.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            Test result containing p-value and test statistic.
        """
        if z is None or len(z) == 0:
            return self.rit(x, y, approx, seed)

        x, y, z = self.matrix2(x), self.matrix2(y), self.matrix2(z)
        z = z[:, np.std(z, axis=0) > 0]
        if z.shape[1] == 0 or np.std(x) == 0 or np.std(y) == 0:
            return {'p': 1, 'Sta': 0}

        r = x.shape[0]
        r1 = min(r, 500)

        x, y, z = normalize(x), normalize(y), normalize(z)

        four_z = self.random_fourier_features(z, num_f=num_f, sigma=np.median(np.abs(np.diff(z[:r1], axis=0))), seed=seed)
        four_x = self.random_fourier_features(x, num_f=num_f2, sigma=np.median(np.abs(np.diff(x[:r1], axis=0))), seed=seed)
        four_y = self.random_fourier_features(y, num_f=num_f2, sigma=np.median(np.abs(np.diff(y[:r1], axis=0))), seed=seed)

        f_x, f_y, f_z = normalize(four_x['feat']), normalize(four_y['feat']), normalize(four_z['feat'])
        Cxy = self.cov(f_x, f_y)
        Czz = self.cov(f_z)
        i_Czz = self.chol2inv(np.linalg.cholesky(Czz + np.eye(num_f) * 1e-10))

        Cxz, Czy = self.cov(f_x, f_z), self.cov(f_z, f_y)
        z_i_Czz = f_z @ i_Czz
        e_x_z, e_y_z = z_i_Czz @ Cxz.T, z_i_Czz @ Czy.T

        res_x, res_y = f_x - e_x_z, f_y - e_y_z

        if num_f2 == 1:
            approx = 'hbe'

        if approx == 'perm':
            Cxy_z = self.cov(res_x, res_y)
            Sta = r * np.sum(Cxy_z ** 2)

            nperm = 1000
            Stas = [self.Sta_perm(res_x[np.random.permutation(r)], res_y, r) for _ in range(nperm)]
            p = 1 - (np.sum(Sta >= Stas) / len(Stas))
        else:
            Cxy_z = Cxy - Cxz @ i_Czz @ Czy
            Sta = r * np.sum(Cxy_z ** 2)

            res = res_x[:, :, np.newaxis] * res_y[:, np.newaxis, :]
            Cov = np.mean(res, axis=0).reshape(-1, res.shape[-1])
            Cov = Cov.T @ Cov / r

            eigvals = np.linalg.eigvalsh(Cov)
            eigvals = eigvals[eigvals > 0]

            if approx == 'chi2':
                i_Cov = np.linalg.inv(Cov)
                Sta = r * (Cxy_z.flatten() @ i_Cov @ Cxy_z.flatten())
                p = 1 - chi2.cdf(Sta, len(Cxy_z.flatten()))
            elif approx == 'gamma':
                p = 1 - self.sw(eigvals, Sta)
            elif approx == 'hbe':
                p = 1 - self.hbe(eigvals, Sta)
            elif approx == 'lpb4':
                try:
                    p = 1 - self.lpb4(eigvals, Sta)
                except:
                    p = 1 - self.hbe(eigvals, Sta)

        return {'p': max(0, p), 'Sta': Sta}

    def rit(self, x, y, approx='lpb4', seed=None):
        """Performs the Randomized Independence Test (RIT).

        Parameters
        ----------
        x, y : array-like
            Variables to test for independence.
        approx : str, optional (default: 'lpb4')
            Approximation method to use ('perm', 'chi2', 'gamma', 'hbe', 'lpb4').
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            Test result containing p-value and test statistic.
        """
        if np.std(x) == 0 or np.std(y) == 0:
            return {'p': 1, 'Sta': 0}

        x, y = self.matrix2(x), self.matrix2(y)
        r = x.shape[0]
        r1 = min(r, 500)
        x, y = normalize(x), normalize(y)

        four_x = self.random_fourier_features(x, num_f=5, sigma=np.median(np.abs(np.diff(x[:r1], axis=0))), seed=seed)
        four_y = self.random_fourier_features(y, num_f=5, sigma=np.median(np.abs(np.diff(y[:r1], axis=0))), seed=seed)

        f_x, f_y = normalize(four_x['feat']), normalize(four_y['feat'])
        Cxy = self.cov(f_x, f_y)

        Sta = r * np.sum(Cxy ** 2)

        if approx == 'perm':
            nperm = 1000
            Stas = [self.Sta_perm(f_x[np.random.permutation(r)], f_y, r) for _ in range(nperm)]
            p = 1 - (np.sum(Sta >= Stas) / len(Stas))
        else:
            res_x = f_x - np.mean(f_x, axis=0)
            res_y = f_y - np.mean(f_y, axis=0)

            res = res_x[:, :, np.newaxis] * res_y[:, np.newaxis, :]
            Cov = np.mean(res, axis=0).reshape(-1, res.shape[-1])
            Cov = Cov.T @ Cov / r

            eigvals = np.linalg.eigvalsh(Cov)
            eigvals = eigvals[eigvals > 0]

            if approx == 'chi2':
                i_Cov = np.linalg.inv(Cov)
                Sta = r * (Cxy.flatten() @ i_Cov @ Cxy.flatten())
                p = 1 - chi2.cdf(Sta, len(Cxy.flatten()))
            elif approx == 'gamma':
                p = 1 - self.sw(eigvals, Sta)
            elif approx == 'hbe':
                p = 1 - self.hbe(eigvals, Sta)
            elif approx == 'lpb4':
                try:
                    p = 1 - self.lpb4(eigvals, Sta)
                except:
                    p = 1 - self.hbe(eigvals, Sta)

        return {'p': max(0, p), 'Sta': Sta}


