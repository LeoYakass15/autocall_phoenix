import numpy as np

from autocall.vol_surface import SSVI, calibrate_ssvi, flat_surface, local_vol


def test_flat_surface_has_flat_local_vol():
    k, T = np.linspace(-1, 1, 21), 1.5
    assert np.allclose(flat_surface(0.25).implied_vol(k, T), 0.25)
    assert np.allclose(local_vol(flat_surface(0.25), k, T), 0.25, atol=1e-4)


def test_calibration_recovers_the_surface():
    times, k = np.array([0.5, 1.0, 2.0]), np.linspace(-0.7, 0.4, 12)
    true = SSVI(times, np.array([0.30, 0.28, 0.27]) ** 2 * times, rho=-0.5, eta=0.9, gamma=0.35)
    fitted = calibrate_ssvi(times, k, true.implied_vol(k[None, :], times[:, None]))
    assert abs(fitted.rho - true.rho) < 0.01
    assert np.allclose(fitted.implied_vol(k[None, :], times[:, None]), true.implied_vol(k[None, :], times[:, None]),
                       atol=1e-4)


def test_local_skew_is_steeper_than_implied_skew():
    surface = SSVI(np.array([1.0]), np.array([0.09]), rho=-0.5, eta=1.0)
    k = np.array([-0.05, 0.05])
    implied_skew = np.diff(surface.implied_vol(k, 1.0))[0]
    local_skew = np.diff(local_vol(surface, k, 1.0))[0]
    assert local_skew < implied_skew < 0
