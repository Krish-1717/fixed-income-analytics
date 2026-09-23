"""
curves/bootstrap.py -- Yield curve bootstrapping from par/spot rates
Part of fixed-income-analytics. Pure Python, no external dependencies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _linear_interp(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if abs(x1 - x0) < 1e-12:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class YieldCurve:
    """
    Yield curve represented as (maturity, continuously-compounded spot rate) pairs.

    Build from market data via class methods:
      YieldCurve.from_par_rates(par_rates)
      YieldCurve.from_spot_rates(spot_rates)

    Query via:
      curve.spot_rate(t)       -- CC spot rate at t years
      curve.discount_factor(t) -- e^{-r(t)*t}
      curve.forward_rate(t1, t2) -- CC forward rate between t1 and t2
      curve.par_rate(t)        -- par rate for maturity t
    """
    _spots: List[Tuple[float, float]] = field(default_factory=list)  # (t, r)

    @classmethod
    def from_spot_rates(cls, spot_rates: Dict[float, float]) -> "YieldCurve":
        """Build directly from {maturity: spot_rate} dict."""
        curve = cls()
        curve._spots = sorted(spot_rates.items())
        return curve

    @classmethod
    def from_par_rates(
        cls,
        par_rates: Dict[float, float],
        freq: int = 2,
    ) -> "YieldCurve":
        """
        Bootstrap spot rates from par rates using iterative stripping.

        Parameters
        ----------
        par_rates : {maturity_in_years: par_rate}  -- annual par rates
        freq      : coupon payments per year (default 2 = semi-annual)
        """
        maturities = sorted(par_rates.keys())
        dt = 1.0 / freq
        spot_rates: Dict[float, float] = {}

        for mat in maturities:
            par = par_rates[mat]
            coupon = par / freq
            n = int(round(mat * freq))

            # Sum of discount factors for all coupon periods before maturity
            pv_coupons = 0.0
            for i in range(1, n):
                t_i = i * dt
                r_i = cls._interp_spot(spot_rates, t_i)
                pv_coupons += coupon * math.exp(-r_i * t_i)

            # Solve for the final discount factor
            # 1 = coupon * sum(df_i) + (1 + coupon) * df_n
            # df_n = (1 - coupon * sum(df_i)) / (1 + coupon)
            df_n = (1.0 - pv_coupons) / (1.0 + coupon)
            if df_n <= 0:
                raise ValueError(f"Non-positive discount factor at maturity {mat}y — check par rates")
            spot_n = -math.log(df_n) / mat
            spot_rates[mat] = spot_n

        curve = cls()
        curve._spots = sorted(spot_rates.items())
        return curve

    @staticmethod
    def _interp_spot(spot_rates: Dict[float, float], t: float) -> float:
        """Linearly interpolate (or extrapolate) CC spot rate at t."""
        if not spot_rates:
            return 0.0
        keys = sorted(spot_rates.keys())
        if t <= keys[0]:
            return spot_rates[keys[0]]
        if t >= keys[-1]:
            return spot_rates[keys[-1]]
        for i in range(len(keys) - 1):
            if keys[i] <= t <= keys[i + 1]:
                return _linear_interp(keys[i], spot_rates[keys[i]],
                                      keys[i + 1], spot_rates[keys[i + 1]], t)
        return spot_rates[keys[-1]]

    def spot_rate(self, t: float) -> float:
        """CC spot rate at maturity t years (linear interpolation)."""
        if not self._spots:
            raise ValueError("Yield curve is empty")
        spots_dict = dict(self._spots)
        return self._interp_spot(spots_dict, t)

    def discount_factor(self, t: float) -> float:
        """Discount factor: e^{-r(t)*t}."""
        return math.exp(-self.spot_rate(t) * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Continuously-compounded forward rate between t1 and t2.
        f(t1,t2) = [r(t2)*t2 - r(t1)*t1] / (t2 - t1)
        """
        if t2 <= t1:
            raise ValueError("t2 must be > t1")
        r1 = self.spot_rate(t1) * t1
        r2 = self.spot_rate(t2) * t2
        return (r2 - r1) / (t2 - t1)

    def par_rate(self, maturity: float, freq: int = 2) -> float:
        """Compute par rate for a bond of given maturity (annual, periodic cpn)."""
        n = int(round(maturity * freq))
        dt = 1.0 / freq
        sum_df = sum(self.discount_factor(i * dt) for i in range(1, n + 1))
        df_n = self.discount_factor(maturity)
        if sum_df == 0:
            raise ValueError("Zero sum of discount factors")
        return freq * (1.0 - df_n) / sum_df

    def term_structure(self, maturities: Optional[List[float]] = None) -> List[Tuple[float, float, float]]:
        """
        Return (maturity, spot, par_rate) for a list of maturities.
        Default maturities: 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.
        """
        if maturities is None:
            maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
            maturities = [m for m in maturities if m <= max(t for t, _ in self._spots)]
        result = []
        for m in maturities:
            try:
                s = self.spot_rate(m)
                p = self.par_rate(m)
                result.append((m, s, p))
            except Exception:
                pass
        return result


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    par_rates = {
        0.5: 0.028,
        1.0: 0.030,
        2.0: 0.032,
        3.0: 0.034,
        5.0: 0.037,
        7.0: 0.039,
        10.0: 0.042,
    }

    curve = YieldCurve.from_par_rates(par_rates)

    print("=" * 55)
    print("  YIELD CURVE BOOTSTRAP  (par -> spot -> forward)")
    print("=" * 55)
    print(f"  {'Maturity':>8}  {'Spot rate':>10}  {'Par rate':>10}")
    print("  " + "-" * 32)
    for mat, spot, par in curve.term_structure():
        print(f"  {mat:>7.1f}y  {spot*100:>9.4f}%  {par*100:>9.4f}%")

    print()
    print(f"  5y->10y forward: {curve.forward_rate(5,10)*100:.4f}%")
    print(f"  1y->2y forward:  {curve.forward_rate(1, 2)*100:.4f}%")
    print(f"  DF(10y):         {curve.discount_factor(10):.6f}")
    print("=" * 55)
