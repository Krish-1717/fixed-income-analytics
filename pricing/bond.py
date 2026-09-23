"""
pricing/bond.py -- Fixed-income bond pricing for fixed-income-analytics
Pure-Python: zero external dependencies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Cash-flow utilities
# ---------------------------------------------------------------------------

def discount_factor(rate: float, t: float) -> float:
    """Continuous compounding discount factor: e^{-r*t}."""
    return math.exp(-rate * t)


def pv_cashflows(cashflows: List[Tuple[float, float]], rate: float) -> float:
    """
    Present value of a list of (time, amount) cash flows
    discounted at a flat continuously-compounded rate.
    """
    return sum(cf * discount_factor(rate, t) for t, cf in cashflows)


def ytm_to_semi(ytm_annual: float) -> float:
    """Convert annual YTM to semi-annual equivalent."""
    return 2.0 * (math.sqrt(1.0 + ytm_annual) - 1.0)


# ---------------------------------------------------------------------------
# Zero-coupon bond
# ---------------------------------------------------------------------------

@dataclass
class ZeroCouponBond:
    """Zero-coupon bond: single cash flow at maturity."""
    face: float        # par / face value
    maturity: float    # years to maturity

    def price(self, ytm: float) -> float:
        """Price given continuously-compounded yield."""
        return self.face * math.exp(-ytm * self.maturity)

    def ytm(self, price: float) -> float:
        """Continuously-compounded YTM given market price."""
        if price <= 0:
            raise ValueError("Price must be positive")
        return -math.log(price / self.face) / self.maturity

    def duration(self) -> float:
        """Macaulay duration equals maturity for ZCB."""
        return self.maturity

    def modified_duration(self, ytm: float) -> float:
        """Modified duration: D / (1 + y) -- approximate for continuous."""
        return self.maturity  # for cc, mod_dur == mac_dur

    def dv01(self, ytm: float) -> float:
        """Dollar value of 1 basis point."""
        return self.price(ytm) * self.maturity * 0.0001


# ---------------------------------------------------------------------------
# Coupon bond
# ---------------------------------------------------------------------------

@dataclass
class CouponBond:
    """
    Fixed-rate coupon bond with equal periodic coupons.

    Parameters
    ----------
    face : par value
    coupon_rate : annual coupon rate (e.g. 0.05 for 5%)
    maturity : years to maturity
    freq : coupon payments per year (1=annual, 2=semi-annual)
    """
    face: float
    coupon_rate: float
    maturity: float
    freq: int = 2

    def cashflows(self) -> List[Tuple[float, float]]:
        """Return list of (time, amount) for all coupon + principal payments."""
        n = int(round(self.maturity * self.freq))
        dt = 1.0 / self.freq
        coupon = self.face * self.coupon_rate / self.freq
        cfs = [(i * dt, coupon) for i in range(1, n + 1)]
        cfs[-1] = (cfs[-1][0], cfs[-1][1] + self.face)   # add principal
        return cfs

    def price(self, ytm: float) -> float:
        """
        Price the bond given annually-quoted YTM (periodic compounding).
        Uses standard bond pricing formula with periodic discounting.
        """
        r = ytm / self.freq   # per-period rate
        cfs = self.cashflows()
        n = len(cfs)
        coupon = self.face * self.coupon_rate / self.freq
        # Closed-form when cashflows are even
        if r == 0:
            return coupon * n + self.face
        pv_coupons = coupon * (1 - (1 + r) ** (-n)) / r
        pv_face = self.face * (1 + r) ** (-n)
        return pv_coupons + pv_face

    def ytm(self, price: float, tol: float = 1e-10, max_iter: int = 200) -> float:
        """YTM via bisection on price(ytm) = price."""
        lo, hi = 1e-6, 10.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            p = self.price(mid)
            if abs(p - price) < tol:
                return mid
            if p > price:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def macaulay_duration(self, ytm: float) -> float:
        """Macaulay duration: weighted average time to cash flow."""
        price = self.price(ytm)
        r = ytm / self.freq
        cfs = self.cashflows()
        weighted = sum(
            t * cf * (1 + r) ** (-i - 1) * self.freq
            for i, (t, cf) in enumerate(cfs)
        )
        # Simpler: weighted time / price
        total_pv = 0.0
        weighted_t = 0.0
        for i, (t, cf) in enumerate(cfs):
            pv = cf / (1 + r) ** (i + 1)
            total_pv += pv
            weighted_t += t * pv
        return weighted_t / total_pv

    def modified_duration(self, ytm: float) -> float:
        """Modified duration: Macaulay / (1 + y/freq)."""
        return self.macaulay_duration(ytm) / (1 + ytm / self.freq)

    def convexity(self, ytm: float) -> float:
        """
        Convexity: second derivative of price w.r.t. yield, normalised by price.
        """
        price = self.price(ytm)
        r = ytm / self.freq
        cfs = self.cashflows()
        n = len(cfs)
        total = 0.0
        for i, (t, cf) in enumerate(cfs):
            pv = cf / (1 + r) ** (i + 1)
            total += pv * (i + 1) * (i + 2)
        return total / (price * (1 + r) ** 2 * self.freq ** 2)

    def dv01(self, ytm: float) -> float:
        """Dollar value of 1 basis point (DV01 / PV01)."""
        return abs(self.price(ytm - 0.0001) - self.price(ytm + 0.0001)) / 2

    def price_change_approx(self, ytm: float, delta_y: float) -> float:
        """
        Approximate price change for a yield shift of delta_y
        using duration + convexity (Taylor expansion).
        """
        p = self.price(ytm)
        md = self.modified_duration(ytm)
        cx = self.convexity(ytm)
        return p * (-md * delta_y + 0.5 * cx * delta_y ** 2)


# ---------------------------------------------------------------------------
# Floating-rate note (simple reset model)
# ---------------------------------------------------------------------------

@dataclass
class FloatingRateNote:
    """
    Simplified FRN: resets every period to the prevailing libor/sofr rate.
    At reset date, always prices near par (ignoring credit/liquidity spread).
    """
    face: float
    maturity: float
    spread: float = 0.0   # spread over floating rate
    freq: int = 2

    def price_at_reset(self, current_rate: float) -> float:
        """Price immediately after last reset = par (if spread=0)."""
        if self.spread == 0:
            return self.face
        # Value of spread payments
        n = int(round(self.maturity * self.freq))
        r = current_rate / self.freq
        spread_pmt = self.face * self.spread / self.freq
        if r == 0:
            pv_spread = spread_pmt * n
        else:
            pv_spread = spread_pmt * (1 - (1 + r) ** (-n)) / r
        return self.face + pv_spread


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  FIXED-INCOME ANALYTICS — pricing/bond.py demo")
    print("=" * 60)

    # Zero-coupon bond
    zcb = ZeroCouponBond(face=1000, maturity=10)
    print(f"  ZCB(10y) @ 4%:  price={zcb.price(0.04):.4f}  ytm check={zcb.ytm(zcb.price(0.04))*100:.4f}%")
    print(f"  ZCB DV01:        {zcb.dv01(0.04):.4f}")

    print()

    # Coupon bond — 5y 4% semi-annual, priced at 3.5% YTM
    bond = CouponBond(face=1000, coupon_rate=0.04, maturity=5, freq=2)
    ytm = 0.035
    p = bond.price(ytm)
    md = bond.modified_duration(ytm)
    cx = bond.convexity(ytm)
    dv01 = bond.dv01(ytm)
    print(f"  5y 4% s/a bond @ YTM={ytm*100:.1f}%:")
    print(f"    Price:              {p:.4f}")
    print(f"    Macaulay duration:  {bond.macaulay_duration(ytm):.4f} yr")
    print(f"    Modified duration:  {md:.4f}")
    print(f"    Convexity:          {cx:.4f}")
    print(f"    DV01:               {dv01:.4f}")
    print(f"    Approx dP (+50bp):  {bond.price_change_approx(ytm, 0.005):+.4f}")
    print(f"    Actual dP (+50bp):  {bond.price(ytm+0.005)-p:+.4f}")

    print()

    # YTM round-trip
    recovered = bond.ytm(p)
    print(f"  YTM round-trip:     input={ytm*100:.4f}% recovered={recovered*100:.4f}%")
    print("=" * 60)
