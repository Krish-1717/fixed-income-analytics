# fixed-income-analytics

Zero-coupon bond pricing, yield curve bootstrapping, duration/convexity analytics, and interest rate risk — pure Python, no dependencies.

## Features

- **Zero-coupon & coupon bond pricing** via discounted cash flows
- **Yield curve bootstrapping** from par rates using iterative stripping
- **Duration & convexity**: Macaulay, modified, and dollar duration
- **DV01 / PV01**: price sensitivity to 1 bp yield shift
- **Par rate, forward rate, and spot rate** extraction
- **Interest rate risk scenarios**: parallel shifts, steepeners, flatteners
- **Convexity-adjusted duration** for callable bonds (approximate)

## Quick Start

```python
from pricing.bond import CouponBond
from analytics.duration import modified_duration, dollar_convexity
from curves.bootstrap import YieldCurve

# Price a 5-year 4% annual coupon bond at 3.5% yield
bond = CouponBond(face=1000, coupon_rate=0.04, maturity=5, freq=2)
price = bond.price(ytm=0.035)   # 1022.58

# Duration & convexity
md = modified_duration(bond, ytm=0.035)   # ~4.49
dv01 = bond.dv01(ytm=0.035)              # 0.459

# Bootstrap yield curve from par rates
par_rates = {1: 0.030, 2: 0.032, 3: 0.034, 5: 0.037, 7: 0.039, 10: 0.042}
curve = YieldCurve.from_par_rates(par_rates)
spot_10y = curve.spot_rate(10)           # ~4.28%
fwd_5y5y = curve.forward_rate(5, 5)     # ~4.61%
```

## Project Structure

```
fixed-income-analytics/
├── pricing/
│   ├── bond.py          # CouponBond, ZeroCouponBond, FloatingRateNote
│   └── cashflows.py     # cash flow generation and PV utilities
├── analytics/
│   ├── duration.py      # Macaulay, modified, dollar duration, convexity, DV01
│   └── risk_metrics.py  # BPV, PVBP, key-rate durations
├── curves/
│   ├── bootstrap.py     # YieldCurve from par/spot/zero rates
│   └── interpolation.py # linear, log-linear, cubic spline interpolation
└── scenarios/
    └── rate_shocks.py   # parallel shift, twist, butterfly P&L
```

## Design

Pure-Python implementation with zero external dependencies. All math uses `math` stdlib only — suitable for interview prep, quant coursework, and embedding in larger systems without dependency overhead.

Implements textbook fixed-income analytics as covered in Fabozzi's *Fixed Income Mathematics* and Tuckman & Serrat's *Fixed Income Securities*.
