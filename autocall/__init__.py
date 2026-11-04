"""Worst-of Phoenix autocallable on Bloomberg data: local vol pricer and structuring toolkit."""

from .market import Market, bloomberg_market
from .pricer import Pricing, solve_coupon
from .product import Phoenix
