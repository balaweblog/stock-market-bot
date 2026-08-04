"""
controllers package: Advisor Controllers & Pipeline Execution Drivers
"""

def run_stock_advisor(*args, **kwargs):
    from .stock_controller import main
    return main(*args, **kwargs)

def run_nifty_advisor(*args, **kwargs):
    from .nifty_stock_controller import run
    return run(*args, **kwargs)

def run_swing_advisor(*args, **kwargs):
    from .swing_controller import run
    return run(*args, **kwargs)

def run_option_advisor(*args, **kwargs):
    from .option_controller import run
    return run(*args, **kwargs)

def run_mutual_fund_advisor(*args, **kwargs):
    from .mutual_fund_controller import run
    return run(*args, **kwargs)
