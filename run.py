"""
Run file
"""
from model import run_model_split

if __name__ == '__main__':
    run_model_split(apply_cross_validation=False, create_submission=True)
