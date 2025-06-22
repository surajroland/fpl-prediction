from setuptools import setup, find_packages

setup(
    name="fpl-xgboost",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        "xgboost>=2.0.3",
        "pandas>=2.1.4", 
        "numpy>=1.25.2",
        "scikit-learn>=1.3.2"
    ]
)
