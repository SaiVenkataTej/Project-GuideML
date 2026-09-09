from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="guideml",
    version="1.0.0",
    author="GuideML Contributors",
    description="Automated Machine Learning Recommender System for Tabular Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "interface": ["templates/*", "static/**/*"],
    },
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib>=1.3.0",
        "matplotlib",
        "seaborn",
        "Flask",
        "markdown",
        "scipy",
        "optuna",
        "shap",
        "gunicorn>=21.2.0",
    ],
    entry_points={
        "console_scripts": [
            "guideml=interface.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
