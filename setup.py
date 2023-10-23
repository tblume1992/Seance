# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Seance",
    version="0.1.1",
    author="Tyler Blume",
    url="https://github.com/tblume1992/Seance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description = "A Wrapper around MLForecast.",
    author_email = 't-blume@hotmail.com',
    keywords = ['forecasting', 'time series', 'lightgbm', 'mlforecast'],
      install_requires=[           
                        'numpy',
                        'pandas',
                        'optuna',
                        'mlforecast',
                        'lightgbm',
                        'catboost',
                        'xgboost'
                        ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


