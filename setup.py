# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Seance",
    version="0.0.5",
    author="Tyler Blume",
    url="https://github.com/tblume1992/Seance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description = "A Wrapper around MLForecast.",
    author_email = 'tblume@mail.USF.edu', 
    keywords = ['forecasting', 'time series', 'lightgbm', 'mlforecast'],
      install_requires=[           
                        'numpy',
                        'pandas',
                        'optuna',
                        'mlforecast'
                        ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


