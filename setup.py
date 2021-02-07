from setuptools import setup


setup(
    name = "FractionalOctave",
    version = "0.1",
    author = "Cong-Van Nguyen",
    author_email = "congvannguyen@gmail.com",
    description = "Python package to do spectral analysis using fractional-octave filterbanks",
    license = "MIT",
    url = "https://github.com/cvn-git/FractionalOctave",
    packages=['FractionalOctave'],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    extras_require={
        "development": [
            "flake8",
            "mypy",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
)
