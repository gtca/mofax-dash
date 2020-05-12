import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mofax-dash-gtca",
    version="0.1.0",
    author="Danila Bredikhin",
    author_email="danila.bredikhin@embl.de",
    description="Dash app for mofax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gtca/mofax-dash",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=['bin/mofax'],
    python_requires=">=3.6",
    install_requires=["numpy", "pandas", "matplotlib", "seaborn", "h5py", "dash", "dash-bootstrap-components"],
)
