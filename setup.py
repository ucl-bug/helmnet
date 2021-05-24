import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="helmnet",  # Replace with your own username
    version="0.1.0",
    author="Antonio Stanziola",
    author_email="a.stanziola@ucl.ac.uk",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bug.medphys.ucl.ac.uk",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
