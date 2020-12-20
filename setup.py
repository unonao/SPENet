import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SPENet",
    version="0.0.1",
    description="Sum of Powers of Eigenvalues of Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/unonao/SPENet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
