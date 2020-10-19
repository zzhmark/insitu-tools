import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="insituTools-Odehan", # Replace with your own username
    version="0.0.1",
    author="Zuohan Zhao",
    author_email="zzhmark@126.com",
    description="Find and compare In Situ expression in Drosophila embryos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zzhmark/insitu-tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)