import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="seu-insitu-tools",  # Replace with your own username
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={"console_scripts": ["insituTools=insituTools.cli:main"]},
    license="MIT",
    include_package_data=True,
)
