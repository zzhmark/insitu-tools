# insitu-tools
A set of command line tools that find and compare *In situ* expression patterns between mRNA expression images, specifically, the *Drosophila* embryos.

The algorithm used is described in this [paper](https://dl.acm.org/doi/10.1145/974614.974636).

Some steps are taken to enhance performance, including:

* Coversion of Color images to grayscale, differentiating stains and embryo texture.

* Linear Transformation of intensity to normalize signals across images.

* 4 orientations' comparison to determine the best score.

* Normalize local GMM scores using the self comparison score.

* Global GMM scores (mutual information scores) are calculated on the union area of the 2 embryos.

* Down sampling is put in registration.

Usages are described in [wiki](https://github.com/zzhmark/insitu-tools/wiki).

# Installation

```
pip install seu-insitu-tools
```