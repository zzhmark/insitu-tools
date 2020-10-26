# insitu-tools

A set of command line tools that find and compare *In situ* expression patterns between mRNA expression images, specifically, the *Drosophila* embryos.

The algorithm used is described in this [paper](https://dl.acm.org/doi/10.1145/974614.974636).

![Flow chart](https://github.com/zzhmark/insitu-tools/blob/main/insituTools.png)

Some steps are taken to enhance performance, including:

* Coversion of Color images to grayscale, differentiating stains and embryo texture.

* Linear Transformation of intensity to normalize signals across images.

* 4 orientations' comparison to determine the best score.

* Normalize local GMM scores using the self comparison score.

* Global GMM scores (mutual information scores) are calculated on the union area of the 2 embryos.

## Installation

```
pip install seu-insitu-tools
```

## Usages

Detailed descriptions of commands and their arugments are located in our [wiki](https://github.com/zzhmark/insitu-tools/wiki).

## Tutorial
Check out the [example](https://github.com/zzhmark/insitu-tools/wiki/Example:-Comparing-Fly-Embryo-Staining-Patterns) in our wiki for quick learning of the tools.
