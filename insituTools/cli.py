import os

import fire
import cv2
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, dok_matrix

from insituTools import algorithm
from insituTools import __version__

class InsituTools(object):
    """
    In Situ expression patterns analysis toolset.

    | CLI that implements Peng's algorithm that detects and compares
    | patterns in In Situ expression image of Drosophila embryos. It
    | also has the potential of being applied to other kinds of images.
    """

    @classmethod
    def version(cls):
        """
        Show current insituTools package version.
        """
        return __version__

    @classmethod
    def extract(
        cls,
        inputImage,
        outputMask,
        outputImage=None,
        filterSize=3,
        threshold=3.0,
        grayscale=False,
    ):
        """
        Foreground extraction.

        | Extract the foreground area based the
        | local standard deviation of the image.

        :param inputImage: The path to the input image.
        :param outputMask: The path to the output mask.
        :param outputImage: The path to the output image.
                            No image will be generated if omitted.
        :param filterSize: An integer specifying the size of the filter
                            that calculates local standard deviation.
                            Defaults to 3.
        :param threshold: A float determine the minimum foreground
                            standard deviation.
                            Defaults to 3.0.
        :param grayscale: Read the input image as grayscale, which can
                            save some computational resource.
        """
        image = cv2.imread(
            inputImage, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        )
        if image is None:
            raise IOError("Image reading failure.")
        image, mask = algorithm.extract(
            image, filterSize, threshold, outputImage is None
        )
        if outputImage is not None:
            if not cv2.imwrite(outputImage, image):
                raise IOError("Image writing failure.")
        if not cv2.imwrite(outputMask, mask):
            raise IOError("Mask writing failure.")

    @classmethod
    def register(
        cls,
        inputImage,
        inputMask,
        outputImage,
        outputMask,
        targetSize=None,
        noRotation=False,
        noIntensityRescale=False,
        noRectification=False,
        grayscale=False,
    ):
        """
        Image and mask registration.

        | Register the image so that its longest axis is horizontal.
        | Also, the image can be convert to grayscale and its
        | intensity can be linearly transformed.

        :param inputImage: The path to the input image.
        :param inputMask: The path to the input mask.
        :param outputImage: The path to the output image.
                            No image will be generated if omitted.
        :param outputMask: The path to the output mask.
        :param targetSize: An integer tuple that specifies the width and height of the
                            registered image and mask before down sampling if has.
                            When omitted, no resizing will be performed.
                            The tuple should be like (WIDTH, HEIGHT).
        :param noRotation: Skip rotation, which can save some computational resource
                            when rotation is already performed.
        :param noIntensityRescale: Leave the intensity distribution as original.
        :param noRectification: Color image will be converted to grayscale using the
                                default algorithm in OpenCV.
        :param grayscale: Read the input image as grayscale, which can
                            save some computational resource.
        """
        image = cv2.imread(
            inputImage, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        )
        mask = cv2.imread(inputMask, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise IOError("Image or mask reading failure.")
        image, mask = algorithm.register(
            image,
            mask,
            targetSize,
            not noRotation,
            not noIntensityRescale,
            not noRectification,
        )
        if not cv2.imwrite(outputImage, image) or not cv2.imwrite(outputMask, mask):
            raise IOError("Image or mask writing failure.")

    @classmethod
    def globalGMM(
        cls,
        inputImage,
        inputMask,
        outputMask,
        outputLabel,
        outputLevels,
        outputImage=None,
        downSampleFactor=1,
        numberOfGlobalKernels=5,
    ):
        """
        Global Gaussian mixture model fitting.

        | Apply the GMM decomposition globally to segment
        | the foreground. It uses EM method to adaptively
        | cluster the pixels based on their intensity.

        :param inputImage: The path to the input image，
                            read as grayscale.
        :param inputMask: The path to the input mask.
        :param outputMask: The path to the output mask, which is down sampled
                            when down sampling factor is greater than 1.
        :param outputLabel: The path to the output label file.
        :param outputLevels:The path to the file to keep the
                                globally generated levels.
        :param outputImage: The path to the output image.
                            No image will be generated if omitted.
        :param numberOfGlobalKernels: The number of Gaussian mixture kernels
                                        for each iteration during expectation maximization.
                                        Note that if the number of kernels exceeds
                                        that of the sample pixels, the latter will
                                        take the place.
        :param downSampleFactor: An integer that specifies the side length of the local
                                    patch that is used to pool the image and mask.
                                    Default as 1, which means no down sampling will be performed.
        """
        image = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(inputMask, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise IOError("Image or mask reading failure.")
        image, mask, label, levels = algorithm.global_gmm(
            image, mask, downSampleFactor, numberOfGlobalKernels, outputImage is None
        )
        if image is not None:
            if not cv2.imwrite(outputImage, image):
                raise IOError("Image writing failure.")
        if not cv2.imwrite(outputLabel, label) or not cv2.imwrite(outputMask, mask):
            raise IOError("Mask or label writing failure.")
        with open(outputLevels, "w") as f:
            f.write("\n".join(str(i) for i in levels))

    @classmethod
    def localGMM(
        cls,
        inputLabel,
        inputLevels,
        outputLabel,
        outputLevels,
        inputImage=None,
        outputImage=None,
        limitOfLocalKernels=10,
    ):
        """
        Local Gaussian mixture model fitting.

        | Apply the GMM decomposition locally to further segment
        | the clusters generated by globalGMM. It estimates the
        | spatial distribution of pixels in the same cluster from
        | globalGMM and generate multiple blobs.

        :param inputImage: The path to the input image，
                            read as grayscale.
        :param inputLabel: The path to the input global label file.
        :param inputLevels: The path to the text file that
                            keeps the globally generated levels separated in lines.
                            Note that the number of these levels should
                            match with the label file, otherwise either
                            some labels will be omitted or some levels
                            will have no labeling.
        :param outputLabel: The path to the output label.
        :param outputLevels: The path to the file to keep the
                                locally generated levels.
        :param outputImage: The path to the output image.
                            No image will be generated if omitted.
        :param limitOfLocalKernels: The upper limit of the number of
                                    Gaussian mixture kernels for the
                                    adaptive Bayesian algorithm. Note
                                    that if the limit exceeds that of
                                    the sample pixels, the latter will
                                    take the place.
        """
        image = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(inputLabel, cv2.IMREAD_GRAYSCALE)
        if image is None and inputImage is not None or label is None:
            raise IOError("Image or label reading failure.")
        with open(inputLevels) as f:
            levels = [float(i) for i in f.readlines()]
        image, label, levels = algorithm.local_gmm(
            label, levels, limitOfLocalKernels, image
        )
        if image is not None:
            if not cv2.imwrite(outputImage, image):
                raise IOError("Image writing failure.")
        if not cv2.imwrite(outputLabel, label):
            raise IOError("Label writing failure.")
        with open(outputLevels, "w") as f:
            f.write("\n".join(str(i) for i in levels))

    @classmethod
    def findPatterns(
        cls,
        inputImage,
        outputDirectory=None,
        name=None,
        filterSize=3,
        threshold=3.0,
        targetSize=None,
        downSampleFactor=None,
        numberOfGlobalKernels=5,
        limitOfLocalKernels=10,
        noRotation=False,
        noIntensityRescale=False,
        noRectification=False,
        grayscale=False,
        saveImage=False,
        noRemoveBackground=False,
        noExtractionImage=False,
        noRegistrationImage=False,
        noGlobalGMMImage=False,
        noLocalGMMImage=False,
    ):
        """
        Find global and local patterns, beginning from the raw image.

        | This command merges all image processing steps.

        :param inputImage: The path to the input image.
        :param outputDirectory: The path to the directory to store all
                                the output files. Duplicated files will
                                be overwritten. If omitted, output files
                                will be stored in the same directory as
                                the input image.
        :param name: The mutual basename prefix for each output file.
                        if omitted, the prefix will be the basename of
                        the input image.
        :param filterSize: An integer specifying the size of the filter
                            that calculates local standard deviation.
                            Defaults to 3.
        :param threshold: A float determine the minimum foreground
                            standard deviation.
                            Defaults to 3.0.
        :param targetSize: A pair of integers that specifies the width and height of the
                            registered image and mask before down sampling if has.
                            When omitted, no resizing will be performed.
                            The input style should be like WIDTH,HEIGHT
        :param downSampleFactor: An integer that specifies the side length of the local
                                    patch that is used to pool the image and mask.
                                    Default as 1, which means no down sampling will be performed.
        :param numberOfGlobalKernels: The number of Gaussian mixture kernels
                                        for each iteration during expectation maximization.
                                        Note that if the number of kernels exceeds
                                        that of the sample pixels, the latter will
                                        take the place.
        :param limitOfLocalKernels: The upper limit of the number of
                                    Gaussian mixture kernels for the
                                    adaptive Bayesian algorithm. Note
                                    that if the limit exceeds that of
                                    the sample pixels, the latter will
                                    take the place.
        :param noRotation: Skip rotation, which can save some computational resource
                            when rotation is already performed.
        :param noIntensityRescale: Leave the intensity distribution as original.
        :param noRectification: Color image will be converted to grayscale using the
                                default algorithm in OpenCV.
        :param grayscale: Read the input image as grayscale, which can
                            save some computational resource.
        :param saveImage: Generate and save the image for each step.
                            By default, no images are generated to
                            save computational resources.
        :param noRemoveBackground: In the extraction step, the background of the image
                                    will not be turned to white.
        :param noExtractionImage: When specified, the image from the extraction
                                    step will not be saved. It only works when
                                    saveImage is specified.
        :param noRegistrationImage: When specified, the image from the registration
                                    step will not be saved. It only works when
                                    saveImage is specified.
        :param noGlobalGMMImage: When specified, the image from the global GMM
                                    step will not be saved. It only works when
                                    saveImage is specified.
        :param noLocalGMMImage:When specified, the image from the local GMM
                                    step will not be saved. It only works when
                                    saveImage is specified.
        """
        input_image = cv2.imread(
            inputImage, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        )
        if input_image is None:
            raise IOError("Image reading failure.")
        if outputDirectory is None:
            outputDirectory = os.path.dirname(inputImage)
        elif not os.path.isdir(outputDirectory):
            raise IsADirectoryError("Invalid output directory.")
        if name is None:
            name = os.path.basename(inputImage).split(".")[0]
        extract_image, mask = algorithm.extract(
            input_image, filterSize, threshold, noRemoveBackground
        )
        register_image, mask = algorithm.register(
            extract_image if extract_image is not None else input_image,
            mask,
            targetSize,
            not noRotation,
            not noIntensityRescale,
            not noRectification,
        )
        global_image, global_mask, global_label, global_levels = algorithm.global_gmm(
            register_image, mask, downSampleFactor, numberOfGlobalKernels, not saveImage
        )
        local_image, local_label, local_levels = algorithm.local_gmm(
            global_label, global_levels, limitOfLocalKernels, register_image
        )
        if saveImage:
            if (
                not (noRemoveBackground or noExtractionImage)
                and not cv2.imwrite(
                    os.path.join(outputDirectory, name + "_extract_image.bmp"),
                    extract_image,
                )
                or not noRegistrationImage
                and not cv2.imwrite(
                    os.path.join(outputDirectory, name + "_register_image.bmp"),
                    register_image,
                )
                or not noGlobalGMMImage
                and not cv2.imwrite(
                    os.path.join(outputDirectory, name + "_globalGMM_image.bmp"),
                    global_image,
                )
                or not noLocalGMMImage
                and not cv2.imwrite(
                    os.path.join(outputDirectory, name + "_localGMM_image.bmp"),
                    local_image,
                )
            ):
                raise IOError("Image writing failure.")
        if (
            not cv2.imwrite(
                os.path.join(outputDirectory, name + "_mask.bmp"), global_mask
            )
            or not cv2.imwrite(
                os.path.join(outputDirectory, name + "_globalGMM_label.bmp"),
                global_label,
            )
            or not cv2.imwrite(
                os.path.join(outputDirectory, name + "_localGMM_label.bmp"), local_label
            )
        ):
            raise IOError("Mask or label writing failure.")
        with open(
            os.path.join(outputDirectory, name + "_globalGMM_levels.txt"), "w"
        ) as f:
            f.write("\n".join(str(i) for i in global_levels))
        with open(
            os.path.join(outputDirectory, name + "_localGMM_levels.txt"), "w"
        ) as f:
            f.write("\n".join(str(i) for i in local_levels))

    @classmethod
    def score(
        cls,
        *inputPrefixes,
        maskSuffix="_mask.bmp",
        globalLabelSuffix="_globalGMM_label.bmp",
        localLabelSuffix="_localGMM_label.bmp",
        localLevelsSuffix="_localGMM_levels.txt",
        outputTablePath=None,
        reference=None,
        globalScoreCutoff=0.0,
        sparseCutoff=None,
        noFlipping=False
    ):
        """
        Give scores of processed image labels.

        | Comparison is done based on globalGMM and localGMM segmentation results.
        | For globalGMM, the normalized mutual information score is calculated between 2 images.
        | For localGMM, the score between 2 images sums up best match blob scores.
        | The blob score considers both intensity difference and overlap between 2 blobs from 2 images.
        | Finally, a hybrid score is calculated as the product of the global and local score.

        :param inputPrefixes: A series of file path prefixes, which
                            includes the directory and the basename to the
                            input labels and levels separated by whitespace.
                            Files for each sample should include a global
                            label file, a local label file and a local levels
                            file.
        :param maskSuffix: The suffix for all input masks. The full path
                            concatenates the inputPrefix and the maskSuffix.
        :param globalLabelSuffix: The suffix for all input global GMM label images.
                                    The full path concatenates the inputPrefix
                                    and the globalSuffix.
        :param localLabelSuffix: The suffix for all input local GMM label images.
                                    The full path concatenates the inputPrefix
                                    and the localSuffix.
        :param localLevelsSuffix: The suffix for all input local GMM levels files.
                                    The full path concatenates the inputPrefix
                                    and the levelSuffix.
        :param outputTablePath: The path to the table file, a csv or an npz,  to
                                keep the score table. When omitted, no file will
                                be generated, and the result will be printed, in
                                the form of matrix or sparse matrix.
        :param reference: A list of integers that specify the indices of the
                            input images to be treated as references,
                            that is, the comparison will be performed
                            between the reference group and non-reference
                            group. The indices should range from 0 to n-1.
                            When omitted, pairwise comparison will be performed,
                            and a matrix will be generated.
                            The input should be like n0,n1,n2,...
        :param globalScoreCutoff: A float ranging from 0 to 1 that speeds up
                                    scoring by cutting hybrid scores with global
                                    scores smaller than it down to 0, skipping
                                    the local scoring. If smaller than sparseCutoff,
                                    its place will be taken by sparseCutoff.
        :param sparseCutoff: A float ranging from 0 to 1 that cuts all figures
                            smaller than it down to 0 in the score table to make way for
                            sparse matrix generation. When omitted, the output table
                            will not be turned to a sparse one.
        :param noFlipping: By default, the program computes the orientation invariant
                        score for each pairwise comparison. When specified, no
                        flipping will take place and comparisons are based on the
                        original orientation.
        """
        if len(inputPrefixes) == 0:
            fire.Fire(InsituTools, "score -- --help")
            return
        masks = [np.array([])] * len(inputPrefixes)
        global_labels = [np.array([])] * len(inputPrefixes)
        local_labels = [np.array([])] * len(inputPrefixes)
        local_levels_list = [[]] * len(inputPrefixes)
        for i, prefix in zip(range(len(inputPrefixes)), inputPrefixes):
            masks[i] = cv2.imread(prefix + maskSuffix, cv2.IMREAD_GRAYSCALE)
            global_labels[i] = cv2.imread(
                prefix + globalLabelSuffix, cv2.IMREAD_GRAYSCALE
            )
            local_labels[i] = cv2.imread(
                prefix + localLabelSuffix, cv2.IMREAD_GRAYSCALE
            )
            if masks[i] is None or global_labels[i] is None or local_labels[i] is None:
                raise IOError("Mask or label reading failure.")
            with open(prefix + localLevelsSuffix) as f:
                local_levels_list[i] = [float(i) for i in f.readlines()]
        n_row = len(inputPrefixes) if reference is None else len(reference)
        n_col = len(inputPrefixes)
        if sparseCutoff is None:
            score_table = np.zeros((n_row, n_col))
        else:
            score_table = dok_matrix((n_row, n_col))
            if sparseCutoff > globalScoreCutoff:
                globalScoreCutoff = sparseCutoff
        score_table = algorithm.score(
            score_table,
            masks,
            global_labels,
            local_labels,
            local_levels_list,
            reference,
            globalScoreCutoff,
            not noFlipping,
        )
        if sparseCutoff is None:
            samples = [os.path.basename(i) for i in inputPrefixes]
            ref_samples = (
                samples if reference is None else [samples[i] for i in reference]
            )
            score_table = pd.DataFrame(score_table, index=ref_samples, columns=samples)
        else:
            score_table = score_table.tolil()
            score_table[score_table < sparseCutoff] = 0
        if outputTablePath is None:
            return score_table
        else:
            if sparseCutoff is None:
                score_table.to_csv(outputTablePath)
            else:
                save_npz(outputTablePath, score_table)


def main():
    fire.Fire(InsituTools)
