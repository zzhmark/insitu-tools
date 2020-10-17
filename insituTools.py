import os
import algorithm
import fire
import cv2


class InsituTools(object):
    """
    In Situ expression patterns analysis toolset.

    | CLI that implements Peng's algorithm that detects and compares
    | patterns in In Situ expression image of Drosophila embryos. It
    | also has the potential of being applied to other kinds of images.
    """

    @classmethod
    def extract(
            cls,
            inputImage,
            outputMask,
            outputImage=None,
            filterSize=3,
            threshold=3.0,
            grayscale=False
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
            inputImage,
            cv2.IMREAD_GRAYSCALE if grayscale
            else cv2.IMREAD_COLOR
        )
        if image is None:
            raise IOError
        image, mask = algorithm.extract(
            image,
            filterSize,
            threshold,
            outputImage is None
        )
        if outputImage is not None:
            if not cv2.imwrite(outputImage, image):
                raise IOError
        if not cv2.imwrite(outputMask, mask):
            raise IOError

    @classmethod
    def register(
            cls,
            inputImage,
            inputMask,
            outputImage,
            outputMask,
            targetSize=None,
            downSampleFactor=None,
            noRotation=False,
            noIntensityRescale=False,
            noRectification=False,
            grayscale=False
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
                            The tuple should be like (WIDTH, HEIGHT)
        :param downSampleFactor: An integer that specifies the side length of the local
                                    patch that is used to pool the image and mask.
                                    When omitted, no down sampling will be performed.
        :param noRotation: Skip rotation, which can save some computational resource
                            when rotation is already performed.
        :param noIntensityRescale: Leave the intensity distribution as original.
        :param noRectification: Color image will be converted to grayscale using the
                                default algorithm in OpenCV.
        :param grayscale: Read the input image as grayscale, which can
                            save some computational resource.
        """
        image = cv2.imread(
            inputImage,
            cv2.IMREAD_GRAYSCALE if grayscale
            else cv2.IMREAD_COLOR
        )
        mask = cv2.imread(inputMask, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise IOError
        image, mask = algorithm.register(
            image,
            mask,
            targetSize,
            downSampleFactor,
            not noRotation,
            not noIntensityRescale,
            not noRectification
        )
        if not cv2.imwrite(outputImage, image) or \
                not cv2.imwrite(outputMask, mask):
            raise IOError

    @classmethod
    def globalGMM(
            cls,
            inputImage,
            inputMask,
            outputLabel,
            outputLevels,
            outputImage=None,
            numberOfGlobalKernels=5
    ):
        """
        Global Gaussian mixture model fitting.

        | Apply the GMM decomposition globally to segment
        | the foreground. It uses EM method to adaptively
        | cluster the pixels based on their intensity.

        :param inputImage: The path to the input image，
                            read as grayscale.
        :param inputMask: The path to the input mask.
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
        """
        image = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(inputMask, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise IOError
        image, label, levels = algorithm.global_gmm(
            image,
            mask,
            numberOfGlobalKernels,
            outputImage is None
        )
        if image is not None:
            if not cv2.imwrite(outputLabel, image):
                raise IOError
        if not cv2.imwrite(outputLabel, label):
            raise IOError
        with open(outputLevels) as f:
            f.write('\n'.join(str(i) for i in levels))

    @classmethod
    def localGMM(
            cls,
            inputImage,
            inputLabel,
            inputLevels,
            outputLabel,
            outputLevels,
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
        if image is None or label is None:
            raise IOError
        with open(inputLevels) as f:
            levels = [float(i) for i in f.readlines()]
        image, label, levels = algorithm.local_gmm(
            image,
            label,
            levels,
            limitOfLocalKernels,
            outputImage is None
        )
        if image is not None:
            if not cv2.imwrite(outputImage, image):
                raise IOError
        if not cv2.imwrite(outputLabel, label):
            raise IOError
        with open(outputLevels) as f:
            f.write('\n'.join(str(i) for i in levels))

    @classmethod
    def recognizePatterns(
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
            noRemoveBackGround=False
    ):
        """
        Recognize patterns from the raw image.

        | This command merges all image processing steps.
        
        :param inputImage: The path to the input image.
        :param outputDirectory: The path to the directory to store all
                                the output files. Duplicated files will
                                be overwritten.
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
                                    When omitted, no down sampling will be performed.
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
        :param noRemoveBackGround: In the extraction step, the background of the image
                                    will not be turned to white.
        """
        inputImage = cv2.imread(
            inputImage,
            cv2.IMREAD_GRAYSCALE if grayscale
            else cv2.IMREAD_COLOR
        )
        if inputImage is None:
            raise IOError
        if outputDirectory is None:
            outputDirectory = os.path.dirname(inputImage)
        elif not os.path.isdir(outputDirectory):
            raise IsADirectoryError
        if name is None:
            name = os.path.basename(inputImage).split('.')[0]
        extract_image, mask = algorithm.extract(
            inputImage,
            filterSize,
            threshold,
            noRemoveBackGround
        )
        register_image, mask = algorithm.register(
            extract_image if extract_image is not None
            else inputImage,
            mask,
            targetSize,
            downSampleFactor,
            not noRotation,
            not noIntensityRescale,
            not noRectification
        )
        global_image, global_label, global_levels = \
            algorithm.global_gmm(
                register_image,
                mask,
                numberOfGlobalKernels,
                not saveImage
            )
        local_image, local_label, local_levels = \
            algorithm.local_gmm(
                register_image,
                global_label,
                global_levels,
                limitOfLocalKernels,
                not saveImage
            )
        if saveImage:
            if not cv2.imwrite(
                    os.path.join(
                        outputDirectory,
                        name + '_extract_image.bmp'
                    ),
                    inputImage
            ) or not cv2.imwrite(
                os.path.join(
                    outputDirectory,
                    name + '_register_image.bmp'
                ),
                register_image
            ) or not cv2.imwrite(
                os.path.join(
                    outputDirectory,
                    name + '_globalGMM_image.bmp'
                ),
                global_image
            ) or not cv2.imwrite(
                os.path.join(
                    outputDirectory,
                    name + '_localGMM_image.bmp'
                ),
                local_image
            ):
                raise IOError
        if not cv2.imwrite(
                os.path.join(
                    outputDirectory,
                    name + '_extract_mask.bmp'
                ),
                mask
        ) or not cv2.imwrite(
            os.path.join(
                outputDirectory,
                name + '_register_mask.bmp'
            ),
            mask
        ) or not cv2.imwrite(
            os.path.join(
                outputDirectory,
                name + '_globalGMM_label.bmp'
            ),
            global_label
        ) or not cv2.imwrite(
            os.path.join(
                outputDirectory,
                name + '_localGMM_label.bmp'
            ),
            local_label
        ):
            raise IOError
        with open(
                os.path.join(
                    outputDirectory,
                    name + '_globalGMM_levels.txt'
                ),
                'w'
        ) as f:
            f.write('\n'.join(str(i) for i in global_levels))
        with open(
                os.path.join(
                    outputDirectory,
                    name + '_localGMM_levels.txt'
                ),
                'w'
        ) as f:
            f.write('\n'.join(str(i) for i in local_levels))


    @classmethod
    def score(
            cls,
            *inputImages,
            outputCSV=None,
            reference=None,
            silent=False
    ):
        """
        Give scores of processed image labels.

        | Comparison is done based on globalGMM and
        localGMM segmentation results.
        | For globalGMM, the normalized mutual information
        score is calculated between 2 images.
        | For localGMM, the score between 2 images sums up
        best match blob scores.
        | The blob score considers both intensity difference
        and overlap between 2 blobs from 2 images.
        | Finally, a hybrid score is calculated as the product
        of the global and local score.

        :param inputImages: A series of file path to the input images
                            separated by whitespace.
        :param outputCSV: The path to the csv file to keep the score table.
                            When omitted, no file will be generated.
        :param reference: A list of integers that specify the indices of the
                            input images to be treated as references,
                            that is, the comparison will be performed
                            between the reference group and non-reference
                            group. The indices should range from 0 to n-1.
                            When omitted, pairwise comparison will be performed,
                            and a matrix will be generated.
                            The input should be like n0,n1,n2,...
        :param silent: No standard IO output, otherwise the scores
                        will be printed. But note that when outputCSV
                        is omitted, scores will be printed anyway.
        """
        pass


if __name__ == '__main__':
    fire.Fire(InsituTools)
