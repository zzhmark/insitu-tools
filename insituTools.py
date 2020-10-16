import os
import algorithm
import fire
import cv2


class InsituTools(object):

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
        Usage: insituTools extract <inputImgae> <outputMask> [options]
        :param inputImage: The path to the input image,
                            which is loaded as a color image.
        :param outputMask: The path to the output mask.
        :param outputImage: The path to the output image.
                            No image will be generated if omitted.
        :param filterSize: An integer specifying the size of the filter
                            that calculates local standard deviation.
                            Default to 3.
        :param threshold: A float determine the minimum foreground
                            standard deviation.
                            Default to 3.0.
        :param grayscale: Whether the input image is grayscale.
                            Default to False.
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
        Usage: insituTools register <inputImage> <inputMask> <outputImage> <outputMask>
        :param inputImage:
        :param inputMask:
        :param outputImage:
        :param outputMask:
        :param targetSize:
        :param downSampleFactor:
        :param noRotation:
        :param noIntensityRescale:
        :param noRectification:
        :param grayscale:
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
            input_image,
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
            saveImage=False
    ):
        """
        :param input_image:
        :param outputDirectory:
        :param name:
        :param filterSize:
        :param threshold:
        :param targetSize:
        :param downSampleFactor:
        :param numberOfGlobalKernels:
        :param limitOfLocalKernels:
        :param noRotation:
        :param noIntensityRescale:
        :param noRectification:
        :param grayscale:
        :param saveImage:
        :return:
        """
        input_image = cv2.imread(
            input_image,
            cv2.IMREAD_GRAYSCALE if grayscale
            else cv2.IMREAD_COLOR
        )
        if input_image is None:
            raise IOError
        if outputDirectory is None:
            outputDirectory = os.path.dirname(input_image)
        elif not os.path.isdir(outputDirectory):
            raise IsADirectoryError
        if name is None:
            name = os.path.basename(input_image).split('.')[0]
        extract_image, mask = algorithm.extract(
            input_image,
            filterSize,
            threshold,
            True
        )
        register_image, mask = algorithm.register(
            extract_image if extract_image is not None
            else input_image,
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
                    input_image
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
            )
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
            cls
    ):
        pass


if __name__ == '__main__':
    fire.Fire(InsituTools)
