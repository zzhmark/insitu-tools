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
            threshold=3.0
    ):
        """
        :param inputImage: The path to the input image,
                            which is loaded as a color image.
        :param outputMask: The path to the output mask.
        :param outputImage: The path to the output image.\n
                            No image will be generated if omitted.
        :param filterSize: An integer specifying the size of the filter
                            that calculates local standard deviation.\n
                            Default to 3.
        :param threshold:   A float determine the minimum foreground
                            standard deviation.\n
                            Default to 3.0
        Usage: insituTools extract [options]
        """
        image, mask = algorithm.extract(
            cv2.imread(inputImage, cv2.IMREAD_COLOR),
            (filterSize, filterSize),
            threshold,
            outputImage is None
        )
        if image is not None:
            cv2.imwrite(outputImage, image)
        cv2.imwrite(outputMask, mask)

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
            noRectification=False
    ):
        """
        :param inputImage:
        :param inputMask:
        :param outputImage:
        :param outputMask:
        :param targetSize:
        :param downSampleFactor:
        :param noRotation:
        :param noIntensityRescale:
        :param noRectification:
        """
        image, mask = algorithm.register(
            cv2.imread(inputImage, cv2.IMREAD_COLOR),
            cv2.imread(inputMask, cv2.IMREAD_GRAYSCALE),
            targetSize,
            (downSampleFactor, downSampleFactor),
            not noRotation,
            not noIntensityRescale,
            not noRectification
        )
        cv2.imwrite(outputImage, image)
        cv2.imwrite(outputMask, mask)

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
        image, label, levels = algorithm.global_gmm(
            cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE),
            cv2.imread(inputMask, cv2.IMREAD_GRAYSCALE),
            numberOfGlobalKernels,
            outputImage is None
        )
        if image is not None:
            cv2.imwrite(outputLabel, image)
        cv2.imwrite(outputLabel, label)
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
        with open(inputLevels) as f:
            levels = [float(i) for i in f.readlines()]
            image, label, levels = algorithm.local_gmm(
                cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE),
                cv2.imread(inputLabel, cv2.IMREAD_GRAYSCALE),
                levels,
                limitOfLocalKernels,
                outputImage is None
            )
            if image is not None:
                cv2.imwrite(outputImage, image)
            cv2.imwrite(outputLabel, label)
            cv2.imwrite(outputLevels, levels)

    @classmethod
    def recognizePatterns(
            cls,
            inputImage,
            outputPrefix=None,
            filterSize=3,
            threshold=3.0,
            targetSize=None,
            downSampleFactor=None,
            numberOfGlobalKernels=5,
            limitOfLocalKernels=10,
            noRotation=False,
            noIntensityRescale=False,
            noRectification=False,
            saveImage=False,
    ):
        image, mask = algorithm.extract(
            cv2.imread(inputImage, cv2.IMREAD_COLOR),
            (filterSize, filterSize),
            threshold,
            not saveImage
        )
        if saveImage:
            cv2.imwrite(os.join.path(outputPrefix), image)

    @classmethod
    def score(
            cls
    ):
        pass


if __name__ == '__main__':
    fire.Fire(InsituTools)
