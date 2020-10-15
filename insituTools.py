import os

import algorithm
import fire
import cv2


class insituTools(object):

    def extract(
            self,
            inputImage,
            outputMask,
            outputImage=None,
            kernelSize=3,
            threshold=3.0
    ):
        """
        :param inputImage: The path to the input image,
                            which is loaded as a color image.
        :param outputMask: The path to the output mask.
        :param outputImage: The path to the output image.\n
                            No image will be generated if omitted.
        :param kernelSize: An integer specifying the size of the filter
                            that calculates local standard deviation.\n
                            Default to 3.
        :param threshold:   A float determine the minimum foreground
                            standard deviation.\n
                            Default to 3.0
        Usage: insituTools extract [options]
        """
        output = algorithm.extract(
            cv2.imread(inputImage, cv2.IMREAD_COLOR),
            (kernelSize, kernelSize),
            threshold,
            outputImage is None
        )
        if output[0] is not None:
            cv2.imwrite(outputImage, output[0])
        cv2.imwrite(outputMask, output[1])

    def register(
            self,
            inputImage,
            inputMask,
            outputImage,
            outputMask,
            outputSize=None,
            downSampleFactor=None,
            rotate=True,
            rescale=True,
            rectify=True
    ):
        """
        :param inputImage:
        :param inputMask:
        :param outputImage:
        :param outputMask:
        :param outputSize:
        :param downSampleFactor:
        :param rotate:
        :param rescale:
        :param rectify:
        """
        output = algorithm.register(
            cv2.imread(inputImage, cv2.IMREAD_COLOR),
            cv2.imread(inputMask, cv2.IMREAD_GRAYSCALE),
            outputSize,
            (downSampleFactor, downSampleFactor),
            rotate,
            rescale,
            rectify
        )
        cv2.imwrite(outputImage, output[0])
        cv2.imwrite(outputMask, output[1])

    def globalGMM(
            self,
            inputImage,
            inputMask,
            outputLabel,
            outputLevels,
            outputImage=None,
            numberOfGMMKernels=5
    ):
        output = algorithm.global_gmm(
            cv2.imread(inputImage, cv2.IMREAD_COLOR),
            cv2.imread(inputMask, cv2.IMREAD_GRAYSCALE),
            numberOfGMMKernels,
            outputImage is None
        )
        if outputImage is None:
            cv2.imwrite(outputLabel, output[0])
            with open(outputLevels) as f:
                for i in output[1]:
                    f.write(str(i))
        else:
            cv2.imwrite(outputImage, output[0])
            cv2.imwrite(outputLabel, output[1])
            with open(outputLevels) as f:
                f.write(output[2])

    def localGMM(
            self
    ):
        pass

    def score(
            self
    ):
        pass

if __name__ == '__main__':
    fire.Fire(insituTools)