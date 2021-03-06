"""Tinycat commandline application interface

Raises:
    ImportError: raises when app has not found
"""

import subprocess
import sys

import tinycat as cat
from tinycat.app import apps


def main():
    print(cat.__doc__)
    print("Supported apps: %s" % apps)
    print("Version: %s" % cat.__version__)

    if len(sys.argv) > 2:
        cmd = "%s %s" % (sys.argv[1], " ".join(sys.argv[2:]))
        print("calling %s" % cmd)

        subprocess.call(cmd, shell=True)

    # data_path = r'D:\Neurophet_work\SourceCode\Registration\inference_data\hist_train'
    # # histogram_ref_file = r'D:\Neurophet_work\SourceCode\Registration\histogram_mni_model.txt'
    #
    # histogram_normaliser = cat.normalization.HistogramNormalisationLayer(
    #     data_path=data_path,
    #     modality='T1',
    #     model_filename=None,
    #     binary_masking_func=cat.binary_masking.BinaryMaskingLayer(),
    #     norm_type='percentile',
    #     cutoff=(0.01, 0.99),
    # )
    #
    # image_list = histogram_normaliser.image_list_from_path()
    # histogram_normaliser.train(image_list=image_list)
    #
    # img = cat.load(r'D:\Neurophet_work\Dataset\seg_104_eval_data\seg_aqua\1155642_T1.nii.gz')
    # # histogram_normaliser.train()
    # norm_image, mask = histogram_normaliser.layer_op(img)
    #
    # import matplotlib.pyplot as plt
    #
    # img = img.get_data()
    # plt.subplot(1, 3, 1)
    # plt.imshow(img[:, :, img.shape[2]//2], 'gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(norm_image[:, :, norm_image.shape[2]//2, 0], 'gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(mask[:, :, mask.shape[2]//2, 0], 'gray')
    # plt.show()


if __name__ == "__main__":
    main()
