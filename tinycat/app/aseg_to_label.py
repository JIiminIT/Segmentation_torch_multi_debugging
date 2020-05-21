import os
import time
import argparse
from glob import glob

import numpy as np
import nibabel as nib
from tinycat.lut import AQUA_LABEL_V1, LEFT_LABELS_v1

def convert_aseg_to_label(root_dir, mode="aqua"):
    """Converts aseg.mgz to predefined label format
    aparc+aseg label table

        1000 ~     : left cerebral cortex
        2000 ~     : right cerebral cortex
        1-999      : subcortical regions, cerebellum

    teslab label version
    - v1 : 5 Labels defined from cortex / subcortex

    aqua label version
    - v1 : 112 Labels from aparc+aseg.mgz
    - v2 : 104 Labels from aparc+aseg.mgz

    Args:
        root_dir (str): root directory of surfer_result
        mode (str, optional): label mode between 'aqua'
            and 'teslab'. Defaults to 'aqua'.
    """
    surfer_subjects = glob(os.path.join(root_dir, "surfer_result/*.nii*"))

    for _, subject in enumerate(surfer_subjects):
        tic = time.time()
        subject_name = os.path.basename(subject).split(".nii")[0]
        print("Subject name:", subject_name)

        aseg_filename = os.path.join(subject, "aparc+aseg.mgz")
        output_filename = root_dir + subject_name + "_104_Label.nii.gz"
        if os.path.exists(output_filename):
            print("{} already exists, skipping...".format(aseg_filename))
            continue

        try:
            aseg_mgz = nib.load(aseg_filename)
        except FileNotFoundError:
            print("Exception :{}".format(aseg_filename))
            continue

        aseg = aseg_mgz.get_data().astype(np.uint32)

        output_label = np.zeros_like(aseg)

        if mode == "aqua":
            for idx, label in enumerate(AQUA_LABEL_V1):
                output_label[aseg == label] = idx

        if mode == "teslab":
            output_label[(aseg >= 1000).nonzero()] = 1
            output_label[(aseg < 1000).nonzero()] = 2
            output_label[(aseg == 6).nonzero()] = 3
            output_label[(aseg == 8).nonzero()] = 3
            output_label[(aseg == 45).nonzero()] = 3
            output_label[(aseg == 47).nonzero()] = 3
            output_label[(aseg == 7).nonzero()] = 4
            output_label[(aseg == 46).nonzero()] = 4
            output_label[(aseg == 4).nonzero()] = 5
            output_label[(aseg == 31).nonzero()] = 5
            output_label[(aseg == 43).nonzero()] = 5
            output_label[(aseg == 63).nonzero()] = 5
            output_label[(aseg == 0).nonzero()] = 0

        output = nib.Nifti1Image(output_label, aseg_mgz.affine, header=aseg_mgz.header)
        output.to_filename(output_filename)

        print("\n{}secs elapsed \n{}".format(time.time() - tic, "-" * 70))
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", required=True, type=str, help="root directory of surfer_result"
    )
    parser.add_argument(
        "--mode",
        required=False,
        type=str,
        help="aqua(104 Label brain) or teslab(5 Label brain)",
        default="aqua",
    )
    args = parser.parse_args()
    convert_aseg_to_label(args.root_dir, args.mode)


if __name__ == "__main__":
    main()
