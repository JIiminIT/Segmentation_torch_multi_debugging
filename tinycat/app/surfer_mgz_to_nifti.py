import argparse
import os
import time
from glob import glob

import nibabel as nib


def convert_surfer_mgz_to_nifti(root_dir):
    """
    freesurfer mgz to nifti
    :param str root_dir: root directory of surfer_result
    """
    surfer_subjects = glob(os.path.join(root_dir, "surfer_result/*.nii*"))

    for _, subject in enumerate(surfer_subjects):
        tic = time.time()
        subject_name = os.path.basename(subject).split(".nii")[0]
        print("Subject name:", subject_name)

        aseg_filename = os.path.join(subject, "orig.mgz")
        output_filename = root_dir + subject_name + ".nii.gz"

        try:
            aseg_mgz = nib.load(aseg_filename)
        except FileNotFoundError:
            print("Exception :{}".format(aseg_filename))
            continue

        aseg_mgz = nib.Nifti1Image(
            aseg_mgz.get_data(), aseg_mgz.affine, header=aseg_mgz.header
        )
        aseg_mgz.to_filename(output_filename)

        print("\n{}secs elapsed \n{}".format(time.time() - tic, "-" * 70))
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", required=True, type=str, help="root directory of surfer_result"
    )
    args = parser.parse_args()
    convert_surfer_mgz_to_nifti(args.root_dir)


if __name__ == "__main__":
    main()
