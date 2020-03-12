import os
import warnings
from glob import glob
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import tinycat as cat
from tinycat.evaluation import report_metrics


class Evaluator(object):
    """Segmentation evaluation interface
    Calcuates evaluation metrics with provided prediction directory and label directory
    and saves into result directory in csv format.

    Usage:
        # label filename and prediction filename must be identical.
        evaluate_segmentation -p prediction/directory -l label/directory -r result/directory
    """

    def _parse(self):
        parser = ArgumentParser()
        parser.add_argument(
            "--prediction_directory",
            "-p",
            help="evaluation directory with prediction nifti files",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--label_directory",
            "-l",
            help="evaluation directory with label nifti files",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--result_directory",
            "-r",
            help="result directory to save csv formatted evaluation result",
            default="",
            type=str,
        )
        parser.add_argument(
            "--prediction_subfix",
            help="wildcard subfix to search prediction files (default: *.nii)",
            default="*.nii*",
        )
        parser.add_argument(
            "--label_subfix",
            help="wildcard subfix to search label files (default: *.nii)",
            default="*.nii*",
        )
        parser.add_argument(
            "--filename_prefix", help="result filename prefix if needed", default=""
        )
        parser.add_argument(
            "--metrics",
            help="calculates 'dice', 'jacc', 'accr' if included.",
            default="dice",
        )
        parser.add_argument(
            "--n_classes", help="number of classes", default=10, type=int
        )

        return parser

    def __init__(self):
        parser = self._parse()
        self.args, _ = parser.parse_known_args()

        self.label_filenames = sorted(
            glob(os.path.join(self.args.label_directory, self.args.label_subfix))
        )
        self.prediction_filenames = sorted(
            glob(
                os.path.join(
                    self.args.prediction_directory, self.args.prediction_subfix
                )
            )
        )

        if len(self.label_filenames) != len(self.prediction_filenames):
            warnings.warn(
                "number of label files and prediction files are not identical."
            )

    @staticmethod
    def positional_intersection(string1, string2):
        intersection = ""
        for char1, char2 in zip(string1, string2):
            if char1 == char2:
                intersection += char1
        return intersection

    @staticmethod
    def intersection(string1, string2):
        return "".join(sorted(set(string1) & set(string2), key=string1.index))

    def run(self):
        dataframes = []
        for lbl, pred in tqdm(zip(self.label_filenames, self.prediction_filenames)):
            lbl_basename = os.path.basename(lbl)
            pred_basename = os.path.basename(pred)

            subject_name = self.positional_intersection(lbl_basename, pred_basename)
            if not subject_name:
                subject_name = self.intersection(lbl_basename, pred_basename)
                if not subject_name:
                    continue

            print(" processing:", lbl_basename, pred_basename, subject_name)

            warned = False
            if not warned:
                if lbl_basename != pred_basename:
                    warnings.warn(
                        "Label filename and prediction filename is not identical."
                    )
                    warned = True

            lbl_object = cat.load_image_3d(lbl)
            pred_object = cat.load_image_3d(pred)

            if lbl_object.shape != pred_object.shape:
                warnings.warn(
                    "Label and prediction array shape is not compatibile, resampling..."
                )
                pred_object = cat.nifti.resample_mri(
                    pred_object,
                    dim_init=pred_object.shape,
                    dim_result=lbl_object.shape,
                    template_spacing=lbl_object.header.get_zooms(),
                    spline_order=0,
                )

            lbl_array = lbl_object.get_data()
            pred_array = pred_object.get_data()
            if self.args.metrics == "accuracy":
                lbl_array[lbl_array >= self.args.n_classes] = 0
                pred_array[pred_array >= self.args.n_classes] = 0

            # If binary segmentation, label and prediction result is binarized
            if self.args.n_classes == 2:
                lbl_array = lbl_array > 0
                pred_array = pred_array > 0

                lbl_coord = cat.nifti.get_coordinate_system(lbl_object.affine)
                pred_coord = cat.nifti.get_coordinate_system(pred_object.affine)
                if lbl_coord != pred_coord:
                    warnings.warn(
                        "Converting predicted coordinate system to original label's system"
                    )
                    cat.nifti.coordinate_converter(pred_array, pred_coord, lbl_coord)

            dataframes.append(
                report_metrics(
                    pred_array,
                    lbl_array,
                    lbl_basename,
                    metrics=self.args.metrics,
                    n_classes=self.args.n_classes,
                )
            )

        result = pd.concat(dataframes)
        result.to_csv(
            "%s%s_eval.csv"
            % (self.args.filename_prefix, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        )


def main():
    evaluator = Evaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
