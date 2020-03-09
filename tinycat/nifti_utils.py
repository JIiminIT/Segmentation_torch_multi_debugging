"""tinycat pre-defined utilities"""
import nibabel as nib
import tinycat as cat
import glob
import os

__all__ = [
    "to_nifti",
    "items",
    "load",
    "load_image_3d",
    "load_sample_mri",
    "Nifti1Image",
    "get_nifti_items",
]


class Nifti1Image(nib.Nifti1Image):
    """class for wrapping Nifti1Image"""

    def save(self, filename, dataobj=None, affine=None, header=None):
        if dataobj is None:
            dataobj = self.dataobj
        if affine is None:
            affine = self.affine
        if header is None:
            header = self.header
        cat.Nifti1Image(dataobj, affine, header).to_filename(filename)


def load(filename):
    """nibabel load wrapper with more attributes

    Args:
        filename (str): filename of the nifti image
    """
    proxy = nib.load(filename)

    if isinstance(proxy, nib.Nifti1Image):
        dataobj = proxy.dataobj
        header = proxy.header
        affine = proxy.affine
        proxy = Nifti1Image(dataobj, affine, header)

    proxy.axcodes = nib.aff2axcodes(proxy.affine)
    return proxy


def load_image_3d(filename):
    """load n-dimensional image and convert it into 3d array

    Args:
        filename (str): filename of the image

    Returns:
        cat.Nifti1Image: converted 3d image
    """

    image = load(filename)

    if len(image.shape) > 3:
        dataobj = image.dataobj

        while len(dataobj.shape) > 3:
            if dataobj.shape[0] == 1:
                dataobj = dataobj[0]

            if dataobj.shape[-1] == 1:
                dataobj = dataobj[..., 0]

        image = Nifti1Image(dataobj=dataobj, affine=image.affine, header=image.header)

    return image


def load_sample_mri():
    """load sample defaced mri

    Returns:
        Nifti1Image: example mri object
    """

    try:
        filepath = __file__
    except NameError:
        filepath = "tinycat/"

    example_mri_filename = os.path.join(
        os.path.dirname(filepath), "core/impl/resources/example_mri.nii.gz"
    )

    return load(example_mri_filename)


def to_nifti(filename, compress=True):
    """convert and save non-nifti nibabel instance into nifti format

    Args:
        filename (str): filename of mri to be converted
        compress (bool, optional): If true, save nifti into gz compressed format ('nii.gz')
    """

    image = nib.load(filename)
    image = nib.Nifti1Image(image.get_data(), image.affine, header=image.header)

    if compress:
        extension = ".nii.gz"
    else:
        extension = ".nii"

    subject = ".".join(filename.split(".")[:-1])
    image.to_filename(subject + extension)


def items(dirname="./", subfix="*nii**", recursive=True):
    """ Simple wrapper for glob.glob to get list of files with subfix

    Args:
        dirname (str, optional): Defaults to './'.
        subfix (str, optional): Defaults to '*nii*'.
        recursive (bool, optional): Defaults to True.

    Returns:
        glob.glob (list)
    """
    return glob.glob(os.path.join(dirname, subfix), recursive=recursive)


def get_nifti_items(dirname="./", subfix="*nii**", recursive=True):
    """ Get every nifti1 items into iterable nibabel object

    Returns:
        list: list of nifti items
    """

    filenames = items(dirname, subfix=subfix, recursive=recursive)
    nifti_items = []

    for filename in filenames:
        nifti_items.append(load(filename))

    return nifti_items
