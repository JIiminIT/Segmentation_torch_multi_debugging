import tinycat as cat
import numpy as np
import os


ASEG_FILENAME = "aparc+aseg.mgz"
WMPARC_FILENAME = "wmparc.mgz"


def convert_aseg_to_9_v1_and_save(directory="./", result_filename="converted.nii.gz"):
    aparc_aseg = cat.load(os.path.join(directory, ASEG_FILENAME))
    converted = convert_aseg_to_9_v1(aparc_aseg.get_data())
    cat.Nifti1Image(converted, aparc_aseg.affine, header=aparc_aseg.header).to_filename(
        result_filename
    )


def convert_aseg_to_9_v2_and_save(directory="./", result_filename="converted.nii.gz"):
    aparc_aseg = cat.load(os.path.join(directory, ASEG_FILENAME))
    converted = convert_aseg_to_9_v2(aparc_aseg.get_data())
    cat.Nifti1Image(converted, aparc_aseg.affine, header=aparc_aseg.header).to_filename(
        result_filename
    )

def convert_aseg_to_104_v1_and_save(directory="./", result_filename="converted.nii.gz"):
    aparc_aseg = cat.load(os.path.join(directory, ASEG_FILENAME))
    converted = convert_aseg_to_104_v1(aparc_aseg.get_data())
    cat.Nifti1Image(converted, aparc_aseg.affine, header=aparc_aseg.header).to_filename(
        result_filename
    )


def convert_aseg_to_9_v1(aseg):
    """Convert aparc+aseg.mgz into 9-label segmentation labels
    
    Args:
        aseg (np.ndarray): array of aseg
    
    Returns:
        np.array: converted labels
    """

    empty_aseg = np.zeros_like(aseg)

    # Remove optic-chiasm
    aseg[aseg == 85] = 0

    # Extract gray matter
    cerebral_gm = np.copy(empty_aseg)

    # Aparc cortex labels
    cerebral_gm[aseg >= 1000] = 1

    # Non-wm hypointensities
    cerebral_gm[aseg == 80] = 1
    cerebral_gm[aseg == 81] = 1
    cerebral_gm[aseg == 82] = 1

    # Thalamus
    cerebral_gm[aseg == 9] = 1
    cerebral_gm[aseg == 10] = 1
    cerebral_gm[aseg == 48] = 1
    cerebral_gm[aseg == 49] = 1

    # Hippocampus
    cerebral_gm[aseg == 17] = 1
    cerebral_gm[aseg == 53] = 1

    # Amygdala
    cerebral_gm[aseg == 18] = 1
    cerebral_gm[aseg == 54] = 1

    # Create ventricles
    ventricles = np.copy(empty_aseg)
    ventricles[aseg == 4] = 1
    ventricles[aseg == 5] = 1
    ventricles[aseg == 14] = 1
    ventricles[aseg == 15] = 1
    ventricles[aseg == 31] = 1  # Left choroid plexus
    ventricles[aseg == 43] = 1
    ventricles[aseg == 44] = 1
    ventricles[aseg == 63] = 1  # Right choroid plexus
    ventricles[aseg == 72] = 1
    ventricles[aseg == 213] = 1
    ventricles[aseg == 221] = 1

    # Grey matter is distributed at the surface of the cerebral hemispheres (cerebral cortex)
    # and of the cerebellum (cerebellar cortex), as well as in the depths of the
    # cerebrum (thalamus; hypothalamus; subthalamus, basal ganglia – putamen, globus pallidus, nucleus accumbens; septal nuclei),
    # cerebellar (deep cerebellar nuclei – dentate nucleus, globose nucleus, emboliform nucleus, fastigial nucleus),
    # brainstem (substantia nigra, red nucleus, olivary nuclei, cranial nerve nuclei).

    # Cerebellar gm
    cerebellar_gm = np.copy(empty_aseg)
    cerebellar_gm[aseg == 8] = 1
    cerebellar_gm[aseg == 47] = 1

    # Cerebellar wm
    cerebellar_wm = np.copy(empty_aseg)
    cerebellar_wm[aseg == 7] = 1
    cerebellar_wm[aseg == 46] = 1

    # CSF
    csf = np.copy(empty_aseg)
    csf[aseg == 24] = 1

    # Extract white matter
    cerebral_wm = np.copy(empty_aseg)
    cerebral_wm[(aseg > 0) & (aseg < 1000)] = 1

    empty_aseg[cerebral_wm > 0] = 2
    empty_aseg[cerebral_gm > 0] = 1
    empty_aseg[cerebellar_gm > 0] = 3
    empty_aseg[cerebellar_wm > 0] = 4
    empty_aseg[ventricles > 0] = 5
    empty_aseg[csf > 0] = 6

    return empty_aseg


def convert_aseg_to_9_v2(aseg):
    empty_aseg = np.zeros_like(aseg)

    # Remove optic-chiasm
    aseg[aseg == 85] = 0

    # Extract gray matter
    cerebral_gm = np.copy(empty_aseg)

    # Aparc cortex labels
    cerebral_gm[aseg >= 1000] = 1

    # Non-wm hypointensities
    cerebral_gm[aseg == 80] = 1
    cerebral_gm[aseg == 81] = 1
    cerebral_gm[aseg == 82] = 1

    # Thalamus
    cerebral_gm[aseg == 9] = 1
    cerebral_gm[aseg == 10] = 1
    cerebral_gm[aseg == 48] = 1
    cerebral_gm[aseg == 49] = 1

    # Hippocampus
    cerebral_gm[aseg == 17] = 1
    cerebral_gm[aseg == 53] = 1

    # Amygdala
    cerebral_gm[aseg == 18] = 1
    cerebral_gm[aseg == 54] = 1

    # Create ventricles
    ventricles = np.copy(empty_aseg)
    ventricles[aseg == 4] = 1
    ventricles[aseg == 5] = 1
    ventricles[aseg == 14] = 1
    ventricles[aseg == 15] = 1
    # ventricles[aseg == 31] = 1  # Left choroid plexus
    ventricles[aseg == 43] = 1
    ventricles[aseg == 44] = 1
    # ventricles[aseg == 63] = 1  # Right choroid plexus
    ventricles[aseg == 72] = 1
    ventricles[aseg == 213] = 1
    ventricles[aseg == 221] = 1

    # Grey matter is distributed at the surface of the cerebral hemispheres (cerebral cortex)
    # and of the cerebellum (cerebellar cortex), as well as in the depths of the
    # cerebrum (thalamus; hypothalamus; subthalamus, basal ganglia – putamen, globus pallidus, nucleus accumbens; septal nuclei),
    # cerebellar (deep cerebellar nuclei – dentate nucleus, globose nucleus, emboliform nucleus, fastigial nucleus),
    # brainstem (substantia nigra, red nucleus, olivary nuclei, cranial nerve nuclei).

    # Cerebellar gm
    cerebellar_gm = np.copy(empty_aseg)
    cerebellar_gm[aseg == 8] = 1
    cerebellar_gm[aseg == 47] = 1

    # Cerebellar wm
    cerebellar_wm = np.copy(empty_aseg)
    cerebellar_wm[aseg == 7] = 1
    cerebellar_wm[aseg == 46] = 1

    # CSF
    csf = np.copy(empty_aseg)
    csf[aseg == 24] = 1

    # Extract white matter
    cerebral_wm = np.copy(empty_aseg)
    cerebral_wm[(aseg > 0) & (aseg < 1000)] = 1

    empty_aseg[cerebral_wm > 0] = 2
    empty_aseg[cerebral_gm > 0] = 1

    SUBCORT_LABELS = (10, 28, 31, 49, 60, 63)
    for label in SUBCORT_LABELS:
        empty_aseg[aseg == label] = 2

    empty_aseg[cerebellar_gm > 0] = 3
    empty_aseg[cerebellar_wm > 0] = 4
    empty_aseg[ventricles > 0] = 5
    empty_aseg[csf > 0] = 6

    return empty_aseg

def convert_aseg_to_104_v1(aseg):
    """Convert aparc+aseg.mgz into 104-label segmentation labels
    
    Args:
        aseg (np.ndarray): array of aseg
    
    Returns:
        np.array: converted labels
    """
    empty_aseg = np.zeros_like(aseg)
    for label_target, label_source in enumerate(cat.lut.AQUA_LABEL_V2):
        empty_aseg[aseg == label_source] = label_target
    return empty_aseg

