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

def convert_aseg_to_104_v2_and_save(directory="./", result_filename="converted.nii.gz"):
    aparc_aseg = cat.load(os.path.join(directory, ASEG_FILENAME))
    converted = convert_aseg_to_104_v2(aparc_aseg.get_data())
    cat.Nifti1Image(converted, aparc_aseg.affine, header=aparc_aseg.header).to_filename(
        result_filename
    )

def convert_104_v1_to_104_v2_and_save(aqua_104_v1_filename, result_filename="converted.nii.gz"):
    aqua_104_v1 = cat.load(aqua_104_v1_filename)
    converted = convert_104_v1_to_104_v2(aqua_104_v1.get_data())
    cat.Nifti1Image(converted, aqua_104_v1.affine, header=aqua_104_v1.header).to_filename(
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
    for label_target, label_source in enumerate(cat.lut.AQUA_LABEL_V1):
        empty_aseg[aseg == label_source] = label_target
    return empty_aseg

def label_to_LR(
    img,
    w_size=5,
    pnt=31,
    left=105,
    right=106,
):
    """왼쪽과 오른쪽으로 분리되지 않은 label을 왼쪽과 오른쪽으로 나눕니다.

    Args:
        seg104 (Nifti1Wrapper): desikan+aseg segmentation label
        img_out_dir (str): 계산 과정에서 생성되는 .nii를 저장할 디렉토리 ('.../store/nifti/')
        w_size (int, optional): 고려할 kernel 크기. Defaults to 5.
        pnt (int, optional): 왼쪽 오른쪽으로 나눌 label 번호. Defaults to 31.
        left (int, optional): 왼쪽으로 나눌 label 번호. Defaults to 105.
        right (int, optional): 오른쪽으로 나눌 label 번호. Defaults to 106.

    Returns:
        img_(np.ndarray): 지정한 label을 왼쪽과 오른쪽으로 분리한 array
    """

    # pnt번 레이블을 left, right 레이블로 분할

    coo_i, coo_j, coo_k = np.where(img == pnt)
    for i, j, k in zip(coo_i, coo_j, coo_k):
        img_w = img[
            i - w_size : i + w_size, j - w_size : j + w_size, k - w_size : k + w_size
        ]
        cnt_pnt = np.sum(np.isin(img_w, 31))
        cnt_left = np.sum(np.isin(img_w, cat.lut.LEFT_LABELS_v1)) - cnt_pnt
        if cnt_left > ((w_size ** 3) / 2):
            img[i, j, k] = left
        if cnt_left < ((w_size ** 3) / 2):
            img[i, j, k] = right

    return img

def convert_104_v1_to_104_v2(aqua_104_v1):
    """Convert 104-label segmentation labels v1 into 104-label segmentation labels v2
    Args:
        aseg (np.ndarray): array of aseg
    
    Returns:
        np.array: converted labels
    """

    LEFT_LATERAL_VENTRICLE = 2
    RIGHT_LATERAL_VENTRICLE = 18
    LEFT_CHOROID_PLEXUS = 16
    RIGHT_CHOROID_PLEXUS = 30
    POST_CC = 32
    MID_POST_CC = 33
    CENTRAL_CC = 34
    MID_ANT_CC = 35
    ANT_CC = 36

    # Convert WM-hypointensities to Left/Right-Cerebral-White-Matter
    aqua_104_v1 = label_to_LR(aqua_104_v1, pnt=31, left=1, right=17)

    # Remove Left-choroid-plexus, Right-choroid-plexus
    aqua_104_v1[aqua_104_v1 == LEFT_CHOROID_PLEXUS] = LEFT_LATERAL_VENTRICLE
    aqua_104_v1[aqua_104_v1 == RIGHT_CHOROID_PLEXUS] = RIGHT_LATERAL_VENTRICLE

    # Combine CCs to CC
    aqua_104_v1[aqua_104_v1 == MID_POST_CC] = POST_CC
    aqua_104_v1[aqua_104_v1 == CENTRAL_CC] = POST_CC
    aqua_104_v1[aqua_104_v1 == MID_ANT_CC] = POST_CC
    aqua_104_v1[aqua_104_v1 == ANT_CC] = POST_CC

    # pull index
    labels = np.unique(aqua_104_v1)
    aqua_104_v2 = np.zeros_like(aqua_104_v1)
    for i in range(len(labels)):
        aqua_104_v2[aqua_104_v1 == labels[i]] = i

    return aqua_104_v2

def convert_aseg_to_104_v2(aseg):
    """Convert aparc+aseg.mgz into 104-label segmentation labels v2

    Combine: Posterior/Mid/Central/Mid-Anterior/Anterior-Corpus-Callosum to Corpus Callosum
    Remove: Left-choroid-plexus, Right-choroid-plexus
    Convert: WM-hypointensities to Left/Right-Cerebral-White-Matter
    Convert: Left/Right-choroid-plexus to Left/Right-Lateral-Ventricle


    Args:
        aseg (np.ndarray): array of aseg
    
    Returns:
        np.array: converted labels
    """

    aqua_104_v1 = convert_aseg_to_104_v1(aseg)

    return convert_104_v1_to_104_v2(aqua_104_v1)