import SimpleITK as sitk


MNI_TEMPLATE = "D:/190208_Dataset/mni_icbm152_t1_tal_nlin_asym_09c.nii"
EXAMPLE_MOVING_IMAGE = "D:/190208_Dataset/moving_image.nii.gz"


def register_img_to_img(
    moving_img_path=EXAMPLE_MOVING_IMAGE,
    fixed_img_path=MNI_TEMPLATE,
    registration_method="affine",
    result_filename="result.nii.gz",
):
    moving_img = sitk.ReadImage(moving_img_path)
    moving_img = sitk.Cast(moving_img, sitk.sitkFloat32)
    fixed_img = sitk.ReadImage(fixed_img_path)
    fixed_img = sitk.Cast(fixed_img, sitk.sitkFloat32)

    # Use custom elastix image filter
    elastix = sitk.ElastixImageFilter()

    if registration_method == "rigid":
        elastix.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    elif registration_method == "affine":
        elastix.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elif registration_method == "non-rigid":
        parameter_map_vector = sitk.VectorOfParameterMap()
        parameter_map_vector.append(sitk.GetDefaultParameterMap("affine"))
        parameter_map_vector.append(sitk.GetDefaultParameterMap("bspline"))
        elastix.SetParameterMap(parameter_map_vector)

    # I am getting the error message: "Too many samples map outside moving image buffer". What does that mean?
    # https://github.com/SuperElastix/elastix/wiki/FAQ
    elastix.SetParameter("AutomaticTransformInitialization", "true")
    elastix.SetFixedImage(fixed_img)
    elastix.SetMovingImage(moving_img)
    elastix.Execute()
    result_img = elastix.GetResultImage()

    if result_filename:
        sitk.WriteImage(result_img, result_filename)

    return result_img


if __name__ == "__main__":
    register_img_to_img()
