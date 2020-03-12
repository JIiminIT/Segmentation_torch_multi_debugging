# Tinycat

imaging팀 공동사용을 목적으로하는 tabbycat의 경량화 버젼 라이브러리

## 설치

```
# 가상환경 구성 (생략가능)
pip install virtualenv
virtualenv tinycat-env
tinycat-env\Scripts\activate

# 설치
python setup.py install

# 테스트
tinycat

tinycat: neurophet nifti & neural network package
==================================================

Contents
--------
Tinycat imports and wraps several functions from the nibabel namespace,
Implements various numpy-based matrix calculation & neuroimaging functional APIs

Supported apps: ['evaluator', 'to_nifti', 'aseg_to_label', 'surfer_mgz_to_nifti']
Version: 1.0.0
```

## apps

tinycat python 패키지를 설치한 경우 커맨드라인 명령만으로 실행 할 수 있는 유틸리티들이 함께 설치됩니다.

### evaluate_segmentation

```
# segmentation label과 prediction 결과의 폴더를 지정하여
# csv 파일로 dice coefficient, jaccard coefficient 등의 metric을
# 연산할 수 있는 모듈입니다.
evaluate_segmentation --help
usage: evaluate_segmentation [-h] --prediction_directory PREDICTION_DIRECTORY
                             --label_directory LABEL_DIRECTORY
                             [--result_directory RESULT_DIRECTORY]
                             [--prediction_subfix PREDICTION_SUBFIX]
                             [--label_subfix LABEL_SUBFIX]
                             [--filename_prefix FILENAME_PREFIX]
                             [--metrics METRICS] [--n_classes N_CLASSES]

optional arguments:
  -h, --help            show this help message and exit
  --prediction_directory PREDICTION_DIRECTORY, -p PREDICTION_DIRECTORY
                        evaluation directory with prediction nifti files
  --label_directory LABEL_DIRECTORY, -l LABEL_DIRECTORY
                        evaluation directory with label nifti files
  --result_directory RESULT_DIRECTORY, -r RESULT_DIRECTORY
                        result directory to save csv formatted evaluation
                        result
  --prediction_subfix PREDICTION_SUBFIX
                        wildcard subfix to search prediction files (default:
                        *.nii)
  --label_subfix LABEL_SUBFIX
                        wildcard subfix to search label files (default: *.nii)
  --filename_prefix FILENAME_PREFIX
                        result filename prefix if needed
  --metrics METRICS     calculates 'dice', 'jacc', 'accr' if included.
  --n_classes N_CLASSES
                        number of classes

# directory만을 지정해주는 경우, 경로 내 nifti 파일의 수가 동일하고 이름이 일정한 경우 정상적인 결과가 출력됩니다.
# result_directory가 지정되지 않은 경우 결과 csv파일은 실행 경로에 저장됩니다.
evaluate_segmentation --label_directory D:/labels/ --prediction_directory D:/predictions/ --n_classes 105
```

### aseg_to_label

```
# freesurfer의 subject 폴더 내의 aseg+aparc.mgz 영상을 통해 root_dir 경로에 
# tES (8-label), AQUA (104-label)로 변환한 nifti 영상을 출력합니다.
# freesurfer의 결과물이 다음과 같이 존재한다는 전재하에 실행합니다.
# root_dir
# ㄴ surfer_result
#     ㄴ a.nii.gz
#     ㄴ b.nii.gz
aseg_to_label --mode aqua --root_dir d:/root_dir/
```

### to_nifti

```
# nifti 형식이 아닌 파일을 .nii.gz 형식으로 변환해 출력합니다.
to_nifti a.mgz
```

## 구성

* evaluation  
* label  
* lut  
* metrics  
* multicore  
* nifti  
* normalization  
* patch  
* plot  
* volume  

### tf

tf에는 tensorflow로 작성된 인공지능 관련 모듈들이 포함됩니다.

### thirdparty

thirdparty subpackage에는 외부 소프트웨어 제어 관련 모듈들이 포함됩니다.

* elastix
* freesurfer
* gpu

```python
# gpu.py
import tinycat as cat
cat.thirdparty.gpu.describe()
--------------------------------------------------------------------------------
GPU_ID: 0
UUID: GPU-270833d5-08a1-a9f5-55b6-6614529a5060
LOAD: 0.0
MEMORY_UTIL: 0.022054036458333332
MEMORY_TOTAL: 12288.0
MEMORY_USED: 271.0
MEMORY_FREE: 12017.0
DRIVER: 442.59
NAME: TITAN X (Pascal)
SERIAL: 0324416079891
DISPLAY_MODE: Enabled
DISPLAY_ACTIVE: Enabled
--------------------------------------------------------------------------------
```
