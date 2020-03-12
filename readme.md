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
