# wmt-segmentation-cuellar-and-moya

Make sure the directory tree is like this:


+ -- wmt-segmentation-cuellar-and-moya
    + --BrainPTM
        + --Annotations
        + --data
    + --WMTdata
        + --Annotations
        + --AnnotationsTif
        + --T1_tif

*install fcm-segmentation:

```
pip install intensity-normalization
```

*run preprocessing
```
 python preprocessing.py
```

*Transform data to tif
```
python nii_to_tif.py *(directory to nifti input folder)* .../WMTdata/T1 *(directory to tif output folder)* .../WMTdata/T1_tif
```

*Transform annotations to Tif
```
python nii_to_tif.py *(directory to nifti input folder)* .../WMTdata/Annotations *(directory to tif output folder)* .../WMTdata/AnnotationsTif
```
*run segmentation
```
python segmentation.py
```
