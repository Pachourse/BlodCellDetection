# README.md - installation guide for Yolo V5 part

# Installation guide

## Files to edit

```
├── BloodCell-Detection-Datatset    -   dataset git repository
├── BloodCellDetection              -   this git repository
│   └── folder where is this readme
└── yolo5                           -   yolov5 git remository
```

operations to execute on files : 


```
mv ~/BloodCell-Detection-Datatset/data.yaml ~/yolo5/
``` shell

edit using vim the firsts lines : 

```
train: ../BloodCell-Detection-Datatset/train/images
val: ../BloodCell-Detection-Datatset/valid/images
```.yaml

## Given files
 - `custom_yolov5.yaml` : is a given file, we used the basic yolo5s model and added data information for classes.

 - `best.pt` : our result of training. 

 - `requirements.txt` : our test requirements

# Execution

training
```
python train.py --img 416 --batch 16 --epochs 100 --data data.yaml --cfg ~/BloodCellDetection/custom_yolov5s.yaml --weights '' --name yoloV5s_results001 --cache
```shell

detect on images - RUN FROM YOLO5 REPO FOLDER
```
python detect.py --weights ~/BloodCellDetection/yoloSetup/best.pt --img 416 --save-txt --conf 0.4 --source ../BloodCell-Detection-Datatset/test/images/
```shell

Be shure to have the `--save-txt` option to have a `.txt` file as entry for the next program.

Inspired by this tutorial Tutorial :
https://colab.research.google.com/drive/1gDZ2xcTOgR39tGGs-EZ6i3RTs16wmzZQ#scrollTo=1Rvt5wilnDyX

## Use results

Results are stored in `~/yolov5/runs/detect/exp*`
