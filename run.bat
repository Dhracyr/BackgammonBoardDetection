cd "G:/Schule/Studium/BeginningOfAll/GitHub Repos/BackgammonBoardDetection"
python yolov7/train.py --data "G:/Schule/Studium/BeginningOfAll/GitHub Repos/BackgammonBoardDetection/data/dataset_combined/data.yaml" --cfg "cfg/training/yolov7.yaml" --weights "yolov7.pt" --epochs 25 --img 640 --batch 16 --device 0
