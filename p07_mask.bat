@echo off
cls
python .\eval_mask_model.py --model resnet50 --data data120 --index %2 --mask %1