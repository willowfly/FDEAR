@echo off
python analyze_metrics.py --data data120 --model resnet50
python analyze_metrics.py --data hfz --model resnet50
python analyze_metrics.py --data wrmq --model resnet50
python analyze_metrics.py --data zy --model resnet50
python analyze_metrics.py --data fj --model resnet50

python analyze_metrics.py --data data120 --model efficientnet_b3
python analyze_metrics.py --data hfz --model efficientnet_b3
python analyze_metrics.py --data wrmq --model efficientnet_b3
python analyze_metrics.py --data zy --model efficientnet_b3
python analyze_metrics.py --data fj --model efficientnet_b3

python analyze_metrics.py --data data120 --model densenet121
python analyze_metrics.py --data hfz --model densenet121
python analyze_metrics.py --data wrmq --model densenet121
python analyze_metrics.py --data zy --model densenet121
python analyze_metrics.py --data fj --model densenet121

python analyze_metrics.py --data data120 --model regnet_x_8gf
python analyze_metrics.py --data hfz --model regnet_x_8gf
python analyze_metrics.py --data wrmq --model regnet_x_8gf
python analyze_metrics.py --data zy --model regnet_x_8gf
python analyze_metrics.py --data fj --model regnet_x_8gf