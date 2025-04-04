for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do 
    python RED_GNesterov.py --dataset_name "set1" --momentum --theta $i
done