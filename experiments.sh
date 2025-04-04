for i in 0.05 0.01 0.005 0.001
do 
    python RED_GNesterov.py --dataset_name "set1" --momentum --theta $i
done