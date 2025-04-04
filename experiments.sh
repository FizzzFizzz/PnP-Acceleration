for i in 15
do 
    python new_PnP_main_GeneralizedNesterov_RED_deblur_color.py --r $i --dataset_name "set1" --momentum_Nesterov --dont_save_images
done