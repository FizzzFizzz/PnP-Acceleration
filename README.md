# PnP-Acceleration

> 2025/03/24:
If you want to test it on your own, it would be beneficial if you are familiar with DPIR (https://github.com/cszn/DPIR). The code is based on this pioneer project.


To test: How to test PnP with momentum ?
----
Step 1: Create an environment according to DPIR or KAIR.

Step 2: 
Run 
```
python RED_GNesterov.py
```
to run the standard RED algorithm

Run 
```
python RED_GNesterov.py --Nesterov --r 4
```
to run RED algorithm with Generalize Nesterov momentum of parameter r=4. You can use the restarting criterion of Su with the command "--restarting_su" with Nesterov.

Run 
```
python RED_GNesterov.py --momentum --theta 0.1
```
to run RED algorithm with momentum of parameter fixes parameter theta. You can use the restarting criterion of Li with the command "--restarting_li" with Nesterov.




