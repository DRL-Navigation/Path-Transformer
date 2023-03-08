# Path-Transformer

## Introductionn

This is the open-source repo for our thesis -- Path Transformer: Generating Partial Reference Paths for Smooth Movement in Local Obstacle Avoidance

## Quick Start

The project is divided into three sections, namely experience generating, model training, and model testing, in sequential order. 

### Experience Generating

Start generating:
```
cd ${root directory}/exp_gen/PRM+A*
# if you are running for the first time
catkin_make --only-pkg-with-deps img_env
# start generating
source devel/setup.bash
roslaunch img_env gen_exp.launch
# open up a new terminal
source devel/setup.bash
python env_test.py
```

Stop:
Cease the Python terminal by using Ctrl+C.

Output:
You will find the newly collected experience pool in ${root directory}/exp_gen/PRM+A*/output/.

### Model Training:

Put the newly collected experience pool in ${root directory}/train_model/ using methods such as copying or creating a symbolic link.
Execute the Python script:
```
python train.py
```

Output:
You will find the training logs and saved models in ${root directory}/train_model/output/.

### Model Test:
Put the newly trained model in ${root directory}/test_model/DT_test/model/ using methods such as copying or creating a symbolic link.

Start testing:
```
cd ${root directory}/test_model/DT_test
# if you are running for the first time
catkin_make --only-pkg-with-deps img_env
# start testing
source devel/setup.bash
roslaunch img_env DT_test.launch
# open up a new terminal
source devel/setup.bash
python env_test.py
```

Output:
Please be patient and you will see the test results in the Python terminal.
