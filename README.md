# Double-Blind-Collaborative-Learning-DBCL
 Code for Double Blind Collaborative Learning (DBCL)

 Double Blind Collaborative Learning (DBCL), based on random matrix sketching, is used to defend  privacy  leakage. This code is an implementation of this algorithm with python. 
 
 In this project, we implement FedAvg for Collaborative Learning for models with and without Sketch. To build the model with Sketch, we replace the Convolutional layer and linear layer with SketchConv and SketchLinear.
 e.g., replacing nn.Conv2d(32, 64, 5) with SketchConv(32, 64, kernel_size=5, q=q).


 ## Requirements
- python3
- pytorch
- torchvision
- numpy

## Run
```[bash]
python3 main.py
```

## Configuration
see conf.py
