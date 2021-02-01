# Double-Blind-Collaborative-Learning-DBCL
 Code for Double Blind Collaborative Learning (DBCL)

 Double Blind Collaborative Learning (DBCL), based on random matrix sketching, is used to defend  privacy  leakage. This code is an implementation of this algorithm with python. 

 In this project, we implement FedAvg for Collaborative Learning for models with and without Sketch. To build the model with Sketch, we replace the Convolutional layer and linear layer with SketchConv and SketchLinear.
 e.g., replacing nn.Conv2d(32, 64, 5) with SketchConv(32, 64, kernel_size=5, q=q).


 ## Requirements
- python3 (Python 3.8.6)
- pytorch (1.6.0+cu101)
- torchvision (0.8.0a0+ac3ba94)
- numpy (1.18.5)

## Run
```[bash]
python3 main.py
```

## Configuration
see conf.py

## Code structure

conf.py: configurations of the project

main.py: the entry for the code

utils.py: load dataset and generate local dataset for each clients

model/Client.py: generate each client, including get parameter from server, training its local model and send parameters to the server.

model/Network.py: definition and structure of networks

model/Server.py: select clients, generate Sketch matrixs and send parameters to them in every communication rounds; when clients finish local training, collect and aggeregate all parameters from selected clients to update the model.

model/Sketch.py: generate sketch matrix $$S$$ and compute the sketch of an input $$X$$ and transpose sketch of the sketched matrix.

model/SketchConv.py : SketchConv Layer

model/SketchLinear.py : SketchLinear Layer

