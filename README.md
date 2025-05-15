# HPGNN

Welcome to the source code repository for our paper: **Leveraging Higher-order Topological Structures and Personalized PageRank for Heterophily Mitigation in Graph Neural Networks**





# Prerequisites:
Ensure you have the following libraries installed:
```
pytorch
pytorch-geometric
torch_scatter
torch_sparse
networkx
numpy
```
#  Node Classification 
go to folder `HPGNN/`

## Run experiment with Texas:


```sh
cd HPGNN
python main.py --RPMAX 100 \
        --lr 0.1 \
        --alpha 0.6 \
        --weight_decay 0.0001 \
        --dprate 0.5
```

