# TrafficFlow Prediction
## Dataset
spatio-temporal data
```
data/train.npz
data/test.npz
data/val.npz
```
graph data
```
data/sensor_graph/adj_mat.pkl
```
## Usage
train the model
```shell
python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
```
predict the result
```shell
python predict.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
```