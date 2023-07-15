import torch
import numpy as np
import argparse
import time
import util
import os
# import matplotlib.pyplot as plt
from engine import trainer

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mat.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='import dimension')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=170, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=1, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
# parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save', type=str, default='data/process_model_save/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()


def main():
    # set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # load data

    # 选择显卡
    device = torch.device(args.device)
    # 加载数据
    # args.adjdata = data/sensor_graph/adj_mx.pkl
    # args.adjtype = doubletransition
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    # batch_size = 64
    # 包含训练集,测试集,验证集
    # dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    dataloader = util.load_predict_dataset(args.data, args.batch_size)
    scaler = dataloader['scaler']
    # (3549, 12, 170, 1)
    # print(dataloader['x_predict'].shape)
    # 一个是出度,一个是入度
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    print(args)

    # True
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    # False
    if args.aptonly:
        supports = None
    # in_dim = 2 , seq_length = 12, num_nodes = 207, nhid = 32???, dropout = 0.3
    # lr = 1e-3, weight_decay=1e-4, device=cuda0, supports(出度,入度), gcn_bool = True, addaptadj = True
    # adjinit None
    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                     adjinit)
    print("start inference...", flush=True)
    engine.model.load_state_dict(torch.load(r'F:\test\PythonProject\XLYTF\best_model\WaveNet\_exp1_best_14.5.pth'))

    # data = {}
    # data['x_' + 'predict'] = np.load(os.path.join(args.data, 'test_x.npy'))
    # scaler = util.StandardScaler(mean=data['x_' + 'predict'][..., 0].mean(), std=data['x_' + 'predict'][..., 0].std())
    # data['x_' + 'predict'][..., 0] = scaler.transform(data['x_' + 'predict'][..., 0])
    # data['scaler'] = scaler
    #
    # # torch.Size([3549, 12, 170, 1])
    # predict_x = torch.Tensor(data['x_' + 'predict']).to(device)
    # predict_x = predict_x.transpose(1, 3)
    # with torch.no_grad():
    #     preds = engine.model(predict_x).transpose(1, 3)
    # print(preds)
    # print(preds.shape)
    # exit(0)
    # print(predict_x)
    # print(predict_x.shape)
    # exit(0)

    outputs = []
    for iter, x in enumerate(dataloader['predict_loader'].get_iterator_predict()):
        testx = torch.Tensor(x).to(device)  # torch.Size([64, 12, 170, 1])
        testx = testx.transpose(1, 3)  # torch.Size([64, 1, 170, 12])
        testx = torch.nn.functional.pad(testx, (1, 0, 0, 0))  # torch.Size([64, 1, 170, 13])
        with torch.no_grad():
            # [64,12,170,1] -> [64,1,170,12]
            # preds = engine.model(testx).transpose(1, 3)
            preds = engine.model(testx).transpose(1, 3)
            preds = scaler.inverse_transform(preds)  # [64,1,207,12]

        outputs.append(preds)  # [64,1,170,12]

    # torch.Size([3549, 1, 170, 12])
    yhat = torch.cat(outputs, dim=0)
    # print(yhat)
    yhat = yhat.transpose(1, 3).detach().cpu()  # [3549,12,170,1]
    yhat = np.abs(yhat)
    np.save(os.path.join(r'F:\test\PythonProject\XLYTF\best_model\WaveNet', 'predict.npy'), yhat)
    # print(yhat.shape)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
