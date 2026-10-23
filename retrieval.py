import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss,ProxyAnchorLoss
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam,SGD
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.optim import lr_scheduler
from model import Extractor, Discriminator, Generator, set_bn_eval
from utils import DomainDataset
import scipy.io as sio
import torch.nn.functional as F
from RetrievalEvaluation import RetrievalEvaluation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


def retrieval_SHREC(backbone, encoder, test_query_loader,test_target_loader, top_k=5, save_dir='retrieval_results'):


    # backbone_query.eval()
    # backbone_target.eval()
    backbone.eval()
    encoder.eval()


    sketchLabMatrix = []
    shapeLabMatrix = []
    sketchMatrix = []
    shapeMatrix = []
    sketchPaths = []
    shapePaths = []

    with torch.no_grad():
        for inputs, labels, paths in test_query_loader:
            inputs = inputs.cuda()
            # outputs = backbone_align(backbone_target(encoder(inputs, is_target=0), is_target=0))
            # outputs= backbone_align(backbone_query(inputs, is_target=0))
            outputs = backbone(encoder(inputs,is_target = 0),is_target = 0)
            sketchMatrix.append(outputs)
            sketchLabMatrix.append(labels)
            sketchPaths.extend(paths)
    start_time = time.time()
    print("程序执行开始时间为：", start_time)
    with torch.no_grad():  
        for inputs, labels, paths in test_target_loader:
            inputs = inputs.cuda()
            # outputs = backbone_align(backbone_target(inputs, is_target=1))
            outputs = backbone(inputs,is_target = 1)
            shapeMatrix.append(outputs)
            shapeLabMatrix.append(labels)
            shapePaths.extend(paths)

    sketchMatrix = torch.cat(sketchMatrix)  # (Nq, d)
    sketchLabMatrix = torch.cat(sketchLabMatrix).cpu().numpy()
    shapeMatrix = torch.cat(shapeMatrix)    # (Nd, d)
    shapeLabMatrix = torch.cat(shapeLabMatrix).cpu().numpy()

    distM = 1 - torch.mm(F.normalize(sketchMatrix, dim=-1),
                         F.normalize(shapeMatrix, dim=-1).t())  # cosine distance
    distM = distM.cpu().numpy()

    # Retrieve top-k
    topk_indices = np.argsort(distM, axis=1)[:, :top_k]
    topk_paths = [[shapePaths[j] for j in row] for row in topk_indices]

    # Save retrieval results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    retrieval_df = pd.DataFrame({
        'query': sketchPaths,
        'label': sketchLabMatrix,
        **{f'top_{i+1}': [paths[i] for paths in topk_paths] for i in range(top_k)}
    })
    
    end_time = time.time()
    print("程序执行结束时间为：", end_time)
    time_diff = end_time - start_time  # 计算时间差
    print("程序执行时间为：", time_diff)

    retrieval_df.to_csv(os.path.join(save_dir, 'sketch_retrieval_results.csv'), index=False)
    print(f"Retrieval results saved to {os.path.join(save_dir, 'sketch_retrieval_results_fake.csv')}")

    return retrieval_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='/media/jiao/datasets', type=str, help='Datasets root path')
    parser.add_argument('--data_name', default='SHREC13_ZS_DTO', type=str, choices=['IM2MN', 'MI3DOR',"SHREC13_ZS_DDGAN","SHREC14_ZS_DDGAN","SHREC13_ZS_DTO","SHREC14_ZS_DTO"],
                        help='Dataset name')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'vgg16'],
                        help='Backbone type')
    # parser.add_argument('--emb_dim', default=2048, type=int, help='Embedding dim')
    # parser.add_argument('--sem_dim', default=512, type=int, help='Embedding dim')
    parser.add_argument('--emb_dim', default=512, type=int, help='Embedding dim')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs over the model to train')
    parser.add_argument('--warmup', default=2, type=int, help='Number of warmups over the extractor to train')
    parser.add_argument('--save_root', default='best', type=str, help='Result saved root path')

     # args parse
    args = parser.parse_args()
    data_root, data_name, backbone_type, emb_dim = args.data_root, args.data_name, args.backbone_type, args.emb_dim
    batch_size, epochs, warmup, save_root = args.batch_size, args.epochs, args.warmup, args.save_root

    # data prepare
    
    train_data = DomainDataset(data_root, data_name, split='train')
    #test_data_query = DomainDataset(data_root, data_name, split='val',flag="query")
    #test_data_target = DomainDataset(data_root, data_name, split='test',flag="target")
    val_data_query = DomainDataset(data_root, data_name, split='val',flag="query")
    val_data_target = DomainDataset(data_root, data_name, split='val',flag="target")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    #test_loader_query = DataLoader(test_data_query, batch_size=batch_size, shuffle=False, num_workers=0)
    #test_loader_target = DataLoader(test_data_target, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader_query = DataLoader(val_data_query, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader_target = DataLoader(val_data_target, batch_size=batch_size, shuffle=True, num_workers=0)


    save_name_pre = '{}_{}_{}'.format(data_name, backbone_type, emb_dim)
    # model define
    # extractor_query = Extractor(backbone_type, emb_dim)
    # # extractor.load_state_dict(torch.load('{}/{}_extractor.pth'.format(save_root, save_name_pre), map_location='cpu'))
    # extractor_query = extractor_query.cuda()

    # extractor_target= Extractor(backbone_type, emb_dim)
    # # extractor.load_state_dict(torch.load('{}/{}_extractor.pth'.format(save_root, save_name_pre), map_location='cpu'))
    # extractor_target = extractor_target.cuda()

    # extractor_align = Alignment_Space(emb_dim,sem_dim)
    # extractor_align = extractor_align.cuda()
    # extractor = [extractor_query,extractor_target,extractor_align]

    extractor_query = Extractor(backbone_type, emb_dim)
    # extractor.load_state_dict(torch.load('{}/{}_extractor_query.pth'.format(save_root, save_name_pre), map_location='cpu'))
    extractor_query = extractor_query.cuda()

    extractor_target = Extractor(backbone_type, emb_dim)
    # extractor.load_state_dict(torch.load('{}/{}__extractor_target.pth'.format(save_root, save_name_pre), map_location='cpu'))
    extractor_target = extractor_target.cuda()

    extractor = [extractor_query,extractor_target]


    generator = Generator(in_channels=8, num_block=8).cuda()
    generator = generator.cuda()

    discriminator = Discriminator(in_channels=8).cuda()
    discriminator = discriminator.cuda()

    # loss setup
    # class_criterion = NormalizedSoftmaxLoss(len(train_data.classes), sem_dim)
    class_criterion = NormalizedSoftmaxLoss(len(train_data.classes), emb_dim)
    class_criterion=class_criterion.cuda()


    extractor_query.load_state_dict(torch.load('{}/{}_extractor_query.pth'.format(save_root, save_name_pre), map_location='cpu'))
    extractor_target.load_state_dict(torch.load('{}/{}_extractor_target.pth'.format(save_root, save_name_pre), map_location='cpu'))
    # extractor_align.load_state_dict(torch.load('{}/{}_extractor_align.pth'.format(save_root, save_name_pre), map_location='cpu'))
    generator.load_state_dict(torch.load('{}/{}_generator.pth'.format(save_root, save_name_pre), map_location='cpu'))
    discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(save_root, save_name_pre), map_location='cpu'))
    class_criterion.load_state_dict(torch.load('{}/{}_proxies.pth'.format(save_root, save_name_pre), map_location='cpu'))
    # best_map = val_SHREC(extractor_target,extractor_align, generator, test_loader_query,test_loader_target)
    retrieval_SHREC(extractor_target, generator, val_loader_query, val_loader_target)


