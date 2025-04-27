#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
import re
import numpy as np
import torch
import csv
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import random
import math
import logging
import shutil
import gc
import sys
import sklearn.manifold
import seaborn as sns
import argparse
import matplotlib.gridspec as gridspec
import __main__ as main

# 导入自定义模块
import model_pytorch as model
import utils_torch as utils


def pick_free_gpu(threshold_mb=2000):
    """选择一个空闲内存大于阈值的GPU"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        text=True
    )
    gpu_memory = [int(x) for x in result.stdout.strip().split('\n')]
    for idx, mem in enumerate(gpu_memory):
        if mem > threshold_mb:
            return str(idx)
    return None


def is_interactive():
    """检查是否在交互式环境中运行"""
    return not hasattr(main, '__file__')


def get_layer_index_by_name(model, layername):
    """通过名称获取层的索引"""
    for name, child in model.named_modules():
        if name.endswith(layername):
            return name
    return None


def add_fit_args(parser):
    """添加训练参数到命令行参数解析器"""
    # 训练设置
    parser.add_argument('--batch_size', type=int, default=batch_size, 
                        help='训练的批量大小')  
    parser.add_argument('--local_epoch', type=int, default=local_epoch, 
                        help='训练的周期数')  
    parser.add_argument('--architecture', type=str, default=architecture, 
                        help='在HART和MobileHART之间选择')  
    parser.add_argument('--projection_dim', type=int, default=projection_dim, 
                        help='投影维度的大小')  
    parser.add_argument('--frame_length', type=int, default=frame_length, 
                help='补丁大小')  
    parser.add_argument('--time_step', type=int, default=time_step, 
            help='步长大小')  
    parser.add_argument('--dataset', type=str, default=dataset_name, 
        help='数据集')  
    parser.add_argument('--token_based', type=bool, default=token_based, 
        help='使用Token或全局平均池化')  
    parser.add_argument('--position_device', type=str, default=position_device, 
        help='测试在训练中不包含的位置上进行，如果为空，使用70/10/20的训练/开发/测试比例')  
    args = parser.parse_args()
    return args


def main():
    # 设置随机种子
    random_seed = 1
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 选择GPU
    gpu_id = pick_free_gpu()
    if gpu_id is None:
        print("没有找到空闲GPU，使用CPU")
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        device = torch.device(f"cuda:{0}")
        print(f"使用GPU:{gpu_id}")
    
    # 设置默认超参数
    global architecture, dataset_name, data_config, show_train_verbose
    global segment_size, num_input_channels, learning_rate, dropout_rate
    global local_epoch, frame_length, time_step, position_device, token_based
    global batch_size, projection_dim, filter_attention_head, conv_kernels
    
    architecture = "HART"  # MobileHART, HART
    dataset_name = 'UCI'  # RealWorld, HHAR, UCI, SHL, MotionSense, COMBINED
    data_config = "BALANCED"  # BALANCED, UNBALANCED
    show_train_verbose = 1  # 显示训练详细信息: 0, 1
    segment_size = 128  # 输入窗口大小
    num_input_channels = 6  # 输入通道数
    learning_rate = 5e-3  # 学习率
    dropout_rate = 0.3  # 模型丢弃率
    local_epoch = 200  # 本地周期
    frame_length = 16  # 补丁大小
    time_step = 16  # 步长
    position_device = ''  # 位置/设备
    # ['chest','forearm','head','shin','thigh','upperarm','waist']
    # ['nexus4', 'lgwatch','s3', 's3mini','gear','samsungold']
    token_based = False  # 是否基于token
    
    # 模型超参数
    batch_size = 256
    projection_dim = 192
    filter_attention_head = 4
    # 调整HART的块数，添加或删除卷积核大小
    # 每个卷积核长度对应一个HART块
    conv_kernels = [3, 7, 15, 31, 31, 31]
    
    # 解析命令行参数（如果不是交互式环境）
    if not is_interactive():
        args = add_fit_args(argparse.ArgumentParser(description='人类活动识别Transformer'))
        local_epoch = args.local_epoch
        batch_size = args.batch_size
        architecture = args.architecture
        projection_dim = args.projection_dim
        frame_length = args.frame_length
        time_step = args.time_step
        dataset_name = args.dataset
        token_based = args.token_based
        position_device = args.position_device
        
    # 输入形状和模型配置
    input_shape = (segment_size, num_input_channels)
    projection_half = projection_dim // 2
    projection_quarter = projection_dim // 4
    
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Transformer层的大小
    
    R = projection_half // filter_attention_head
    assert R * filter_attention_head == projection_half
    
    segment_time = [x for x in range(0, segment_size - frame_length + time_step, time_step)]
    assert R * filter_attention_head == projection_half
    if position_device != '':
        assert dataset_name == "RealWorld" or dataset_name == "HHAR"
    
    # 指定活动和结果存储位置
    if dataset_name == 'UCI':
        ACTIVITY_LABEL = ['Walking', 'Upstair', 'Downstair', 'Sitting', 'Standing', 'Lying']
    elif dataset_name == "RealWorld":
        ACTIVITY_LABEL = ['Downstairs', 'Upstairs', 'Jumping', 'Lying', 'Running', 'Sitting', 'Standing', 'Walking']
    elif dataset_name == "MotionSense":
        ACTIVITY_LABEL = ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging']
    elif dataset_name == "HHAR":
        ACTIVITY_LABEL = ['Sitting', 'Standing', 'Walking', 'Upstair', 'Downstairs', 'Biking']
    else:
        # SHL
        ACTIVITY_LABEL = ['Standing', 'Walking', 'Runing', 'Biking', 'Car', 'Bus', 'Train', 'Subway']
    
    activity_count = len(ACTIVITY_LABEL)
    
    architecture_type = str(architecture) + '_' + str(int(frame_length)) + 'frameLength_' + str(time_step) + 'TimeStep_' + str(projection_dim) + "ProjectionSize_" + str(learning_rate) + 'LR'
    if token_based:
        architecture_type = architecture_type + "_tokenBased"
        
    if position_device != '':
        architecture_type = architecture_type + "_PositionWise_" + str(position_device)
    main_dir = './'
    
    if local_epoch < 20:
        architecture_type = "Tests/" + str(architecture_type)
    filepath = main_dir + 'HART_Results/' + architecture_type + '/' + dataset_name + '/'
    os.makedirs(filepath, exist_ok=True)
    
    attention_path = filepath + "attentionImages/"
    os.makedirs(attention_path, exist_ok=True)
    
    best_model_path = filepath + 'bestModels/'
    os.makedirs(best_model_path, exist_ok=True)
    
    current_model_path = filepath + 'currentModels/'
    os.makedirs(current_model_path, exist_ok=True)
    
    # 打印可用GPU
    print("可用GPU数量: ", torch.cuda.device_count())
    
    # 加载数据集
    if dataset_name == "COMBINED":
        dataset_list = ["UCI", "RealWorld", "HHAR", "MotionSense", "SHL_128"]
        ACTIVITY_LABEL = ['Walk', 'Upstair', 'Downstair', 'Sit', 'Stand', 'Lay', 'Jump', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
        activity_count = len(ACTIVITY_LABEL)
        UCI = [0, 1, 2, 3, 4, 5]
        REALWORLD_CLIENT = [2, 1, 6, 5, 7, 3, 4, 0]
        HHAR = [3, 4, 0, 1, 2, 8]
        MotionSense = [2, 1, 3, 4, 0, 7]
        SHL = [4, 0, 7, 8, 9, 10, 11, 12]
        
        central_train_data = []
        central_train_label = []
        central_test_data = []
        central_test_label = []
        for dataset_name_iter in dataset_list:
            client_count = utils.return_client_by_dataset(dataset_name_iter) 
            loaded_dataset = utils.load_dataset_pytorch(dataset_name_iter, client_count, data_config, random_seed, main_dir + 'datasets/')
            central_train_data.append(loaded_dataset.central_train_data.cpu().numpy())
            central_train_label.append(loaded_dataset.central_train_label.cpu().numpy())
            central_test_data.append(loaded_dataset.central_test_data.cpu().numpy())
            central_test_label.append(loaded_dataset.central_test_label.cpu().numpy())
            print(dataset_name_iter + " has class: " + str(np.unique(central_train_label[-1])))
            del loaded_dataset
        
        central_test_label_aligned = []
        central_train_label_aligned = []
        combined_aligned_data = central_test_data
        for index, dataset_name_iter in enumerate(dataset_list):
            if dataset_name_iter == 'UCI':
                central_train_label_aligned.append(central_train_label[index])
                central_test_label_aligned.append(central_test_label[index])
            elif dataset_name_iter == 'RealWorld':
                central_train_label_aligned.append(np.hstack([REALWORLD_CLIENT[label_index] for label_index in central_train_label[index]]))
                central_test_label_aligned.append(np.hstack([REALWORLD_CLIENT[label_index] for label_index in central_test_label[index]]))
            elif dataset_name_iter == 'HHAR':
                central_train_label_aligned.append(np.hstack([HHAR[label_index] for label_index in central_train_label[index]]))
                central_test_label_aligned.append(np.hstack([HHAR[label_index] for label_index in central_test_label[index]]))
            elif dataset_name_iter == 'MotionSense':
                central_train_label_aligned.append(np.hstack([MotionSense[label_index] for label_index in central_train_label[index]]))
                central_test_label_aligned.append(np.hstack([MotionSense[label_index] for label_index in central_test_label[index]]))
            else:
                central_train_label_aligned.append(np.hstack([SHL[label_index] for label_index in central_train_label[index]]))
                central_test_label_aligned.append(np.hstack([SHL[label_index] for label_index in central_test_label[index]]))
        central_train_data = np.vstack((central_train_data))
        central_test_data = np.vstack((central_test_data))
        central_train_label = np.hstack((central_train_label_aligned))
        central_test_label = np.hstack((central_test_label_aligned))
    else:
        client_count = utils.return_client_by_dataset(dataset_name)
        dataset_loader = utils.load_dataset_pytorch(dataset_name, client_count, data_config, random_seed, main_dir + 'datasets/')
        central_train_data = dataset_loader.central_train_data.cpu().numpy()
        central_train_label = dataset_loader.central_train_label.cpu().numpy()
        central_test_data = dataset_loader.central_test_data.cpu().numpy()
        central_test_label = dataset_loader.central_test_label.cpu().numpy()
        client_orientation_train = dataset_loader.client_orientation_train
        client_orientation_test = dataset_loader.client_orientation_test
        orientations_names = dataset_loader.orientations_names
    
    # 如果在RealWorld或HHAR上使用指定位置/设备，我们移除一个并将其用作测试集，并将其他用于训练
    if position_device != '' or dataset_name == 'UCI':
        if dataset_name == "RealWorld":
            total_data = np.vstack((central_train_data, central_test_data))
            total_label = np.hstack((central_train_label, central_test_label))
            try:
                total_orientation = np.hstack((np.hstack((client_orientation_train)), np.hstack((client_orientation_test))))
            except:
                total_orientation = np.hstack((np.hstack([x for x in client_orientation_train]), np.hstack([x for x in client_orientation_test])))
            total_index = list(range(total_orientation.shape[0]))
            test_data_index = np.where(total_orientation == orientations_names.index(position_device))[0]
            train_data_index = np.delete(total_index, test_data_index)
            
            central_train_data = total_data[train_data_index]
            central_test_data = total_data[test_data_index]
            
            central_train_label = total_label[train_data_index]
            central_test_label = total_label[test_data_index]
        elif dataset_name == "HHAR":
            total_data = np.vstack((central_train_data, central_test_data))
            total_label = np.hstack((central_train_label, central_test_label))
            try:
                total_orientation = np.hstack((np.hstack((client_orientation_train)), np.hstack((client_orientation_test))))
            except:
                total_orientation = np.hstack((np.hstack([x for x in client_orientation_train]), np.hstack([x for x in client_orientation_test])))
            total_index = list(range(total_orientation.shape[0]))
            # 0是nexus
            test_data_index = np.where(total_orientation == orientations_names.index(position_device))[0]
            train_data_index = np.delete(total_index, test_data_index)
            
            central_train_data = total_data[train_data_index]
            central_test_data = total_data[test_data_index]
            
            central_train_label = total_label[train_data_index]
            central_test_label = total_label[test_data_index]
        # 使用位置进行评估时，没有测试集，dev=test是相同的
        central_dev_data = central_test_data
        central_dev_label = central_test_label
    else:
        # 使用70 10 20比例
        central_train_data, central_dev_data, central_train_label, central_dev_label = train_test_split(
            central_train_data, central_train_label, test_size=0.125, random_state=random_seed)
    
    # 计算类权重
    temp_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(central_train_label),
        y=central_train_label.ravel()
    )
    class_weights = {j: temp_weights[j] for j in range(len(temp_weights))}
    
    # 创建PyTorch数据集和数据加载器
    # 创建特征和标签的张量
    train_features = torch.FloatTensor(central_train_data)
    train_labels = torch.LongTensor(central_train_label)
    dev_features = torch.FloatTensor(central_dev_data)
    dev_labels = torch.LongTensor(central_dev_label)
    test_features = torch.FloatTensor(central_test_data)
    test_labels = torch.LongTensor(central_test_label)
    
    # 创建数据集
    train_dataset = utils.HARDataset(train_features, train_labels)
    dev_dataset = utils.HARDataset(dev_features, dev_labels)
    test_dataset = utils.HARDataset(test_features, test_labels)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 创建模型
    if architecture == "HART":
        model_classifier = model.HART(
            input_shape=input_shape,
            activity_count=activity_count,
            projection_dim=projection_dim,
            patch_size=frame_length,
            time_step=time_step,
            num_heads=3,
            filter_attention_head=filter_attention_head,
            conv_kernels=conv_kernels,
            dropout_rate=dropout_rate,
            use_tokens=token_based
        ).to(device)
    else:
        model_classifier = model.MobileHART_XS(
            input_shape=input_shape,
            activity_count=activity_count
        ).to(device)
    
    # 转换类权重为PyTorch张量
    class_weights_tensor = torch.FloatTensor([class_weights[i] for i in range(activity_count)]).to(device)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model_classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    
    # 模型摘要
    total_params = sum(p.numel() for p in model_classifier.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_params}")
    
    # 创建检查点路径
    checkpoint_filepath = filepath + "bestValcheckpoint.pt"
    
    # 训练模型
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_acc = 0.0
    
    # 训练循环
    start_time = time.time()
    for epoch in range(local_epoch):
        # 训练阶段
        model_classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model_classifier(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 记录损失和准确率
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # 验证阶段
        model_classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in dev_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model_classifier(inputs)
                loss = criterion(outputs, targets)
                
                # 记录损失和准确率
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # 打印进度
        if show_train_verbose == 1:
            print(f'Epoch {epoch+1}/{local_epoch} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model_classifier.state_dict(), checkpoint_filepath)
            print(f'保存最佳模型，验证准确率: {val_acc:.4f}')
    
    end_time = time.time() - start_time
    print(f"训练时间: {end_time:.2f}秒")
    
    # 保存当前模型
    torch.save(model_classifier.state_dict(), filepath + 'bestTrain.pt')
    
    # 加载最佳模型
    model_classifier.load_state_dict(torch.load(checkpoint_filepath))
    
    # 测试阶段
    model_classifier.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model_classifier(inputs)
            
            # 记录准确率
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            # 保存预测和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算测试准确率
    test_acc = test_correct / test_total
    print(f"测试准确率: {test_acc * 100:.2f}%")
    
    # 保存历史数据
    with open(filepath + 'history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # 计算F1分数
    weight_val_f1 = f1_score(all_targets, all_preds, average='weighted')
    micro_val_f1 = f1_score(all_targets, all_preds, average='micro')
    macro_val_f1 = f1_score(all_targets, all_preds, average='macro')
    
    # 保存模型统计信息
    model_statistics = {
        "Results on server model on ALL testsets": '',
        "\nTrain:": utils.round_number(max(history['train_accuracy'])),
        "\nValidation:": utils.round_number(max(history['val_accuracy'])),
        "\nTest weighted f1:": utils.round_number(weight_val_f1),
        "\nTest micro f1:": utils.round_number(micro_val_f1),
        "\nTest macro f1:": utils.round_number(macro_val_f1),
    }    
    with open(filepath + 'GlobalACC.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(model_statistics.items())
    
    # 获取中间层的表示
    # 在PyTorch中，需要创建一个中间模型
    intermediate_model = utils.extract_intermediate_model_from_base_model(model_classifier, -4)
    
    # 提取嵌入
    embeddings = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = intermediate_model(inputs)
            embeddings.append(outputs.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    
    # t-SNE可视化
    perplexity = 30.0
    tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=show_train_verbose, random_state=random_seed)
    tsne_projections = tsne_model.fit_transform(embeddings)
    
    # 准备标签
    labels_argmax = np.array(all_targets)
    unique_labels = np.unique(labels_argmax)
    
    # 绘制t-SNE图
    if (dataset_name == 'RealWorld' or dataset_name == 'HHAR') and position_device == '':
        utils.project_tsne_with_position(dataset_name, architecture + "_TSNE_Embeds", filepath, ACTIVITY_LABEL,
                                       labels_argmax, orientations_names, client_orientation_test,
                                       tsne_projections, unique_labels)
    else:
        utils.project_tsne(architecture + "_TSNE_Embeds", filepath, ACTIVITY_LABEL,
                         labels_argmax, tsne_projections, unique_labels)
    
    # 绘制混淆矩阵
    results = confusion_matrix(all_targets, all_preds)
    df_cm = pd.DataFrame(results, index=[i for i in ACTIVITY_LABEL],
                      columns=[i for i in ACTIVITY_LABEL])
    plt.figure(figsize=(14, 14))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, cbar=False)
    plt.ylabel('预测')
    plt.xlabel('真实')
    plt.savefig(filepath + 'HeatMap.png')
    
    # 导出ONNX模型
    dummy_input = torch.randn(1, segment_size, num_input_channels, device=device)
    torch.onnx.export(
        model_classifier,
        dummy_input,
        filepath + architecture + '.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # 绘制学习曲线
    utils.plot_learning_curve(history, local_epoch, filepath)
    
    print("训练完成!")


if __name__ == "__main__":
    main()