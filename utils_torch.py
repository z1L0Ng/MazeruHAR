#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle  # 替代hickle
import seaborn as sns
import logging
from sklearn.model_selection import StratifiedKFold
import scipy.stats


def get_available_gpus():
    """返回可用的GPU设备列表"""
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def get_available_cpus():
    """返回可用的CPU设备"""
    import multiprocessing
    return [f"CPU Core {i}" for i in range(multiprocessing.cpu_count())]


class DataHolder:
    """数据持有者类，用于存储和组织数据集"""
    def __init__(self):
        self.client_data_train = []
        self.client_label_train = []
        self.client_data_test = []
        self.client_label_test = []
        self.central_train_data = []
        self.central_train_label = []
        self.central_test_data = []
        self.central_test_label = []
        self.client_orientation_train = []
        self.client_orientation_test = []
        self.orientations_names = None
        self.activity_labels = []
        self.client_count = None


def return_client_by_dataset(dataset_name):
    """根据数据集名称返回客户端数量"""
    if dataset_name == 'UCI' or dataset_name == 'UCI_ORIGINAL':
        return 5
    elif dataset_name == "RealWorld":
        return 15
    elif dataset_name == "MotionSense":
        return 24
    elif dataset_name == 'SHL':
        return 9
    elif dataset_name == "HHAR":
        return 51
    else:
        raise ValueError('Unknown dataset')


def load_file(filepath):
    """加载单个文件为numpy数组"""
    dataframe = pd.read_csv(filepath, header=None)
    return dataframe.values


def load_group(filenames, prefix=''):
    """加载文件列表，例如给定变量的x、y、z数据"""
    loaded = []
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # 堆叠组使特征成为第3维
    loaded = np.dstack(loaded)
    return loaded


def load_dataset(group, main_dir, prefix=''):
    """加载数据集组，例如训练或测试"""
    filepath = main_dir + 'datasetStandardized/' + prefix + '/' + group + '/'
    filenames = []
    filenames += ['AccX' + prefix + '.csv', 'AccY' + prefix + '.csv', 'AccZ' + prefix + '.csv']
    filenames += ['GyroX' + prefix + '.csv', 'GyroY' + prefix + '.csv', 'GyroZ' + prefix + '.csv']
    X = load_group(filenames, filepath)
    y = load_file(main_dir + 'datasetStandardized/' + prefix + '/' + group + '/Label' + prefix + '.csv')
    return X, y


def project_tsne(file_name, filepath, activity_label, labels_argmax, tsne_projections, unique_labels):
    """使用t-SNE投影数据并可视化"""
    plt.figure(figsize=(16, 16))
    graph = sns.scatterplot(
        x=tsne_projections[:, 0], y=tsne_projections[:, 1],
        hue=labels_argmax,
        palette=sns.color_palette(n_colors=len(unique_labels)),
        s=50,
        alpha=1.0,
        rasterized=True
    )
    legend = graph.legend_
    for j, label in enumerate(unique_labels):
        legend.get_texts()[j].set_text(activity_label[int(label)])

    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelleft=False,
        labelbottom=False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(filepath + file_name + ".svg", bbox_inches="tight", format="svg")
    plt.show()


def project_tsne_with_position(dataset_name, file_name, filepath, activity_label, labels_argmax,
                              orientations_names, client_orientation_test, tsne_projections, unique_labels):
    """使用t-SNE投影数据并根据位置/设备可视化"""
    class_data = [activity_label[i] for i in labels_argmax]
    orientation_data = [orientations_names[i] for i in np.hstack((client_orientation_test))]
    if dataset_name == 'RealWorld':
        orientation_name = 'Position'
    else:
        orientation_name = 'Device'
    panda_data = {'col1': tsne_projections[:, 0], 'col2': tsne_projections[:, 1],
                 'Classes': class_data, orientation_name: orientation_data}
    panda_data_frame = pd.DataFrame(data=panda_data)

    plt.figure(figsize=(16, 16))
    sns.scatterplot(data=panda_data_frame, x="col1", y="col2", hue="Classes", style=orientation_name,
                    palette=sns.color_palette(n_colors=len(unique_labels)),
                    s=50, alpha=1.0, rasterized=True, )
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelleft=False,
        labelbottom=False)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(filepath + file_name + ".png", bbox_inches="tight")
    plt.show()


def create_segments_and_labels_mobiact(df, time_steps, step, label_name="LabelsEncoded", n_features=6):
    """创建数据片段和标签"""
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        acc_x = df['acc_x'].values[i: i + time_steps]
        acc_y = df['acc_y'].values[i: i + time_steps]
        acc_z = df['acc_z'].values[i: i + time_steps]

        gyro_x = df['gyro_x'].values[i: i + time_steps]
        gyro_y = df['gyro_y'].values[i: i + time_steps]
        gyro_z = df['gyro_z'].values[i: i + time_steps]

        # 检索该段中最常用的标签
        label = scipy.stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        labels.append(label)

    # 将片段转换为更好的形状
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, n_features)
    labels = np.asarray(labels)

    return reshaped_segments, labels


class HARDataset(torch.utils.data.Dataset):
    """
    人体活动识别数据集加载器
    """
    def __init__(self, features, labels, transform=None):
        """
        初始化数据集
        
        参数:
            features: 特征张量
            labels: 标签张量
            transform: 变换函数（可选）
        """
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    @staticmethod
    def create_data_loaders(train_features, train_labels, dev_features, dev_labels, 
                          test_features, test_labels, batch_size, num_workers=4):
        """
        创建训练、验证和测试数据加载器
        
        参数:
            train_features, train_labels: 训练数据
            dev_features, dev_labels: 验证数据
            test_features, test_labels: 测试数据
            batch_size: 批量大小
            num_workers: 数据加载的工作线程数
            
        返回:
            train_loader, dev_loader, test_loader
        """
        # 创建数据集
        train_dataset = HARDataset(train_features, train_labels)
        dev_dataset = HARDataset(dev_features, dev_labels)
        test_dataset = HARDataset(test_features, test_labels)
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, dev_loader, test_loader


def load_dataset_pytorch(dataset_name, client_count, data_config, random_seed, main_dir, stratified_split=True):
    """加载数据集并转换为PyTorch格式"""
    # 创建一个DataHolder实例来存储数据
    data_return = DataHolder()
    
    if dataset_name == "UCI":
        # 加载UCI数据集
        try:
            central_train_data = pickle.load(open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/trainX.pkl', 'rb'))
            central_test_data = pickle.load(open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/testX.pkl', 'rb'))
            central_train_label = pickle.load(open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/trainY.pkl', 'rb'))
            central_test_label = pickle.load(open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/testY.pkl', 'rb'))
        except:
            # 尝试加载hickle格式（如果存在）
            import hickle as hkl
            central_train_data = hkl.load(main_dir + 'datasetStandardized/' + str(dataset_name) + '/trainX.hkl')
            central_test_data = hkl.load(main_dir + 'datasetStandardized/' + str(dataset_name) + '/testX.hkl')
            central_train_label = hkl.load(main_dir + 'datasetStandardized/' + str(dataset_name) + '/trainY.hkl')
            central_test_label = hkl.load(main_dir + 'datasetStandardized/' + str(dataset_name) + '/testY.hkl')
            
            # 保存为pickle格式以便将来使用
            os.makedirs(main_dir + 'datasetStandardized/' + str(dataset_name), exist_ok=True)
            pickle.dump(central_train_data, open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/trainX.pkl', 'wb'))
            pickle.dump(central_test_data, open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/testX.pkl', 'wb'))
            pickle.dump(central_train_label, open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/trainY.pkl', 'wb'))
            pickle.dump(central_test_label, open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/testY.pkl', 'wb'))
        
        data_return.central_train_data = central_train_data
        data_return.central_test_data = central_test_data
        data_return.central_train_label = central_train_label
        data_return.central_test_label = central_test_label

    elif dataset_name == "SHL":
        # 加载SHL数据集
        try:
            client_data = pickle.load(open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsData.pkl', 'rb'))
            client_label = pickle.load(open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsLabel.pkl', 'rb'))
        except:
            import hickle as hkl
            client_data = hkl.load(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsData.hkl')
            client_label = hkl.load(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsLabel.hkl')
            
            # 保存为pickle格式
            os.makedirs(main_dir + 'datasetStandardized/' + str(dataset_name), exist_ok=True)
            pickle.dump(client_data, open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsData.pkl', 'wb'))
            pickle.dump(client_label, open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsLabel.pkl', 'wb'))
            
        client_count = client_data.shape[0]
        
        client_data_train = []
        client_label_train = []
        client_data_test = []
        client_label_test = []
        
        for i in range(0, client_count):
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            skf.get_n_splits(client_data[i], client_label[i])
            train_index = []
            test_index = []
            for enu_index, (train_idx, test_idx) in enumerate(skf.split(client_data[i], client_label[i])):
                # 让索引4处的索引用于测试
                if enu_index != 2:
                    train_index.append(test_idx)
                else:
                    test_index = test_idx
            train_index = np.hstack((train_index))
            client_data_train.append(client_data[i][train_index])
            client_label_train.append(client_label[i][train_index])
            client_data_test.append(client_data[i][test_index])
            client_label_test.append(client_label[i][test_index])
        
        client_data_train = np.asarray(client_data_train, dtype=object)
        client_data_test = np.asarray(client_data_test, dtype=object)
        client_label_train = np.asarray(client_label_train, dtype=object)
        client_label_test = np.asarray(client_label_test, dtype=object)
        
        central_train_data = np.vstack((client_data_train))
        central_train_label = np.hstack((client_label_train))
        central_test_data = np.vstack((client_data_test))
        central_test_label = np.hstack((client_label_test))
        
        data_return.client_data_train = client_data_train
        data_return.client_label_train = client_label_train
        data_return.client_data_test = client_data_test
        data_return.client_label_test = client_label_test
        data_return.central_train_data = central_train_data
        data_return.central_train_label = central_train_label
        data_return.central_test_data = central_test_data
        data_return.central_test_label = central_test_label
        
    elif dataset_name == "RealWorld":
        # 加载RealWorld数据集
        orientations_names = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
        client_data_train = {new_list: [] for new_list in range(client_count)}
        client_label_train = {new_list: [] for new_list in range(client_count)}
        client_data_test = {new_list: [] for new_list in range(client_count)}
        client_label_test = {new_list: [] for new_list in range(client_count)}
        
        try:
            client_orientation_data = pickle.load(open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsData.pkl', 'rb'))
            client_orientation_label = pickle.load(open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsLabel.pkl', 'rb'))
        except:
            import hickle as hkl
            client_orientation_data = hkl.load(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsData.hkl')
            client_orientation_label = hkl.load(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsLabel.hkl')
            
            # 保存为pickle格式
            os.makedirs(main_dir + 'datasetStandardized/' + str(dataset_name), exist_ok=True)
            pickle.dump(client_orientation_data, open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsData.pkl', 'wb'))
            pickle.dump(client_orientation_label, open(main_dir + 'datasetStandardized/' + str(dataset_name) + '/clientsLabel.pkl', 'wb'))
        
        client_orientation_test = {new_list: [] for new_list in range(client_count)}
        client_orientation_train = {new_list: [] for new_list in range(client_count)}
        
        orientation_index = 0
        for client_data, client_label in zip(client_orientation_data, client_orientation_label):
            for i in range(0, client_count):
                skf = StratifiedKFold(n_splits=5, shuffle=False)
                skf.get_n_splits(client_data[i], client_label[i])
                train_index = []
                test_index = []
                for enu_index, (train_idx, test_idx) in enumerate(skf.split(client_data[i], client_label[i])):
                    # 让索引2处的索引用于测试
                    if enu_index != 2:
                        train_index.append(test_idx)
                    else:
                        test_index = test_idx
                
                train_index = np.hstack((train_index))
                client_data_train[i].append(client_data[i][train_index])
                client_label_train[i].append(client_label[i][train_index])
                client_data_test[i].append(client_data[i][test_index])
                client_label_test[i].append(client_label[i][test_index])
                
                client_orientation_test[i].append(np.full((len(test_index)), orientation_index))
                client_orientation_train[i].append(np.full((len(train_index)), orientation_index))
            
            orientation_index += 1
        
        for i in range(0, client_count):
            client_data_train[i] = np.vstack((client_data_train[i]))
            client_data_test[i] = np.vstack((client_data_test[i]))
            client_label_train[i] = np.hstack((client_label_train[i]))
            client_label_test[i] = np.hstack((client_label_test[i]))
            client_orientation_test[i] = np.hstack((client_orientation_test[i]))
            client_orientation_train[i] = np.hstack((client_orientation_train[i]))
        
        client_orientation_train = np.asarray(list(client_orientation_train.values()), dtype=object)
        client_orientation_test = np.asarray(list(client_orientation_test.values()), dtype=object)
        
        client_data_train = np.asarray(list(client_data_train.values()), dtype=object)
        client_data_test = np.asarray(list(client_data_test.values()), dtype=object)
        
        client_label_train = np.asarray(list(client_label_train.values()), dtype=object)
        client_label_test = np.asarray(list(client_label_test.values()), dtype=object)
        
        central_train_data = np.vstack((client_data_train))
        central_train_label = np.hstack((client_label_train))
        
        central_test_data = np.vstack((client_data_test))
        central_test_label = np.hstack((client_label_test))
        
        data_return.client_data_train = client_data_train
        data_return.client_label_train = client_label_train
        data_return.client_data_test = client_data_test
        data_return.client_label_test = client_label_test
        data_return.central_train_data = central_train_data
        data_return.central_train_label = central_train_label
        data_return.central_test_data = central_test_data
        data_return.central_test_label = central_test_label
        data_return.client_orientation_train = client_orientation_train
        data_return.client_orientation_test = client_orientation_test
        data_return.orientations_names = orientations_names
        
    else:
        # 加载其他数据集
        client_data = []
        client_label = []
        
        for i in range(0, client_count):
            try:
                client_data.append(pickle.load(open(main_dir + 'datasetStandardized/' + dataset_name + '/UserData' + str(i) + '.pkl', 'rb')))
                client_label.append(pickle.load(open(main_dir + 'datasetStandardized/' + dataset_name + '/UserLabel' + str(i) + '.pkl', 'rb')))
            except:
                import hickle as hkl
                client_data.append(hkl.load(main_dir + 'datasetStandardized/' + dataset_name + '/UserData' + str(i) + '.hkl'))
                client_label.append(hkl.load(main_dir + 'datasetStandardized/' + dataset_name + '/UserLabel' + str(i) + '.hkl'))
                
                # 保存为pickle格式
                os.makedirs(main_dir + 'datasetStandardized/' + dataset_name, exist_ok=True)
                pickle.dump(client_data[-1], open(main_dir + 'datasetStandardized/' + dataset_name + '/UserData' + str(i) + '.pkl', 'wb'))
                pickle.dump(client_label[-1], open(main_dir + 'datasetStandardized/' + dataset_name + '/UserLabel' + str(i) + '.pkl', 'wb'))
        
        if dataset_name == "HHAR":
            try:
                orientations = pickle.load(open(main_dir + 'datasetStandardized/HHAR/deviceIndex.pkl', 'rb'))
            except:
                import hickle as hkl
                orientations = hkl.load(main_dir + 'datasetStandardized/HHAR/deviceIndex.hkl')
                pickle.dump(orientations, open(main_dir + 'datasetStandardized/HHAR/deviceIndex.pkl', 'wb'))
                
            orientations_names = ['nexus4', 'lgwatch', 's3', 's3mini', 'gear', 'samsungold']
        
        client_data_train = []
        client_label_train = []
        client_data_test = []
        client_label_test = []
        client_orientation_train = []
        client_orientation_test = []
        
        for i in range(0, client_count):
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            skf.get_n_splits(client_data[i], client_label[i])
            train_index = []
            test_index = []
            for enu_index, (train_idx, test_idx) in enumerate(skf.split(client_data[i], client_label[i])):
                if enu_index != 2:
                    train_index.append(test_idx)
                else:
                    test_index = test_idx
            train_index = np.hstack((train_index))
            client_data_train.append(client_data[i][train_index])
            client_label_train.append(client_label[i][train_index])
            client_data_test.append(client_data[i][test_index])
            client_label_test.append(client_label[i][test_index])
            client_orientation_train.append(train_index)
            client_orientation_test.append(test_index)
        
        if dataset_name == "HHAR":
            for i in range(0, client_count):
                client_orientation_test[i] = orientations[i][client_orientation_test[i]]
                client_orientation_train[i] = orientations[i][client_orientation_train[i]]
        
        central_train_data = np.vstack((client_data_train))
        central_train_label = np.hstack((client_label_train))
        
        central_test_data = np.vstack((client_data_test))
        central_test_label = np.hstack((client_label_test))
        
        data_return.client_data_train = client_data_train
        data_return.client_label_train = client_label_train
        data_return.client_data_test = client_data_test
        data_return.client_label_test = client_label_test
        data_return.central_train_data = central_train_data
        data_return.central_train_label = central_train_label
        data_return.central_test_data = central_test_data
        data_return.central_test_label = central_test_label
        data_return.client_orientation_train = client_orientation_train
        data_return.client_orientation_test = client_orientation_test
        data_return.orientations_names = orientations_names
    
    # 创建PyTorch数据集和数据加载器
    train_dataset = HARDataset(data_return.central_train_data, data_return.central_train_label)
    test_dataset = HARDataset(data_return.central_test_data, data_return.central_test_label)
    
    data_return.train_dataset = train_dataset
    data_return.test_dataset = test_dataset
    
    return data_return


def plot_learning_curve(history, epochs, filepath):
    """绘制学习曲线（训练和验证准确率、损失）"""
    # 绘制训练和验证准确率
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history['train_accuracy'])
    plt.plot(epoch_range, history['val_accuracy'])
    
    # 标记最大准确率点
    plt.plot(epoch_range, history['val_accuracy'], markevery=[np.argmax(history['val_accuracy'])], 
             ls="", marker="o", color="orange")
    plt.plot(epoch_range, history['train_accuracy'], markevery=[np.argmax(history['train_accuracy'])], 
             ls="", marker="o", color="blue")
    
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.savefig(filepath + "LearningAccuracy.svg", bbox_inches="tight", format="svg")
    plt.show()
    plt.clf()
    
    # 绘制训练和验证损失
    plt.plot(epoch_range, history['train_loss'])
    plt.plot(epoch_range, history['val_loss'])
    
    # 标记最小损失点
    plt.plot(epoch_range, history['train_loss'], markevery=[np.argmin(history['train_loss'])], 
             ls="", marker="o", color="blue")
    plt.plot(epoch_range, history['val_loss'], markevery=[np.argmin(history['val_loss'])], 
             ls="", marker="o", color="orange")
    
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(filepath + "ModelLoss.svg", bbox_inches="tight", format="svg")
    plt.show()
    plt.clf()


def round_number(to_round_nb):
    """四舍五入数字并转换为百分比"""
    return round(to_round_nb, 4) * 100


def extract_intermediate_model_from_base_model(model, layer_idx=-4):
    """从模型中提取中间特征的钩子机制
    
    Args:
        model: 基础模型
        layer_idx: 需要提取特征的层索引或名称
        
    Returns:
        一个接收输入并返回特定层输出的函数
    """
    class IntermediateModel(torch.nn.Module):
        def __init__(self, base_model, target_layer):
            super().__init__()
            self.base_model = base_model
            self.target_layer = target_layer
            self.features = None
            
            # 用于临时保存特征的钩子函数
            def hook_fn(module, input, output):
                self.features = output
            
            # 找到目标层并注册钩子
            if isinstance(target_layer, int):
                # 如果是索引，找到相应位置的层
                for i, (name, module) in enumerate(model.named_modules()):
                    if i == target_layer:
                        self.hook = module.register_forward_hook(hook_fn)
                        break
            else:
                # 如果是名称，通过名称查找层
                for name, module in model.named_modules():
                    if name == target_layer:
                        self.hook = module.register_forward_hook(hook_fn)
                        break
        
        def forward(self, x):
            # 运行整个模型但只返回目标层的输出
            self.base_model(x)
            features = self.features
            self.features = None  # 清空以避免内存泄漏
            return features
    
    return IntermediateModel(model, layer_idx)