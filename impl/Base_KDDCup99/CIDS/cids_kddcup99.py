import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import time
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------- 全局配置 ----------------------
# 特征名
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]
# 攻击类型映射
ATTACK_MAPPING = {
    1: 'Normal',
    2: 'Buffer Overflow (U2R)',
    3: 'Loadmodule (U2R)',
    4: 'Perl (U2R)',
    5: 'Neptune (DoS)',
    6: 'Smurf (DoS)',
    7: 'Guess Password (R2L)',
    8: 'Pod (DoS)',
    9: 'Teardrop (DoS)',
    10: 'Port sweep (Probe)',
    11: 'IP sweep (Probe)',
    12: 'Land (DoS)',
    13: 'Ftp write (R2L)',
    14: 'Backdoor (DoS)',
    15: 'IMAP (R2L)',
    16: 'Satan (Probe)',
    17: 'PHF (R2L)',
    18: 'NMAP (Probe)',
    19: 'Multi-hop (R2L)',
    20: 'Warez-master (R2L)',
    21: 'Warez-client (R2L)',
    22: 'Spy (R2L)',
    23: 'Root-kit (U2R)'
}
# 模型保存路径
MODEL_SAVE_DIR = 'cids_models/'
# 数据集路径（需根据本地文件修改）
TRAIN_DATA_PATH = "../../../DataSet/kddcup99/kddcup.data_10_percent"  # KDD Cup 1999 10%样本
TEST_DATA_PATH = "../../../DataSet/kddcup99/corrected"  # 测试用（实际应使用独立测试集）


# ---------------------- 1. 数据预处理模块 ----------------------
def preprocess_data(data_path, is_train=True, scalers=None, encoders=None):
    """
    数据预处理：清洗→编码→缩放→标签映射
    :param data_path: 数据集路径
    :param is_train: 是否为训练集（训练集需拟合scaler/encoder，测试集复用）
    :param scalers: 已训练的缩放器（测试集用）
    :param encoders: 已训练的编码器（测试集用）
    :return: X（特征）、y（标签）、y_binary（二分类标签）、scalers、encoders
    """
    # 1. 加载数据
    df = pd.read_csv(data_path, header=None, names=FEATURE_NAMES + ['label'])

    # 2. 标签处理（多分类：1=正常，2-23=攻击类型；二分类：0=正常，1=异常）
    def map_label(label):
        if label == 'normal.':
            return 1  # 正常标签为1
        elif 'buffer_overflow.' in label:
            return 2
        elif 'loadmodule.' in label:
            return 3
        elif 'perl.' in label:
            return 4
        elif 'neptune.' in label:
            return 5
        elif 'smurf.' in label:
            return 6
        elif 'guess_passwd.' in label:
            return 7
        elif 'pod.' in label:
            return 8
        elif 'teardrop.' in label:
            return 9
        elif 'portsweep.' in label:
            return 10
        elif 'ipsweep.' in label:
            return 11
        elif 'land.' in label:
            return 12
        elif 'ftp_write.' in label:
            return 13
        elif 'backdoor.' in label:
            return 14
        elif 'imap.' in label:
            return 15
        elif 'satan.' in label:
            return 16
        elif 'phf.' in label:
            return 17
        elif 'nmap.' in label:
            return 18
        elif 'multihop.' in label:
            return 19
        elif 'warezmaster.' in label:
            return 20
        elif 'warezclient.' in label:
            return 21
        elif 'spy.' in label:
            return 22
        elif 'rootkit.' in label:
            return 23
        else:
            return -1  # 未知类型

    df['label_multi'] = df['label'].apply(map_label)
    df['label_binary'] = df['label_multi'].apply(lambda x: 0 if x == 1 else 1)

    # 3. 数据清洗
    df = df.dropna(axis=0, how='any')
    df = df[df['label_multi'] != -1]  # 过滤未知类型

    # 4. 离散特征编码
    discrete_cols = ['protocol_type', 'service', 'flag']
    if is_train:
        encoders = {col: LabelEncoder() for col in discrete_cols}
        for col in discrete_cols:
            df[col] = encoders[col].fit_transform(df[col])
    else:
        for col in discrete_cols:
            known_classes = set(encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else encoders[col].classes_[0])
            df[col] = encoders[col].transform(df[col])

    # 5. 特征与标签分离
    X = df[FEATURE_NAMES].values
    y_multi = df['label_multi'].values
    y_binary = df['label_binary'].values

    # 6. 数据缩放
    if is_train:
        scalers = StandardScaler()
        X_scaled = scalers.fit_transform(X)
    else:
        X_scaled = scalers.transform(X)

    return X_scaled, y_multi, y_binary, scalers, encoders


# ---------------------- 2. DICS模块（基于距离的入侵分类） ----------------------
class DICS:
    def __init__(self):
        self.standard_profiles = None
        self.scaler = None

    def fit(self, X_anomaly, y_anomaly_multi):
        """
        训练DICS：生成22类攻击的标准轮廓（用特征中位数）
        :param X_anomaly: 异常样本特征（仅攻击样本）
        :param y_anomaly_multi: 异常样本多分类标签（2-23）
        """
        # 1. 缩放异常样本
        self.scaler = StandardScaler()
        X_anomaly_scaled = self.scaler.fit_transform(X_anomaly)

        # 2. 按攻击类型分组，计算每类的中位数（标准轮廓）
        self.standard_profiles = {}
        attack_labels = [i for i in range(2, 24)]  # 2-23类攻击
        for label in attack_labels:
            X_label = X_anomaly_scaled[y_anomaly_multi == label]
            if len(X_label) == 0:
                continue
            profile = np.median(X_label, axis=0)
            self.standard_profiles[label] = profile

    def predict(self, X_test):
        """
        预测攻击类型：计算测试样本与各轮廓的欧氏距离，取最小距离对应的类型
        :param X_test: 测试样本特征（已通过OC-SVM判定为异常）
        :return: 预测的攻击类型标签（2-23）
        """
        if self.standard_profiles is None:
            raise ValueError("DICS未训练，请先调用fit()方法")

        # 1. 缩放测试样本
        X_test_scaled = self.scaler.transform(X_test)

        # 2. 计算与每个标准轮廓的欧氏距离
        predictions = []
        for x in X_test_scaled:
            min_dist = float('inf')
            pred_label = -1
            for label, profile in self.standard_profiles.items():
                dist = np.sqrt(np.sum((x - profile) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    pred_label = label
            predictions.append(pred_label)
        return np.array(predictions)


# ---------------------- 3. CIDS级联系统核心类 ----------------------
class CascadedIDS:
    def __init__(self):
        # 三级模块初始化
        self.sids = DecisionTreeClassifier(random_state=42)
        self.aids = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        self.dics = DICS()

        self.scalers = None
        self.encoders = None

        # 训练状态标记
        self.is_trained = False

    def train(self, train_data_path):
        """
        训练CIDS系统
        :param train_data_path: 训练数据集路径
        """
        print("=" * 60)
        print("开始训练CIDS级联入侵检测系统")
        print("=" * 60)
        start_total_time = time.time()

        # ---------------------- 步骤1：数据预处理 ----------------------
        print("\n[步骤1/4] 数据预处理...")
        X_train, y_train_multi, y_train_binary, self.scalers, self.encoders = preprocess_data(
            data_path=train_data_path, is_train=True
        )
        print(f"预处理完成：训练样本数={len(X_train)}，特征数={X_train.shape[1]}")

        # ---------------------- 步骤2：训练SIDS（决策树，识别已知攻击） ----------------------
        print("\n[步骤2/4] 训练SIDS（决策树）...")
        start_sids_time = time.time()
        # SIDS用全量数据训练（多分类：1=正常，2-23=攻击）
        self.sids.fit(X_train, y_train_multi)
        # 验证SIDS性能
        y_sids_pred = self.sids.predict(X_train)
        sids_acc = accuracy_score(y_train_multi, y_sids_pred)
        sids_f1 = f1_score(y_train_multi, y_sids_pred, average='weighted')
        sids_time = time.time() - start_sids_time
        print(f"SIDS训练完成：耗时={sids_time:.2f}s，准确率={sids_acc:.4f}，F1分数={sids_f1:.4f}")

        # ---------------------- 步骤3：训练AIDS（OC-SVM，检测零日攻击） ----------------------
        print("\n[步骤3/4] 训练AIDS（OC-SVM）...")
        start_aids_time = time.time()
        # OC-SVM仅用正常样本训练
        X_train_normal = X_train[y_train_multi == 1]  # 筛选正常样本
        self.aids.fit(X_train_normal)
        # 验证AIDS性能（二分类：正常=1，异常=-1）
        y_aids_pred = self.aids.predict(X_train)
        # 映射为二分类标签（1→0=正常，-1→1=异常）
        y_aids_pred_binary = np.where(y_aids_pred == 1, 0, 1)
        aids_acc = accuracy_score(y_train_binary, y_aids_pred_binary)
        aids_recall = recall_score(y_train_binary, y_aids_pred_binary, average='binary')
        aids_time = time.time() - start_aids_time
        print(f"AIDS训练完成：耗时={aids_time:.2f}s，准确率={aids_acc:.4f}，召回率={aids_recall:.4f}")

        # ---------------------- 步骤4：训练DICS（距离分类，确定攻击类型） ----------------------
        print("\n[步骤4/4] 训练DICS（距离分类）...")
        start_dics_time = time.time()
        # DICS仅用异常样本训练（筛选2-23类攻击样本）
        X_train_anomaly = X_train[y_train_multi >= 2]
        y_train_anomaly_multi = y_train_multi[y_train_multi >= 2]
        self.dics.fit(X_train_anomaly, y_train_anomaly_multi)
        # 验证DICS性能
        y_dics_pred = self.dics.predict(X_train_anomaly)
        dics_acc = accuracy_score(y_train_anomaly_multi, y_dics_pred)
        dics_f1 = f1_score(y_train_anomaly_multi, y_dics_pred, average='weighted')
        dics_time = time.time() - start_dics_time
        print(f"DICS训练完成：耗时={dics_time:.2f}s，准确率={dics_acc:.4f}，F1分数={dics_f1:.4f}")

        self.is_trained = True
        total_time = time.time() - start_total_time
        print("\n" + "=" * 60)
        print(f"CIDS训练全部完成：总耗时={total_time:.2f}s")
        print("=" * 60)

    def predict(self, X_test):
        """
        CIDS推理预测
        :param X_test: 测试样本特征（已预处理）
        :return: 最终预测结果（多分类标签：1=正常，2-23=攻击类型）
        """
        if not self.is_trained:
            raise ValueError("CIDS未训练，请先调用train()方法")

        # ---------------------- 第一级：SIDS筛选已知攻击 ----------------------
        y_sids_pred = self.sids.predict(X_test)
        mask_sids_attack = y_sids_pred >= 2  # SIDS直接识别的攻击
        mask_sids_normal = y_sids_pred == 1  # 待AIDS验证的正常样本

        # ---------------------- 第二级：AIDS验证零日攻击 ----------------------
        X_aids_test = X_test[mask_sids_normal]
        if len(X_aids_test) == 0:
            y_aids_pred = np.array([])
            mask_aids_normal = np.array([])
            mask_aids_anomaly = np.array([])
        else:
            y_aids_pred = self.aids.predict(X_aids_test)
            mask_aids_normal = y_aids_pred == 1  # 正常
            mask_aids_anomaly = y_aids_pred == -1  # 需DICS分类

        # ---------------------- 第三级：DICS分类零日攻击类型 ----------------------
        X_dics_test = X_aids_test[mask_aids_anomaly] if len(X_aids_test) > 0 else np.array([])
        y_dics_pred = self.dics.predict(X_dics_test) if len(X_dics_test) > 0 else np.array([])

        # ---------------------- 整合最终结果 ----------------------
        final_pred = np.zeros_like(y_sids_pred)
        final_pred[mask_sids_attack] = y_sids_pred[mask_sids_attack]
        if len(mask_sids_normal) > 0 and len(mask_aids_normal) > 0:
            final_pred[mask_sids_normal] = 1  # 默认正常
            if len(y_dics_pred) > 0:
                final_pred[mask_sids_normal][mask_aids_anomaly] = y_dics_pred

        return final_pred

    def evaluate(self, test_data_path):
        """
        评估CIDS性能（准确率、精确率、召回率、F1分数）
        :param test_data_path: 测试数据集路径
        :return: 性能指标字典
        """
        if not self.is_trained:
            raise ValueError("CIDS未训练，请先调用train()方法")

        print("\n" + "=" * 60)
        print("开始评估CIDS性能")
        print("=" * 60)

        # 1. 预处理测试数据
        X_test, y_test_multi, y_test_binary, _, _ = preprocess_data(
            data_path=test_data_path, is_train=False,
            scalers=self.scalers, encoders=self.encoders
        )
        print(f"测试数据加载完成：样本数={len(X_test)}")

        # 2. CIDS预测
        start_pred_time = time.time()
        y_pred = self.predict(X_test)
        pred_time = time.time() - start_pred_time
        print(f"预测完成：耗时={pred_time:.2f}s")

        # 3. 计算性能指标
        # 多分类指标（评估攻击类型识别准确性）
        multi_acc = accuracy_score(y_test_multi, y_pred)
        multi_precision = precision_score(y_test_multi, y_pred, average='weighted', zero_division=0)
        multi_recall = recall_score(y_test_multi, y_pred, average='weighted', zero_division=0)
        multi_f1 = f1_score(y_test_multi, y_pred, average='weighted', zero_division=0)

        # 二分类指标（评估正常/异常识别准确性）
        y_pred_binary = np.where(y_pred == 1, 0, 1)
        binary_acc = accuracy_score(y_test_binary, y_pred_binary)
        binary_precision = precision_score(y_test_binary, y_pred_binary, average='binary', zero_division=0)
        binary_recall = recall_score(y_test_binary, y_pred_binary, average='binary', zero_division=0)
        binary_f1 = f1_score(y_test_binary, y_pred_binary, average='binary', zero_division=0)

        # 4. 输出结果
        print("\n【多分类性能（攻击类型识别）】")
        print(f"准确率：{multi_acc:.4f}")
        print(f"精确率：{multi_precision:.4f}")
        print(f"召回率：{multi_recall:.4f}")
        print(f"F1分数：{multi_f1:.4f}")

        print("\n【二分类性能（正常/异常识别）】")
        print(f"准确率：{binary_acc:.4f}")
        print(f"精确率：{binary_precision:.4f}")
        print(f"召回率：{binary_recall:.4f}")
        print(f"F1分数：{binary_f1:.4f}")

        # 5. 混淆矩阵可视化
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Zen Hei', 'Heiti TC']
        plt.rcParams['axes.unicode_minus'] = False

        cm = confusion_matrix(y_test_binary, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常（0）', '异常（1）'],
                    yticklabels=['正常（0）', '异常（1）'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('CIDS二分类混淆矩阵')
        plt.show()

        metrics = {
            'multi_class': {'accuracy': multi_acc, 'precision': multi_precision, 'recall': multi_recall,
                            'f1': multi_f1},
            'binary_class': {'accuracy': binary_acc, 'precision': binary_precision, 'recall': binary_recall,
                             'f1': binary_f1},
            'prediction_time': pred_time
        }
        return metrics

    def save_model(self, save_dir=MODEL_SAVE_DIR):
        """
        保存CIDS模型（含所有三级模块、scaler、encoder）
        :param save_dir: 模型保存目录
        """
        if not self.is_trained:
            raise ValueError("CIDS未训练，无法保存模型")

        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存各组件
        with open(f"{save_dir}/sids.pkl", 'wb') as f:
            pickle.dump(self.sids, f)
        with open(f"{save_dir}/aids.pkl", 'wb') as f:
            pickle.dump(self.aids, f)
        with open(f"{save_dir}/dics.pkl", 'wb') as f:
            pickle.dump(self.dics, f)
        with open(f"{save_dir}/scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        with open(f"{save_dir}/encoders.pkl", 'wb') as f:
            pickle.dump(self.encoders, f)

        print(f"\nCIDS模型已保存至：{save_dir}")

    @classmethod
    def load_model(cls, load_dir=MODEL_SAVE_DIR):
        """
        加载已保存的CIDS模型
        :param load_dir: 模型加载目录
        :return: 加载完成的CIDS实例
        """
        import os
        required_files = ['sids.pkl', 'aids.pkl', 'dics.pkl', 'scalers.pkl', 'encoders.pkl']
        for file in required_files:
            if not os.path.exists(f"{load_dir}/{file}"):
                raise FileNotFoundError(f"模型文件缺失：{load_dir}/{file}")

        cids = cls()

        with open(f"{load_dir}/sids.pkl", 'rb') as f:
            cids.sids = pickle.load(f)
        with open(f"{load_dir}/aids.pkl", 'rb') as f:
            cids.aids = pickle.load(f)
        with open(f"{load_dir}/dics.pkl", 'rb') as f:
            cids.dics = pickle.load(f)
        with open(f"{load_dir}/scalers.pkl", 'rb') as f:
            cids.scalers = pickle.load(f)
        with open(f"{load_dir}/encoders.pkl", 'rb') as f:
            cids.encoders = pickle.load(f)

        cids.is_trained = True
        print(f"CIDS模型已从：{load_dir} 加载完成")
        return cids


# ---------------------- 4. 测试用主函数 ----------------------
def main():
    # 初始化CIDS
    cids = CascadedIDS()

    # 1. 训练模型
    cids.train(train_data_path=TRAIN_DATA_PATH)

    # 2. 评估模型
    metrics = cids.evaluate(test_data_path=TEST_DATA_PATH)

    # 3. 保存模型
    cids.save_model()

    # 4. 加载模型并重新预测
    print("\n" + "=" * 60)
    print("开始验证模型加载功能。。。")
    print("=" * 60)
    cids_loaded = CascadedIDS.load_model()
    # 加载测试数据并预测
    X_test, y_test_multi, _, _, _ = preprocess_data(
        data_path=TEST_DATA_PATH, is_train=False,
        scalers=cids_loaded.scalers, encoders=cids_loaded.encoders
    )
    y_pred_loaded = cids_loaded.predict(X_test[:10])
    print(f"加载模型后，预测前10个样本的结果：")
    for i in range(10):
        print(f"样本{i + 1}：真实标签={ATTACK_MAPPING[y_test_multi[i]]}，预测标签={ATTACK_MAPPING[y_pred_loaded[i]]}")


if __name__ == "__main__":
    main()