import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import time
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# 特征名称定义
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

# ---------------------- 全局配置 ----------------------
MODEL_SAVE_DIR = './cids_nsl_kdd_models/'


# ---------------------- 数据预处理模块 ----------------------
def preprocess_data(data_path, is_train=True, scalers=None, encoders=None):
    df = pd.read_csv(data_path, header=None, names=FEATURE_NAMES + ['label', 'difficulty'])
    df = df.drop(columns=['difficulty'])

    def map_label(label):
        if label == 'normal':
            return 1
        elif label == 'buffer_overflow':
            return 2
        elif label == 'loadmodule':
            return 3
        elif label == 'perl':
            return 4
        elif label == 'neptune':
            return 5
        elif label == 'smurf':
            return 6
        elif label == 'guess_passwd':
            return 7
        elif label == 'pod':
            return 8
        elif label == 'teardrop':
            return 9
        elif label == 'portsweep':
            return 10
        elif label == 'ipsweep':
            return 11
        elif label == 'land':
            return 12
        elif label == 'ftp_write':
            return 13
        elif label == 'backdoor':
            return 14
        elif label == 'imap':
            return 15
        elif label == 'satan':
            return 16
        elif label == 'phf':
            return 17
        elif label == 'nmap':
            return 18
        elif label == 'multihop':
            return 19
        elif label == 'warezmaster':
            return 20
        elif label == 'warezclient':
            return 21
        elif label == 'spy':
            return 22
        elif label == 'rootkit':
            return 23
        else:
            return -1

    df['label_multi'] = df['label'].apply(map_label)
    df['label_binary'] = df['label_multi'].apply(lambda x: 0 if x == 1 else 1)

    df = df.dropna(axis=0, how='any')
    df = df[df['label_multi'] != -1]

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

    X = df[FEATURE_NAMES].values
    y_multi = df['label_multi'].values
    y_binary = df['label_binary'].values

    if is_train:
        scalers = StandardScaler()
        X_scaled = scalers.fit_transform(X)
    else:
        X_scaled = scalers.transform(X)

    return X_scaled, y_multi, y_binary, scalers, encoders


class DICS:
    def __init__(self):
        self.standard_profiles = None
        self.scaler = None

    def fit(self, X_anomaly, y_anomaly_multi):
        self.scaler = StandardScaler()
        X_anomaly_scaled = self.scaler.fit_transform(X_anomaly)
        self.standard_profiles = {}
        attack_labels = range(2, 24)
        for label in attack_labels:
            X_label = X_anomaly_scaled[y_anomaly_multi == label]
            if len(X_label) == 0:
                continue
            profile = np.median(X_label, axis=0)
            self.standard_profiles[label] = profile

    def predict(self, X_test):
        if self.standard_profiles is None:
            raise ValueError("DICS未训练，请先调用fit()方法")
        X_test_scaled = self.scaler.transform(X_test)
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

# ---------------------- CIDS级联系统核心类 ----------------------
class CascadedIDS:
    def __init__(self):
        self.sids = DecisionTreeClassifier(random_state=42)
        self.aids = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        self.dics = DICS()
        self.scalers = None
        self.encoders = None
        self.is_trained = False

    def train(self, train_data_path):
        print("=" * 60)
        print("开始训练基于NSL-KDD的CIDS级联入侵检测系统")
        print("=" * 60)
        start_total_time = time.time()

        print("\n[步骤1/4] 预处理NSL-KDD训练数据...")
        X_train, y_train_multi, y_train_binary, self.scalers, self.encoders = preprocess_data(
            data_path=train_data_path, is_train=True
        )
        print(f"预处理完成：训练样本数={len(X_train)}，特征数={X_train.shape[1]}")

        print("\n[步骤2/4] 训练SIDS（决策树）...")
        start_sids_time = time.time()
        self.sids.fit(X_train, y_train_multi)
        y_sids_pred = self.sids.predict(X_train)
        sids_acc = accuracy_score(y_train_multi, y_sids_pred)
        sids_f1 = f1_score(y_train_multi, y_sids_pred, average='weighted', zero_division=0)
        sids_time = time.time() - start_sids_time
        print(f"SIDS训练完成：耗时={sids_time:.2f}s，准确率={sids_acc:.4f}，F1={sids_f1:.4f}")

        print("\n[步骤3/4] 训练AIDS（OC-SVM）...")
        start_aids_time = time.time()
        X_train_normal = X_train[y_train_multi == 1]
        self.aids.fit(X_train_normal)
        y_aids_pred = self.aids.predict(X_train)
        y_aids_pred_binary = np.where(y_aids_pred == 1, 0, 1)
        aids_acc = accuracy_score(y_train_binary, y_aids_pred_binary)
        aids_recall = recall_score(y_train_binary, y_aids_pred_binary, average='binary', zero_division=0)
        aids_time = time.time() - start_aids_time
        print(f"AIDS训练完成：耗时={aids_time:.2f}s，准确率={aids_acc:.4f}，召回率={aids_recall:.4f}")

        print("\n[步骤4/4] 训练DICS（距离-based分类）...")
        start_dics_time = time.time()
        X_train_anomaly = X_train[y_train_multi >= 2]
        y_train_anomaly_multi = y_train_multi[y_train_multi >= 2]
        self.dics.fit(X_train_anomaly, y_train_anomaly_multi)
        y_dics_pred = self.dics.predict(X_train_anomaly)
        dics_acc = accuracy_score(y_train_anomaly_multi, y_dics_pred)
        dics_f1 = f1_score(y_train_anomaly_multi, y_dics_pred, average='weighted', zero_division=0)
        dics_time = time.time() - start_dics_time
        print(f"DICS训练完成：耗时={dics_time:.2f}s，准确率={dics_acc:.4f}，F1={dics_f1:.4f}")

        self.is_trained = True
        total_time = time.time() - start_total_time
        print("\n" + "=" * 60)
        print(f"CIDS训练全部完成：总耗时={total_time:.2f}s")
        print("=" * 60)

    def predict(self, X_test):
        if not self.is_trained:
            raise ValueError("CIDS未训练，请先调用train()方法")

        # ---------------------- 第一级：SIDS筛选已知攻击 ----------------------
        y_sids_pred = self.sids.predict(X_test)
        mask_sids_attack = y_sids_pred >= 2  # SIDS判定的攻击样本
        mask_sids_normal = y_sids_pred == 1  # 需AIDS验证的正常样本

        # ---------------------- 第二级：AIDS验证零日攻击 ----------------------
        X_aids_test = X_test[mask_sids_normal]
        if len(X_aids_test) == 0:
            y_aids_pred = np.array([])
            mask_aids_normal = np.array([])
            mask_aids_anomaly = np.array([])
        else:
            y_aids_pred = self.aids.predict(X_aids_test)
            mask_aids_normal = y_aids_pred == 1  # AIDS判定的正常样本
            mask_aids_anomaly = y_aids_pred == -1  # AIDS判定的异常样本（需DICS分类）

        # ---------------------- 第三级：DICS分类零日攻击类型 ----------------------
        X_dics_test = X_aids_test[mask_aids_anomaly] if len(X_aids_test) > 0 else np.array([])
        y_dics_pred = self.dics.predict(X_dics_test) if len(X_dics_test) > 0 else np.array([])

        # ---------------------- 整合最终结果 ----------------------
        final_pred = np.zeros_like(y_sids_pred)
        final_pred[mask_sids_attack] = y_sids_pred[mask_sids_attack]
        if len(mask_sids_normal) > 0 and len(mask_aids_normal) > 0:
            final_pred[mask_sids_normal] = 1
            if len(y_dics_pred) > 0:
                final_pred[mask_sids_normal][mask_aids_anomaly] = y_dics_pred

        return final_pred

    def evaluate(self, test_data_path):
        if not self.is_trained:
            raise ValueError("CIDS未训练，请先调用train()方法")

        print("\n" + "=" * 60)
        print("基于NSL-KDD测试集评估CIDS性能")
        print("=" * 60)

        X_test, y_test_multi, y_test_binary, _, _ = preprocess_data(
            data_path=test_data_path, is_train=False,
            scalers=self.scalers, encoders=self.encoders
        )
        print(f"测试数据加载完成：样本数={len(X_test)}")

        start_pred_time = time.time()
        y_pred = self.predict(X_test)
        pred_time = time.time() - start_pred_time
        print(f"预测完成：耗时={pred_time:.2f}s")

        multi_metrics = {
            'accuracy': accuracy_score(y_test_multi, y_pred),
            'precision': precision_score(y_test_multi, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_multi, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test_multi, y_pred, average='weighted', zero_division=0)
        }

        y_pred_binary = np.where(y_pred == 1, 0, 1)
        binary_metrics = {
            'accuracy': accuracy_score(y_test_binary, y_pred_binary),
            'precision': precision_score(y_test_binary, y_pred_binary, average='binary', zero_division=0),
            'recall': recall_score(y_test_binary, y_pred_binary, average='binary', zero_division=0),
            'f1': f1_score(y_test_binary, y_pred_binary, average='binary', zero_division=0)
        }

        print("\n【多分类性能（攻击类型识别）】")
        for k, v in multi_metrics.items():
            print(f"{k.capitalize()}：{v:.4f}")
        print("\n【二分类性能（正常/异常识别）】")
        for k, v in binary_metrics.items():
            print(f"{k.capitalize()}：{v:.4f}")

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 绘制混淆矩阵
        cm = confusion_matrix(y_test_binary, y_pred_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常（0）', '异常（1）'],
                    yticklabels=['正常（0）', '异常（1）'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('CIDS二分类混淆矩阵（NSL-KDD测试集）')
        plt.show()

        return {
            'multi_class': multi_metrics,
            'binary_class': binary_metrics,
            'prediction_time': pred_time
        }

    def save_model(self, save_dir=MODEL_SAVE_DIR):
        if not self.is_trained:
            raise ValueError("CIDS未训练，无法保存模型")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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
    def load_model(cls, save_dir=MODEL_SAVE_DIR):
        """加载已保存的模型，用于验证模型可用性"""
        print("\n" + "=" * 60)
        print(f"从 {save_dir} 加载CIDS模型...")
        print("=" * 60)

        required_files = ['sids.pkl', 'aids.pkl', 'dics.pkl', 'scalers.pkl', 'encoders.pkl']
        for file in required_files:
            file_path = f"{save_dir}/{file}"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"模型文件缺失：{file_path}")

        cids = cls()

        with open(f"{save_dir}/sids.pkl", 'rb') as f:
            cids.sids = pickle.load(f)
        with open(f"{save_dir}/aids.pkl", 'rb') as f:
            cids.aids = pickle.load(f)
        with open(f"{save_dir}/dics.pkl", 'rb') as f:
            cids.dics = pickle.load(f)
        with open(f"{save_dir}/scalers.pkl", 'rb') as f:
            cids.scalers = pickle.load(f)
        with open(f"{save_dir}/encoders.pkl", 'rb') as f:
            cids.encoders = pickle.load(f)

        cids.is_trained = True
        print("模型加载完成，已准备好进行预测")
        return cids


# ---------------------- 验证模型加载功能的函数 ----------------------
def validate_loaded_model(test_data_path, model_dir=MODEL_SAVE_DIR):
    """验证加载的模型是否可正常预测"""
    print("\n" + "=" * 60)
    print("开始验证加载的CIDS模型...")
    print("=" * 60)

    # 1. 加载测试数据（预处理）
    print("\n[1/3] 预处理测试数据...")
    with open(f"{model_dir}/encoders.pkl", 'rb') as f:
        encoders = pickle.load(f)
    with open(f"{model_dir}/scalers.pkl", 'rb') as f:
        scalers = pickle.load(f)

    X_test, y_test_multi, y_test_binary, _, _ = preprocess_data(
        data_path=test_data_path,
        is_train=False,
        scalers=scalers,
        encoders=encoders
    )
    print(f"测试数据预处理完成：样本数={len(X_test)}")

    # 2. 加载模型
    print("\n[2/3] 加载模型...")
    cids_loaded = CascadedIDS.load_model(model_dir)

    # 3. 用加载的模型进行预测并验证
    print("\n[3/3] 用加载的模型进行预测...")
    start_time = time.time()
    y_pred = cids_loaded.predict(X_test)
    pred_time = time.time() - start_time
    print(f"预测完成：耗时={pred_time:.2f}s，预测样本数={len(y_pred)}")

    # 计算关键指标
    binary_acc = accuracy_score(y_test_binary, np.where(y_pred == 1, 0, 1))
    multi_f1 = f1_score(y_test_multi, y_pred, average='weighted', zero_division=0)
    print("\n【加载模型的验证指标】")
    print(f"二分类准确率：{binary_acc:.4f}")
    print(f"多分类F1分数：{multi_f1:.4f}")

    # 随机抽取5个样本展示预测结果
    print("\n【随机抽取5个样本的预测结果】")
    sample_indices = np.random.choice(len(y_pred), 5, replace=False)
    for idx in sample_indices:
        true_label = ATTACK_MAPPING.get(y_test_multi[idx], "未知")
        pred_label = ATTACK_MAPPING.get(y_pred[idx], "未知")
        print(f"样本{idx}：真实={true_label}，预测={pred_label}，{'一致' if true_label == pred_label else '不一致'}")

    return {
        'binary_accuracy': binary_acc,
        'multi_f1': multi_f1,
        'prediction_time': pred_time
    }


# ---------------------- 训练主函数（含模型验证） ----------------------
def main():
    # 数据集路径（请根据实际情况修改）
    TRAIN_DATA_PATH = "../../../DataSet/NSL_KDD/KDDTrain+.txt"
    TEST_DATA_PATH = "../../../DataSet/NSL_KDD/KDDTest+.txt"

    # 1. 初始化并训练模型
    cids = CascadedIDS()
    cids.train(train_data_path=TRAIN_DATA_PATH)

    # 2. 评估模型
    train_eval = cids.evaluate(test_data_path=TEST_DATA_PATH)

    # 3. 保存模型
    cids.save_model()

    # 4. 验证模型加载功能
    loaded_model_eval = validate_loaded_model(
        test_data_path=TEST_DATA_PATH,
        model_dir=MODEL_SAVE_DIR
    )

    # 5. 对比训练时和加载后的关键指标
    print("\n" + "=" * 60)
    print("模型训练时与加载后的指标对比")
    print("=" * 60)
    print(
        f"二分类准确率：训练时={train_eval['binary_class']['accuracy']:.4f}，加载后={loaded_model_eval['binary_accuracy']:.4f}")
    print(f"多分类F1分数：训练时={train_eval['multi_class']['f1']:.4f}，加载后={loaded_model_eval['multi_f1']:.4f}")
    print("\n若指标一致，说明模型加载功能正常")


if __name__ == "__main__":
    main()