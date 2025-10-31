import pandas as pd
from cids_kddcup99 import CascadedIDS, FEATURE_NAMES, ATTACK_MAPPING


def preprocess_single_data(data, scalers, encoders):
    """
    预处理单条/多条新数据（适配CIDS输入）
    :param data: 新数据
    :param scalers: 加载的scalers
    :param encoders: 加载的encoders
    :return: 预处理后的特征矩阵X
    """
    # 1. 转换为DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # 2. 检查特征完整性
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        raise ValueError(f"缺失特征：{missing_features}")

    # 3. 离散特征编码
    discrete_cols = ['protocol_type', 'service', 'flag']
    for col in discrete_cols:
        df[col] = encoders[col].transform(df[col])

    # 4. 特征缩放
    X = df[FEATURE_NAMES].values
    X_scaled = scalers.transform(X)

    return X_scaled


def cids_infer_new_data(new_data, model_dir='./cids_models/'):
    """
    用已加载的CIDS模型预测新数据
    :param new_data: 新数据
    :param model_dir: 模型目录
    :return: 预测结果
    """
    # 1. 加载CIDS模型
    cids = CascadedIDS.load_model(load_dir=model_dir)

    # 2. 预处理新数据
    X_scaled = preprocess_single_data(new_data, cids.scalers, cids.encoders)

    # 3. 模型预测
    y_pred = cids.predict(X_scaled)

    # 4. 结果解析（添加中文解释）
    results = []
    for i in range(len(y_pred)):
        label = y_pred[i]
        result = {
            '样本索引': i,
            '预测标签': label,
            '攻击类型': ATTACK_MAPPING[label],
            '风险等级': '无风险' if label == 1 else '高风险'
        }
        results.append(result)

    # 转换为DataFrame便于查看
    results_df = pd.DataFrame(results)
    return results_df


# ---------------------- 测试：对新数据进行推理 ----------------------
if __name__ == "__main__":
    # 示例1：正常流量数据（模拟）
    normal_data = {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
        'src_bytes': 181, 'dst_bytes': 5450, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 1,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
        'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0,
        'is_host_login': 0, 'is_guest_login': 0, 'count': 8, 'srv_count': 8,
        'serror_rate': 0.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 255, 'dst_host_srv_count': 255, 'dst_host_same_srv_rate': 0.1,
        'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
    }

    # 示例2：DoS攻击流量数据（模拟Smurf攻击）
    attack_data = {
        'duration': 0, 'protocol_type': 'icmp', 'service': 'echo', 'flag': 'REJ',
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
        'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0,
        'is_host_login': 0, 'is_guest_login': 0, 'count': 255, 'srv_count': 255,
        'serror_rate': 0.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 1, 'dst_host_srv_count': 1, 'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
    }

    # 合并测试数据
    test_data = pd.DataFrame([normal_data, attack_data])

    # 调用推理函数
    print("=" * 60)
    print("CIDS模型推理结果")
    print("=" * 60)
    results = cids_infer_new_data(test_data)
    print(results.to_string(index=False))