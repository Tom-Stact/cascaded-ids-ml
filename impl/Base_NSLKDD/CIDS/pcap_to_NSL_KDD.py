import os
import pickle

import pandas as pd
from scapy.all import rdpcap
from scapy.layers.dns import DNS
from scapy.layers.http import HTTPRequest, HTTPResponse
from scapy.layers.inet import IP, TCP, UDP

from cids_nsl_kdd import FEATURE_NAMES

# 协议类型映射
PROTOCOL_PROTO_MAP = {1: 'icmp', 6: 'tcp', 17: 'udp'}

TCP_FLAG_MAP = {
    0x01: 'F',  # 仅Fin
    0x02: 'S',  # 仅Syn
    0x04: 'R',  # 仅Rst
    0x08: 'P',  # 仅Push
    0x10: 'A',  # 仅Ack
    0x20: 'U',  # 仅Urgent
    0x03: 'SF',  # Syn+Fin
    0x05: 'RF',  # Rst+Fin
    0x09: 'PF',  # Push+Fin
    0x11: 'FA',  # Fin+Ack
    0x12: 'SA',  # Syn+Ack（三次握手第二步）
    0x14: 'RA',  # Rst+Ack（复位连接）
    0x18: 'PA',  # Push+Ack（数据传输常用）
    0x22: 'SU',  # Syn+Urgent
    0x24: 'RU',  # Rst+Urgent
    0x28: 'PU',  # Push+Urgent
    0x30: 'AU',  # Ack+Urgent
    0x19: 'PAF',  # Push+Ack+Fin
    0x1A: 'SAF',  # Syn+Ack+Fin
    0x1C: 'RAF',  # Rst+Ack+Fin
}

ACTUAL_FLAG_TO_NSLKDD = {
    'PA': 'SF',
    'SA': 'S1',
    'FA': 'SF',
    'RA': 'RSTR',
    'A': 'SF',
    'F': 'SF',
    'R': 'RSTR',
    'P': 'SF',
    'SF': 'SF',
    'SAF': 'S1',
    'PAF': 'SF',
    'RAF': 'RSTR',
    'SU': 'OTH',
    'RU': 'RSTR',
    'PU': 'SF',
    'AU': 'SF',
}

# 常见服务映射
PORT_SERVICE_MAP = {
    80: 'http', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'dns',
    110: 'pop_3', 143: 'imap4', 443: 'http', 3306: 'mysql', 5432: 'postgres',
    139: 'netbios-ssn', 445: 'microsoft-ds', 161: 'snmp', 162: 'snmp-trap',
    587: 'smtp', 993: 'imap4s', 995: 'pop3s', 20: 'ftp_data'
}


# ---------------------- 2. 加载NSL-KDD训练的预处理组件 ----------------------
def load_nsl_kdd_preprocessors(model_dir='./cids_nsl_kdd_models/'):
    """
    加载NSL-KDD训练时保存的编码器和缩放器
    :param model_dir: 含encoders.pkl和scalers.pkl
    :return: encoders（LabelEncoder字典）, scalers（StandardScaler）
    """
    required_files = ['encoders.pkl', 'scalers.pkl']
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NSL-KDD预处理组件缺失：{file_path}")

    with open(os.path.join(model_dir, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    with open(os.path.join(model_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)

    print(f"成功加载NSL-KDD预处理组件（协议/服务/标志位编码器）")
    return encoders, scalers


# ---------------------- 3. 会话特征计算工具 ----------------------
def calculate_window_features(sessions, window_seconds=2):
    """
    计算NSL-KDD特有的时间窗口特征（如count、srv_count等）
    :param sessions: 所有会话的字典（key:会话标识，value:会话详情）
    :param window_seconds: 时间窗口
    :return: 补充窗口特征后的会话字典
    """
    session_times = [(sess['start_time'], key) for key, sess in sessions.items()]
    session_times.sort()

    for idx, (curr_time, curr_key) in enumerate(session_times):
        curr_sess = sessions[curr_key]
        src_ip = curr_key[0]
        dst_ip = curr_key[2]
        proto = curr_key[4]
        service = curr_sess['service']

        # 1. count：当前会话2秒窗口内，与当前会话同（源IP,目的IP）的会话数
        count = 0
        # 2. srv_count：当前会话2秒窗口内，与当前会话同（源IP,目的IP,服务）的会话数
        srv_count = 0
        # 3. dst_host_count：当前会话2秒窗口内，与当前会话同目的IP的会话数
        dst_host_count = 0
        # 4. dst_host_srv_count：当前会话2秒窗口内，与当前会话同（目的IP,服务）的会话数
        dst_host_srv_count = 0

        # 遍历窗口内的历史会话
        for prev_time, prev_key in session_times[:idx]:
            if curr_time - prev_time > window_seconds:
                continue

            prev_sess = sessions[prev_key]
            prev_src_ip = prev_key[0]
            prev_dst_ip = prev_key[2]
            prev_service = prev_sess['service']

            # 更新count（同源IP+目的IP）
            if prev_src_ip == src_ip and prev_dst_ip == dst_ip:
                count += 1
                if prev_service == service:
                    srv_count += 1

            # 更新dst_host_count（同目的IP）
            if prev_dst_ip == dst_ip:
                dst_host_count += 1
                if prev_service == service:
                    dst_host_srv_count += 1

        # 填充窗口特征
        curr_sess['count'] = count + 1
        curr_sess['srv_count'] = srv_count + 1
        curr_sess['dst_host_count'] = dst_host_count + 1
        curr_sess['dst_host_srv_count'] = dst_host_srv_count + 1

        # 计算比率特征
        curr_sess['same_srv_rate'] = srv_count / count if count > 0 else 1.0
        curr_sess['diff_srv_rate'] = 1.0 - curr_sess['same_srv_rate']
        curr_sess['srv_diff_host_rate'] = 0.0  # 简化：实际需统计同服务不同主机数
        curr_sess['dst_host_same_srv_rate'] = dst_host_srv_count / dst_host_count if dst_host_count > 0 else 1.0
        curr_sess['dst_host_diff_srv_rate'] = 1.0 - curr_sess['dst_host_same_srv_rate']
        curr_sess['dst_host_same_src_port_rate'] = 1.0  # 简化：实际需统计同目的IP+源端口数

    return sessions


# ---------------------- 4. PCAP解析与NSL-KDD特征提取 ----------------------
def pcap_to_nsl_kdd(pcap_path, model_dir='./cids_nsl_kdd_models/', window_seconds=2):
    """
    将PCAP文件转换为NSL-KDD格式的特征数据（可直接输入CIDS模型）
    :param pcap_path: PCAP文件路径
    :param model_dir: NSL-KDD CIDS模型目录（加载预处理组件）
    :param window_seconds: 时间窗口
    :return: X_scaled（缩放后特征，可直接预测）, raw_df（原始特征DataFrame，用于调试）
    """
    # 步骤1：加载NSL-KDD预处理组件
    encoders, scalers = load_nsl_kdd_preprocessors(model_dir)

    # 步骤2：读取PCAP文件，初始化会话字典
    try:
        packets = rdpcap(pcap_path)
        print(f"成功读取PCAP文件：{pcap_path}，共{len(packets)}个数据包")
    except Exception as e:
        raise RuntimeError(f"读取PCAP失败：{str(e)}")

    # 会话标识：(源IP, 源端口, 目的IP, 目的端口, 协议proto)
    sessions = {}
    for pkt in packets:
        if IP not in pkt:
            continue

        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
        protocol_str = PROTOCOL_PROTO_MAP.get(proto, 'other')

        # 处理端口
        sport = 0
        dport = 0
        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport

        if (src_ip, sport) < (dst_ip, dport):
            session_key = (src_ip, sport, dst_ip, dport, proto)
        else:
            session_key = (dst_ip, dport, src_ip, sport, proto)

        # 初始化会话
        if session_key not in sessions:
            sessions[session_key] = {
                'start_time': pkt.time,
                'end_time': pkt.time,
                'protocol_str': protocol_str,
                'service': 'other',
                'flag': 'OTH',
                'src_bytes': 0,
                'dst_bytes': 0,
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': 0,
                'srv_count': 0,
                'serror_rate': 0.0,
                'srv_serror_rate': 0.0,
                'rerror_rate': 0.0,
                'srv_rerror_rate': 0.0,
                'same_srv_rate': 0.0,
                'diff_srv_rate': 0.0,
                'srv_diff_host_rate': 0.0,
                'dst_host_count': 0,
                'dst_host_srv_count': 0,
                'dst_host_same_srv_rate': 0.0,
                'dst_host_diff_srv_rate': 0.0,
                'dst_host_same_src_port_rate': 0.0,
                'dst_host_srv_diff_host_rate': 0.0,
                'dst_host_serror_rate': 0.0,
                'dst_host_srv_serror_rate': 0.0,
                'dst_host_rerror_rate': 0.0,
                'dst_host_srv_rerror_rate': 0.0
            }

        # 更新会话基础信息
        curr_sess = sessions[session_key]
        curr_sess['end_time'] = pkt.time
        pkt_len = len(pkt)

        # 区分源/目的字节数
        if session_key[0] == src_ip and session_key[1] == sport:
            curr_sess['src_bytes'] += pkt_len
        else:
            curr_sess['dst_bytes'] += pkt_len

        if TCP in pkt:
            tcp_flags = pkt[TCP].flags
            actual_flag = TCP_FLAG_MAP.get(tcp_flags, 'OTH')
            nslkdd_flag = ACTUAL_FLAG_TO_NSLKDD.get(actual_flag, 'OTH')
            curr_sess['flag'] = nslkdd_flag

        if dport != 0:
            curr_sess['service'] = PORT_SERVICE_MAP.get(dport, 'other')
        elif sport != 0:
            curr_sess['service'] = PORT_SERVICE_MAP.get(sport, 'other')
        if HTTPRequest in pkt or HTTPResponse in pkt:
            curr_sess['service'] = 'http'
        elif DNS in pkt:
            curr_sess['service'] = 'dns'

        if curr_sess['service'] in ['telnet', 'ssh', 'ftp']:
            curr_sess['hot'] += 1

        if TCP in pkt and pkt[TCP].flags == 0x04 and curr_sess['service'] in ['telnet', 'ssh']:
            curr_sess['num_failed_logins'] += 1

        if src_ip == dst_ip and sport == dport and sport != 0:
            curr_sess['land'] = 1

        if TCP in pkt and (pkt[TCP].flags & 0x20) != 0:
            curr_sess['urgent'] += 1

    # 步骤3：计算时间窗口特征
    sessions = calculate_window_features(sessions, window_seconds=window_seconds)

    # 步骤4：构建NSL-KDD特征DataFrame
    feature_list = []
    for sess_key, sess in sessions.items():
        duration = sess['end_time'] - sess['start_time']
        # 构建特征行
        feature_row = [
            duration,
            sess['protocol_str'],
            sess['service'],
            sess['flag'],
            sess['src_bytes'],
            sess['dst_bytes'],
            sess['land'],
            sess['wrong_fragment'],
            sess['urgent'],
            sess['hot'],
            sess['num_failed_logins'],
            sess['logged_in'],
            sess['num_compromised'],
            sess['root_shell'],
            sess['su_attempted'],
            sess['num_root'],
            sess['num_file_creations'],
            sess['num_shells'],
            sess['num_access_files'],
            sess['num_outbound_cmds'],
            sess['is_host_login'],
            sess['is_guest_login'],
            sess['count'],
            sess['srv_count'],
            sess['serror_rate'],
            sess['srv_serror_rate'],
            sess['rerror_rate'],
            sess['srv_rerror_rate'],
            sess['same_srv_rate'],
            sess['diff_srv_rate'],
            sess['srv_diff_host_rate'],
            sess['dst_host_count'],
            sess['dst_host_srv_count'],
            sess['dst_host_same_srv_rate'],
            sess['dst_host_diff_srv_rate'],
            sess['dst_host_same_src_port_rate'],
            sess['dst_host_srv_diff_host_rate'],
            sess['dst_host_serror_rate'],
            sess['dst_host_srv_serror_rate'],
            sess['dst_host_rerror_rate'],
            sess['dst_host_srv_rerror_rate']
        ]
        feature_list.append(feature_row)

    raw_df = pd.DataFrame(feature_list, columns=FEATURE_NAMES)
    if raw_df.empty:
        raise ValueError("PCAP文件中未提取到有效IP会话数据")

    # 步骤5：离散特征编码
    discrete_cols = ['protocol_type', 'service', 'flag']
    for col in discrete_cols:
        known_classes = set(encoders[col].classes_)
        raw_df[col] = raw_df[col].apply(
            lambda x: x if x in known_classes else encoders[col].classes_[0]
        )
        raw_df[col] = encoders[col].transform(raw_df[col])

    # 步骤6：特征缩放
    X_scaled = scalers.transform(raw_df[FEATURE_NAMES].values)

    print(f"PCAP转NSL-KDD完成：共{X_scaled.shape[0]}个会话特征，{X_scaled.shape[1]}维特征")
    return X_scaled, raw_df


# ---------------------- 5. 测试代码 ----------------------
if __name__ == "__main__":
    # 配置路径
    PCAP_PATH = "demo.pcap"
    MODEL_DIR = "cids_nsl_kdd_models/"

    try:
        X_scaled, raw_df = pcap_to_nsl_kdd(
            pcap_path=PCAP_PATH,
            model_dir=MODEL_DIR
        )

        from cids_Inferencer_nslkdd import CIDSInferencer

        inferencer = CIDSInferencer(model_dir=MODEL_DIR)

        batch_results = inferencer.predict_batch(
            samples=raw_df.to_dict('records')  # 转换为字典列表输入
        )

        encoders, _ = load_nsl_kdd_preprocessors(MODEL_DIR)
        print("\n" + "=" * 80)
        print("PCAP流量CIDS检测结果")
        print("=" * 80)
        for idx, (result, sess_features) in enumerate(zip(batch_results, raw_df.to_dict('records'))):
            print(f"\n【会话{idx + 1}】")
            print(f"基础信息：协议={encoders['protocol_type'].inverse_transform([sess_features['protocol_type']])[0]}, "
                  f"服务={encoders['service'].inverse_transform([sess_features['service']])[0]}, "
                  f"持续时间={sess_features['duration']:.2f}秒")
            print(f"检测结果：{result['攻击类型']}（是否正常：{result['是否正常流量']}）")
            print(f"特征预览：源字节={sess_features['src_bytes']}, 目的字节={sess_features['dst_bytes']}, "
                  f"TCP标志={encoders['flag'].inverse_transform([sess_features['flag']])[0]}")

    except Exception as e:
        print(f"测试失败：{str(e)}")
