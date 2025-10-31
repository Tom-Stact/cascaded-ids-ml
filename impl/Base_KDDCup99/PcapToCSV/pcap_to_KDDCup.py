import pickle

import pandas as pd
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP, ICMP

# ---------------------- 修复相对导入：添加系统路径 ----------------------
try:
    from Learn.CascadedIntrusionDetection_MachineLearning.impl.Base_KDDCup.CIDS.cids_system import (
        CascadedIDS, DICS
    )
except ImportError as e:
    print(f"警告：未找到CIDS模型模块，仅完成PCAP转换。错误：{e}")
    CascadedIDS = None
    DICS = None

# ---------------------- 1. 特征定义 ----------------------
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

# 协议、服务、标志位的映射
PROTOCOL_MAP = {1: 'icmp', 6: 'tcp', 17: 'udp'}
FLAG_MAP = {
    'SF': 0, 'REJ': 1, 'RSTO': 2, 'RSTOS0': 3, 'RSTR': 4, 'S0': 5,
    'S1': 6, 'S2': 7, 'S3': 8, 'SH': 9, 'OTH': 10
}
TCP_FLAG_TO_KDD = {
    'SA': 'S1', 'PA': 'SF', 'FA': 'SF', 'RA': 'RSTR', 'A': 'SF',
    'F': 'SF', 'R': 'RSTR', 'SP': 'SF', 'SPA': 'SF', 'FAP': 'SF',
    'SAF': 'S1', 'SAR': 'RSTO'
}
SERVICE_MAP = {
    80: 'http', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp',
    110: 'pop_3', 143: 'imap4', 53: 'dns', 443: 'http', 3389: 'telnet',
    53: 'dns', 67: 'other', 68: 'other', 161: 'other', 'other': 99
}


# ---------------------- 2. 加载训练好的编码器和缩放器 ----------------------
def load_preprocessors(model_dir='./cids_models/'):
    with open(f"{model_dir}/encoders.pkl", 'rb') as f:
        encoders = pickle.load(f)
    with open(f"{model_dir}/scalers.pkl", 'rb') as f:
        scalers = pickle.load(f)
    return encoders, scalers


# ---------------------- 3. 辅助函数：解析TCP标志并映射到KDD格式 ----------------------
def parse_tcp_flag(flag_str):
    return TCP_FLAG_TO_KDD.get(flag_str, 'OTH')


# ---------------------- 4. PCAP解析与特征提取（含数据包映射） ----------------------
def pcap_to_kdd(pcap_path, model_dir='./cids_models/'):
    """
    将PCAP文件转换为KDD格式的特征数据，并建立与原始数据包的映射
    :param pcap_path: PCAP文件路径
    :param model_dir: CIDS模型保存目录
    :return: 预处理后的特征数据, 原始特征DataFrame, 会话-数据包映射字典
    """
    encoders, scalers = load_preprocessors(model_dir)

    packets = rdpcap(pcap_path)
    features = []
    sessions = {}  # 按(源IP, 源端口, 目的IP, 目的端口, 协议)分组会话

    packet_to_session = {}

    # 遍历数据包
    for pkt_idx, pkt in enumerate(packets):
        if IP in pkt:
            ip = pkt[IP]
            proto = ip.proto
            src_ip = ip.src
            dst_ip = ip.dst
            sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0)
            dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0)

            # 标准化会话key，确保双向通信属于同一会话
            if (src_ip, sport) < (dst_ip, dport):
                session_key = (src_ip, sport, dst_ip, dport, proto)
            else:
                session_key = (dst_ip, dport, src_ip, sport, proto)

            # 初始化会话
            if session_key not in sessions:
                sessions[session_key] = {
                    'start_time': pkt.time,
                    'end_time': pkt.time,
                    'src_bytes': 0,
                    'dst_bytes': 0,
                    'flag': 'OTH',
                    'packets': 0,
                    'land': 0,
                    'wrong_fragment': 0,
                    'urgent': 0,
                    'hot': 0,
                    'num_failed_logins': 0,
                    'logged_in': 0,
                    'count': 1,
                    'srv_count': 1,
                    'serror_rate': 0.0,
                    'srv_serror_rate': 0.0,
                    'rerror_rate': 0.0,
                    'srv_rerror_rate': 0.0,
                    'same_srv_rate': 1.0,
                    'diff_srv_rate': 0.0,
                    'srv_diff_host_rate': 0.0,
                    'dst_host_count': 1,
                    'dst_host_srv_count': 1,
                    'dst_host_same_srv_rate': 1.0,
                    'dst_host_diff_srv_rate': 0.0,
                    'dst_host_same_src_port_rate': 1.0,
                    'dst_host_srv_diff_host_rate': 0.0,
                    'dst_host_serror_rate': 0.0,
                    'dst_host_srv_serror_rate': 0.0,
                    'dst_host_rerror_rate': 0.0,
                    'dst_host_srv_rerror_rate': 0.0,
                    'packet_indices': []  # 记录该会话包含的所有数据包索引
                }

            # 更新会话信息
            session = sessions[session_key]
            session['end_time'] = pkt.time
            session['packets'] += 1
            session['src_bytes'] += len(pkt) if src_ip < dst_ip else 0
            session['dst_bytes'] += len(pkt) if dst_ip < src_ip else 0
            session['packet_indices'].append(pkt_idx)
            packet_to_session[pkt_idx] = session_key

            if TCP in pkt:
                tcp = pkt[TCP]
                flags = []
                if tcp.flags & 0x01: flags.append('F')
                if tcp.flags & 0x02: flags.append('S')
                if tcp.flags & 0x04: flags.append('R')
                if tcp.flags & 0x08: flags.append('P')
                if tcp.flags & 0x10: flags.append('A')
                if tcp.flags & 0x20: flags.append('U')
                flag_str = ''.join(flags)
                session['flag'] = parse_tcp_flag(flag_str)

            elif ICMP in pkt:
                session['flag'] = 'OTH'
                session['src_bytes'] += len(pkt) if src_ip < dst_ip else 0
                session['dst_bytes'] += len(pkt) if dst_ip < src_ip else 0

        else:
            packet_to_session[pkt_idx] = None

    session_mapping = []
    for session_idx, (session_key, session) in enumerate(sessions.items()):
        src_ip, sport, dst_ip, dport, proto = session_key
        protocol = PROTOCOL_MAP.get(proto, 'other')

        # 服务映射
        if sport != 0 and sport in SERVICE_MAP:
            service = SERVICE_MAP[sport]
        elif dport != 0 and dport in SERVICE_MAP:
            service = SERVICE_MAP[dport]
        else:
            service = 'other'

        # 构建特征行
        feature_row = {
            'duration': round(session['end_time'] - session['start_time'], 6),
            'protocol_type': protocol,
            'service': service,
            'flag': session['flag'],
            'src_bytes': session['src_bytes'],
            'dst_bytes': session['dst_bytes'],
            'land': 1 if (src_ip == dst_ip and sport == dport and sport != 0) else 0,
            'wrong_fragment': session.get('wrong_fragment', 0),
            'urgent': session.get('urgent', 0),
            'hot': session.get('hot', 0),
            'num_failed_logins': session.get('num_failed_logins', 0),
            'logged_in': session.get('logged_in', 0),
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
            'count': session.get('count', 1),
            'srv_count': session.get('srv_count', 1),
            'serror_rate': round(session.get('serror_rate', 0.0), 6),
            'srv_serror_rate': round(session.get('srv_serror_rate', 0.0), 6),
            'rerror_rate': round(session.get('rerror_rate', 0.0), 6),
            'srv_rerror_rate': round(session.get('srv_rerror_rate', 0.0), 6),
            'same_srv_rate': round(session.get('same_srv_rate', 1.0), 6),
            'diff_srv_rate': round(session.get('diff_srv_rate', 0.0), 6),
            'srv_diff_host_rate': round(session.get('srv_diff_host_rate', 0.0), 6),
            'dst_host_count': session.get('dst_host_count', 1),
            'dst_host_srv_count': session.get('dst_host_srv_count', 1),
            'dst_host_same_srv_rate': round(session.get('dst_host_same_srv_rate', 1.0), 6),
            'dst_host_diff_srv_rate': round(session.get('dst_host_diff_srv_rate', 0.0), 6),
            'dst_host_same_src_port_rate': round(session.get('dst_host_same_src_port_rate', 1.0), 6),
            'dst_host_srv_diff_host_rate': round(session.get('dst_host_srv_diff_host_rate', 0.0), 6),
            'dst_host_serror_rate': round(session.get('dst_host_serror_rate', 0.0), 6),
            'dst_host_srv_serror_rate': round(session.get('dst_host_srv_serror_rate', 0.0), 6),
            'dst_host_rerror_rate': round(session.get('dst_host_rerror_rate', 0.0), 6),
            'dst_host_srv_rerror_rate': round(session.get('dst_host_srv_rerror_rate', 0.0), 6)
        }
        features.append(feature_row)

        # 记录会话映射信息
        session_mapping.append({
            'session_index': session_idx,
            'session_key': session_key,
            'packet_indices': session['packet_indices'],
            'packet_count': len(session['packet_indices']),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol
        })

    df = pd.DataFrame(features, columns=FEATURE_NAMES)

    df['session_index'] = range(len(df))

    # 离散特征编码
    discrete_cols = ['protocol_type', 'service', 'flag']
    for col in discrete_cols:
        if col == 'service':
            df[col] = df[col].apply(lambda x: x if x in encoders[col].classes_ else 'other')
        elif col == 'flag':
            df[col] = df[col].apply(lambda x: x if x in encoders[col].classes_ else 'OTH')
        elif col == 'protocol_type':
            df[col] = df[col].apply(lambda x: x if x in encoders[col].classes_ else 'other')
        df[col] = encoders[col].transform(df[col])

    # 特征缩放
    X_scaled = scalers.transform(df[FEATURE_NAMES].values)

    return X_scaled, df, session_mapping, packets


# ---------------------- 5. 辅助函数：显示数据包详情 ----------------------
def display_packet_details(packet, index):
    """显示数据包的关键信息"""
    details = []
    details.append(f"数据包 #{index} 详情:")
    if IP in packet:
        ip = packet[IP]
        details.append(f"  源IP: {ip.src}")
        details.append(f"  目的IP: {ip.dst}")
        details.append(f"  协议: {PROTOCOL_MAP.get(ip.proto, ip.proto)}")

    if TCP in packet:
        tcp = packet[TCP]
        details.append(f"  源端口: {tcp.sport}")
        details.append(f"  目的端口: {tcp.dport}")
        details.append(f"  TCP标志: {tcp.flags}")

    elif UDP in packet:
        udp = packet[UDP]
        details.append(f"  源端口: {udp.sport}")
        details.append(f"  目的端口: {udp.dport}")

    elif ICMP in packet:
        icmp = packet[ICMP]
        details.append(f"  ICMP类型: {icmp.type}")
        details.append(f"  ICMP代码: {icmp.code}")

    details.append(f"  长度: {len(packet)} bytes")
    details.append(f"  时间戳: {packet.time}")

    return "\n".join(details)


# ---------------------- 6. 测试代码（含映射功能） ----------------------
if __name__ == "__main__":
    pcap_path = "demo.pcap"
    model_dir = "../CIDS/cids_models/"

    try:
        X_scaled, df, session_mapping, packets = pcap_to_kdd(pcap_path, model_dir)
        print("✅ PCAP转换成功！")
        print(f"样本数：{X_scaled.shape[0]}，特征数：{X_scaled.shape[1]}")
        print(f"原始数据包总数：{len(packets)}")

        if CascadedIDS is not None and DICS is not None:
            cids = CascadedIDS.load_model(model_dir)
            y_pred = cids.predict(X_scaled)

            # 攻击类型映射
            ATTACK_MAPPING = {
                1: 'Normal',
                2: 'Buffer Overflow (U2R)', 3: 'Loadmodule (U2R)', 4: 'Perl (U2R)',
                5: 'Neptune (DoS)', 6: 'Smurf (DoS)', 7: 'Guess Password (R2L)',
                8: 'Pod (DoS)', 9: 'Teardrop (DoS)', 10: 'Port sweep (Probe)',
                11: 'IP sweep (Probe)', 12: 'Land (DoS)', 13: 'Ftp write (R2L)',
                14: 'Backdoor (DoS)', 15: 'IMAP (R2L)', 16: 'Satan (Probe)',
                17: 'PHF (R2L)', 18: 'NMAP (Probe)', 19: 'Multi-hop (R2L)',
                20: 'Warez-master (R2L)', 21: 'Warez-client (R2L)', 22: 'Spy (R2L)',
                23: 'Root-kit (U2R)'
            }

            print("\n预测结果与原始数据包映射：")
            for i in range(min(100, len(y_pred))):
                attack_type = ATTACK_MAPPING.get(y_pred[i], 'Unknown')
                mapping = session_mapping[i]

                print(f"\n样本{i + 1}：{attack_type}")
                print(f"  对应原始数据包索引: {mapping['packet_indices']}")
                print(f"  数据包数量: {mapping['packet_count']}")
                print(f"  通信对: {mapping['src_ip']} -> {mapping['dst_ip']} ({mapping['protocol']})")

                # 对检测到的攻击样本，显示详细的数据包信息
                if attack_type != 'Normal':
                    print("  攻击数据包详情:")
                    # 显示该会话的第一个数据包详情
                    first_pkt_idx = mapping['packet_indices'][0]
                    print(display_packet_details(packets[first_pkt_idx], first_pkt_idx))

                    # 如果有多个数据包，提示用户可以查看更多
                    if len(mapping['packet_indices']) > 1:
                        print(f"  ... 还有 {len(mapping['packet_indices']) - 1} 个相关数据包")

        else:
            print("❌ 模型导入失败，未执行预测")

        # 提供按数据包索引查询会话的功能
        print("\n=====================================")
        print("数据包索引查询功能")
        print("=====================================")
        while True:
            try:
                pkt_idx = input("请输入要查询的数据包索引(输入q退出): ")
                if pkt_idx.lower() == 'q':
                    break
                pkt_idx = int(pkt_idx)
                if pkt_idx < 0 or pkt_idx >= len(packets):
                    print(f"无效的索引，数据包索引范围是0到{len(packets) - 1}")
                    continue

                found = False
                for mapping in session_mapping:
                    if pkt_idx in mapping['packet_indices']:
                        print(f"\n数据包 #{pkt_idx} 属于样本 {mapping['session_index'] + 1}")
                        print(display_packet_details(packets[pkt_idx], pkt_idx))
                        found = True
                        break
                if not found:
                    print(f"数据包 #{pkt_idx} 不是IP数据包或未被处理")

            except ValueError:
                print("请输入有效的数字索引")

    except Exception as e:
        print(f"❌ 执行失败：{str(e)}")