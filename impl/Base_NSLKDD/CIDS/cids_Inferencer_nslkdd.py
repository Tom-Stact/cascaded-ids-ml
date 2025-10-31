import numpy as np
import pandas as pd

from cids_nsl_kdd import CascadedIDS
from cids_nsl_kdd import FEATURE_NAMES, ATTACK_MAPPING

MODEL_DIR = './cids_nsl_kdd_models/'


class CIDSInferencer:
    """CIDS模型推理工具"""

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.cids = CascadedIDS.load_model(model_dir)
        print(f"模型已从 {model_dir} 加载，推理器初始化完成")

    def preprocess_single_sample(self, sample):
        """预处理单条样本"""
        # 转换样本为DataFrame
        if isinstance(sample, list):
            if len(sample) != len(FEATURE_NAMES):
                raise ValueError(f"样本特征数量错误，需{len(FEATURE_NAMES)}个特征")
            df = pd.DataFrame([sample], columns=FEATURE_NAMES)
        elif isinstance(sample, dict):
            missing_features = set(FEATURE_NAMES) - set(sample.keys())
            if missing_features:
                raise ValueError(f"样本缺少特征：{missing_features}")
            df = pd.DataFrame([sample])
        else:
            raise TypeError("样本格式仅支持字典或列表")

        # 离散特征编码
        discrete_cols = ['protocol_type', 'service', 'flag']
        for col in discrete_cols:
            known_classes = set(self.cids.encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else self.cids.encoders[col].classes_[0])
            df[col] = self.cids.encoders[col].transform(df[col])

        # 特征标准化
        X_scaled = self.cids.scalers.transform(df[FEATURE_NAMES].values)
        return X_scaled

    def predict_single(self, sample):
        """单样本预测"""
        X_scaled = self.preprocess_single_sample(sample)
        final_pred = self.cids.predict(X_scaled)[0]
        return {
            "预测标签": final_pred,
            "攻击类型": ATTACK_MAPPING.get(final_pred, "未知攻击类型"),
            "是否正常流量": final_pred == 1
        }

    def predict_batch(self, samples):
        """批量样本预测"""

        X_list = []
        for sample in samples:
            X_scaled = self.preprocess_single_sample(sample)
            X_list.append(X_scaled[0])
        X_batch = np.array(X_list)
        final_preds = self.cids.predict(X_batch)
        return [
            {
                "预测标签": label,
                "攻击类型": ATTACK_MAPPING.get(label, "未知攻击类型"),
                "是否正常流量": label == 1
            } for label in final_preds
        ]


# 推理示例
def main():
    try:
        inferencer = CIDSInferencer()

        # 示例1：正常样本预测
        normal_sample = [
            0, 'tcp', 'ftp_data', 'SF', 491, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 2, 2, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 150, 25, 0.17, 0.03,
            0.17, 0.00, 0.00, 0.00, 0.05, 0.00
        ]
        print("【正常样本预测结果】")
        print(inferencer.predict_single(normal_sample))

        # 示例2：攻击样本预测
        attack_sample = [
            0, 'udp', 'private', 'REJ', 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 0.00, 0.00, 1.00, 1.00, 0.00, 0.00, 0.00, 255, 255, 1.00, 0.00,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00
        ]
        print("\n【攻击样本预测结果】")
        print(inferencer.predict_single(attack_sample))

        # 示例3：批量预测
        print("\n【批量预测结果】")
        batch_results = inferencer.predict_batch([normal_sample, attack_sample])
        for i, res in enumerate(batch_results, 1):
            print(f"样本{i}：{res}")

    except Exception as e:
        print(f"推理过程出错：{str(e)}")


if __name__ == "__main__":
    main()