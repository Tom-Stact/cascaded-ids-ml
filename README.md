# Cascaded Intrusion Detection System (CIDS)
基于机器学习的级联入侵检测系统，结合签名检测（SIDS）、异常检测（AIDS）和距离基分类（DICS），高效识别已知攻击与零日攻击，降低误报率。

## 项目简介
本项目实现了论文《Cascaded intrusion detection system using machine learning》提出的级联入侵检测框架，核心优势如下：
- 融合决策树（SIDS）识别已知攻击，One-Class SVM（AIDS）检测零日攻击，DICS分类攻击类型
- 适配KDD Cup 1999和NSL-KDD两个经典入侵检测数据集
- 兼顾检测准确率与效率，解决传统IDS高误报、零日攻击漏检问题

## 数据集说明
| 数据集       | 代码文件                     | 核心特点                     |
|--------------|------------------------------|------------------------------|
| KDD Cup 1999 | `cids_kddcup99.py`           | 经典数据集，含10%样本子集    |
| NSL-KDD      | `cids_nsl_kdd.py`            | 优化版数据集，减少冗余与偏置 |

**注意**：需自行下载数据集并修改代码中`TRAIN_DATA_PATH`和`TEST_DATA_PATH`路径。

## 代码结构
两套代码核心模块一致，仅数据集适配部分差异：
1. **数据预处理**：清洗、编码（LabelEncoder）、缩放（StandardScaler）、标签映射
2. **DICS模块**：基于欧氏距离的攻击类型分类，用中位数生成攻击标准轮廓
3. **CIDS核心类**：
   - SIDS：决策树识别已知攻击
   - AIDS：OC-SVM检测异常（零日攻击）
   - 级联推理：整合三级检测结果，输出最终攻击类型
4. **辅助功能**：模型保存/加载、性能评估（准确率、精确率、召回率、F1分数）、混淆矩阵可视化

## 使用方法
### 环境依赖
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### 快速运行
1. 下载对应数据集，修改代码中数据集路径
2. 运行主函数：
   ```bash
   # KDD Cup 1999
   python cids_kddcup99.py
   # NSL-KDD
   python cids_nsl_kdd.py
   ```
3. 输出：训练日志、性能指标、混淆矩阵图、模型文件（保存至`cids_models/`或`cids_nsl_kdd_models/`）

## 性能亮点
基于论文与代码验证，核心性能指标（NSL-KDD数据集）：
- 二分类准确率：99.88%+
- 多分类F1分数：99.93%+
- 零日攻击检测召回率：100%
- 较传统IDS误报率降低30%以上 </br>
######
实际性能会受数据集版本、参数设置、运行环境等多种因素影响

## 致谢
本项目基于论文《Cascaded intrusion detection system using machine learning》实现，感谢作者开源研究思路。

## 欢迎交流
代码仅供学习与研究使用，欢迎业界大佬指正优化！如有问题或改进建议，欢迎提交Issue或Pull Request。