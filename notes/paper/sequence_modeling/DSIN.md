# Deep Session Interest Network for Click-Through Rate Prediction

# 标题
- 参考论文：Deep Session Interest Network for Click-Through Rate Prediction
- 公司：Alibaba
- 链接：https://arxiv.org/pdf/1905.06482
- Code：https://github.com/shenweichen/DSIN
- `泛读`

# 内容

## 摘要
- 问题：
  - 兴趣变化是电商平台CTR预估的一个重要思路，但是更多的研究都聚焦于“兴趣转变”上，而忽略了行为序列本身的分布特点。
- 方法：
  - 提出DSIN模型：
    - 通过对 Session 的关注，在行为序列的划分上提出改进方案。
    - 合适的 Session 划分能够让单一session内的用户行为趋于同质，让不同session间的用户行为异质。
    - 在每个 Session 内 模型采用 attention 机制并将 attention 中的 position encoder 做了适配改进，学习用户的兴趣
    - 在不同 Session 间，同时利用双向 LSTM(Bi-LSTM) 学习用户兴趣的演变过程
    - 最后通过 local activation unit 学习不同session内的兴趣 对于 target item的影响

## 1 Introduction
- 用户的行为是由一系列的 sessions 组成的
- 一个 sessions 是由一系列的用户行为在给定时间段内组成的
- 并且连续时间内（一个 sessions）用户的点击集中在较为单一的兴趣上，而离散时间（不同sessions）往往发生了兴趣跃迁
- Session的划分规则：文中直接定义为相邻两个用户行为的发生时间如果大于等于30分钟，则划分为不同的两个 session
- DSIN 模型：
  - 第一，提出了把用户的行为划分成不同的 sessions
  - 用 self-attention 在每个sessions 内，来学习兴趣之间的 interaction和correlation，同时提取用户的兴趣在每个 sessions内
  - 第二，不同的 session 之间可能存在相关性，并且可能有连续性的pattern
  - 用双向 LSTM(Bi-LSTM) 学习兴趣之间的交互迁移和进化
  - 最后，用 local activation unit，局部激活单元，类似DIN，来融合这些兴趣，并且表达学习兴趣和 target item之间的关系

- **主要贡献**：
  - 用户行为序列的session划分规则
  - 处理session序列并学习兴趣演变过程时，采用了改进的self-attention网络和双向长短期记忆网络(Bi-LSTM)，最后提出local activation unit来学习不同兴趣和target item直接的关系
  - 在对比实验上取得了相较其他模型更好的效果。取得了SOTA的成绩。