# Deep Interest Network for Click-Through Rate Prediction

# 标题
- 参考论文：Deep Interest Network for Click-Through Rate Prediction
- 公司：Alibaba
- 链接：https://arxiv.org/pdf/1706.06978
- Code：https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/DIN/src/DIN.py
- `泛读`

# 内容

## 摘要
- 问题：
  - 将用户丰富的历史行为（如点击过的商品ID）通过embedding和pooling压缩成一个固定长度的表征向量，无法根据不同的候选广告进行自适应变化
  - 比如系统准备向这个用户推荐“跑鞋”还是“手机”，代表他的都是同一个向量。这个向量试图“一视同仁”地蕴含该用户所有的兴趣点，而且在面对具体推荐任务时显得不够聚焦。
  - 为了增强表达能力而粗暴地增加向量维度，又会带来参数量爆炸和过拟合的风险
- 解决方式，DIN：
  - 局部激活 (Local Activation)，用户行为序列(historical behaviors)和目标商品(target item)做融合，学习到行为序列内商品和目标商品的隐藏关系。
  - 小批量下的正则化方式(mini-batch aware regularization)
  - 自适应的激活函数生成(data adaptive activation function)
  - 帮助更好地落地实际业务中的大数据大模型，防止了模型的过拟合

## 1 介绍
主要贡献：
- 提出新模型（DIN）：针对固定长度向量无法有效表达用户多样化兴趣的局限性，我们提出了深度兴趣网络（DIN）。
  - 其核心是通过局部激活单元，能够根据给定的广告自适应地学习用户历史行为中的兴趣表征，从而极大地提升了模型表达能力，更好地捕捉了用户兴趣的多样性。
- 开发两项训练技术：为了高效训练工业级大规模深度网络，我们开发了两项新技术：
  - 小批量感知正则化：大幅降低了海量参数模型的正则化计算开销，有效防止过拟合。 
  - 数据自适应激活函数：改进了PReLU函数，能根据输入数据分布进行调整，性能优异。
- 验证与部署：
  - 在公开和阿里巴巴数据集上的大量实验证明了DIN模型和训练技术的有效性。相关代码已开源。该方法已在阿里巴巴（全球最大广告平台之一）的展示广告系统中成功部署，为业务带来了显著提升。

## 2 相关工作
- 采用了embedding + MLP 的结构的模型
  - LS-PLM and FM 此一次引入embedding 概念对sparse数据，并且尝试去找特征组合和target的关系
  - DeepCrossing and  W&D 改进了上面的transformation function，变成了MLP，提高了模型的学习能力
  - PNN 尝试拟合高阶特征组合
  - DeepFM 实现了不需要人工特征工程
- DIN 模型也参考了以上结构但是，但是以上模型对用户历史行为会转化成一个固定长度的vector通过 sum/pooling，必然会造成信息流失

## 4 DEEP INTEREST NETWORK

## 4.1 Feature Representation
<p style="text-align: center">
    <img src="./pics/DIN/DIN_4.1_特征处理.png">
      <figcaption style="text-align: center">
        DIN_特征处理
      </figcaption>
    </img>
  </p>

- 离散特征的处理：
  - 在推荐场景下最主要常见的就是用户和商品的ID
  - 所有离散特征都采用了One-hot的encoder方式
  - 用户点击行为序列内的离散特征做multi-hot编码。

## 4.2 Base Model(Embedding&MLP)
<p style="text-align: center">
    <img src="./pics/DIN/DIN_4.2_base_model.png">
      <figcaption style="text-align: center">
        DIN_base_model
      </figcaption>
    </img>
  </p>

- Embedding layer
  - one-hot，转化成一个embedding vector
  - multi-hot，转化成多个embedding vector组成的list
- Pooling layer and Concat layer
  - 用户的点击序列长度并不相同，如果不采取padding和截断，直接选择完整真实的用户点击序列
  - 可以用池化层派上很合适的用处。
  - 常见的池化为加和池化和平均池化。
  - 把所有的用户点击过的good id 对应的embedding进行 sum/average pooling 到一个一样维度的vector
- MLP
  - 将所有特征转为稠密向量拼接后，模型传入MLP层学习特征间的非线性关系。
- Loss
  - CTR点击率预测就是二分类问题，损失函数选择交叉熵损失函数

## 4.3 The structure of Deep Interest Network
<p style="text-align: center">
    <img src="./pics/DIN/DIN_4.3_模型结构.png">
      <figcaption style="text-align: center">
        DIN_模型结构
      </figcaption>
    </img>
  </p>

- 基准模型里面最大的问题是对用户点击序列的池化操作，这样丢失了很多信息，并且无论对应的是什么广告，都是固定长度的表达vector
- 如果直接增加user representation vector长度，会造成参数量爆炸和过拟合的风险
- DIN提出：
  - 增加了一个激活单元，在保证用户行为多样性（diverse）的同时，实现了对于不同候选广告
  - 局部激活（local activation）用户兴趣的特性。即为用户行为中每个商品与候选广告通过激活单元计算出一个激活权重。
  - 用户行为中的商品乘以权重再进行sum pooling。
  - 本质上还是pooling，但是是带有权重的pooling，这个权重和候选广告相关

<p style="text-align: center">
    <img src="./pics/DIN/DIN_4.3_局部激活公式.png">
      <figcaption style="text-align: center">
        DIN_局部激活公式
      </figcaption>
    </img>
  </p>

- e_1, e_2, ... e_h 是用户U的历史行为Embedding向量列表。 
- v_a 是候选广告A的Embedding向量。 
- a(e_j, v_a) 是一个激活单元（通常是一个小型前馈神经网络），它接收历史行为和候选广告作为输入，输出一个权重 w_j。这个权重就代表了历史行为在面对广告时的“相关性”或“注意力得分”。
- Activation Unit内
  - 传入User和Item特征向量后做了外积(out product)，两个向量越相近(线性相关程度高)，外积越饱满信息量越高，两个向量不相近(线性程度低)，外积信息量越低。
  - 再跑一个自适应的激活函数
  - 输出层是一层的MLP，输出是当前User和Item的 w_j
- 一个值得注意的细节是，DIN计算出的注意力权重 w_j 没有经过Softmax归一化。这意味着 sum(w_j) 不一定等于1。
- 这样设计的目的是为了保留用户兴趣的绝对强度。例如，如果一个用户的历史行为大部分都与某个广告高度相关，那么加权和之后的向量模长就会比较大，反之则较小。这种设计使得模型不仅能捕捉兴趣的“方向”，还能感知兴趣的“强度”。
- 值得一提的是，paper已经尝试过了用LSTM对用户历史行为，并没有起到好的效果：
  - 可能是因为用户的行为在存在同时发生的兴趣行为
  - 或者存在急速的变化或者停止的兴趣行为，造成sequence 用户行为变成noisy
  - 一个可能的方向是，设计一个独特的结构去model这个数据

## 5 TRAINING TECHNIQUES
