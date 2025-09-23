# Deep Interest Evolution Network for Click-Through Rate Prediction

# 标题
- 参考论文：Deep Interest Evolution Network for Click-Through Rate Prediction
- 公司：Alibaba
- 链接：https://arxiv.org/pdf/1809.03672
- Code：https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/DIEN/src/DIEN.py
- `泛读`

# 内容

## 摘要
- 问题：
  - 目前特征交叉方向存在一些使用用户行为的CTR预测方法，但是这些方法缺乏对具体行为背后潜在兴趣的专门建模。
  - 目前较少的工作考虑了兴趣的变化。
- 方法：
  - 提出了DIEN模型：
    - 提出兴趣抽取层，来从用户的历史行为序列中获取与时间有关的兴趣，同时一个辅助损失函数在每一步的学习中进行监督
    - 提出兴趣进化层，注意力机制被嵌入到序列结构中，在兴趣进化过程中，相关兴趣的作用增强
  - 线上得到 20.7% 提高在 CTR

## 1 介绍
- 之前DIN模型的问题：
  - DIN模型强调了用户兴趣的多样性，采用基于注意力机制的模型来获取目标项目的相关兴趣
  - DIN模型直接将用户的行为视作用户的兴趣。但是用户的潜在兴趣是很难通过显性的用户行为所反映的。
  - 用户的兴趣是不断变化的，获取动态变化的兴趣对兴趣表示是十分重要的
  - 换句话来说DIN模型将用户的历史行为看作是一个无序的集合，忽略了行为之间的时序依赖关系。用户的兴趣不仅是多样的，更是在持续演化的。
- DIEN的改进：
  - 1）兴趣抽取层，主要作用是通过模拟用户兴趣迁移过程，抽取用户兴趣。
    - GRU进行行为建模。遵循的原则是兴趣直接导致了连续的行为，提出了辅助损失（auxiliary loss）即使用下一个行为来监督当前的隐藏状态
  - 2）兴趣进化层，主要作用是通过在兴趣抽取层基础上加入注意力机制，模拟与当前目标广告相关的兴趣进化过程。
    - 建立了与目标项目相关的兴趣进化轨迹模型，基于从兴趣抽取层获得的兴趣序列，
    - 另外设计了带有注意力机制的更新门的GRU（attentional update gate, 简称AUGRU)

- **主要贡献**：
  - 在电商系统中关注了兴趣进化现象，提出了一个新的网络架构对兴趣进化过程进行建模。兴趣进化模型使兴趣表示更丰富，CTR预测更准确。
  - 与直接以行为为兴趣不同，DIEN中专门设计了兴趣抽取层。针对GRU的隐藏状态对兴趣表示较差的问题，提出了一种辅助损失。
  - 设计了兴趣进化层。兴趣进化层有效地模拟了与目标项目相关的兴趣进化过程。

## 2 相关工作
- MLP, FM, W&D, PNN 等特征交叉模型：
  - 增强了特征交叉，提高了特征表达，但是缺乏用户兴趣的表示能力
- DIN：
  - 提出一个可以得到历史行为和候选广告的attention的机制
  - 但是历史行为序列之间的依赖关系无法得到
- DREAM：
  - 推荐系统里面用RNN来动态表达每个用户和全局历史购买行为的关系
  - 一方面用RNN的hidden state 来表示用户潜在的兴趣，缺乏特殊的监管方式，也就是缺乏辅助loss
  - 另一方面RNN的依赖性取决于相邻的用户行为连续的平等的，但是用户的行为演变可能多变的，每个行为背后的兴趣演化可能有它独立的track。
  - 用RNN只能得到一个固定的用户历史行为演化track，可能会让真正的用户的兴趣track被打断
  - 换句话来说，用户的兴趣行为可能在整个过程中有好几次track，如果都按照NLP里面的思路组成sequence的话，可能有的兴趣演化track已经被丢弃了到最后的hidden state的时候
- QA领域，DMN+模型：
  - 引入attention机制去学习输入sequence的位置和顺序，提出GRU (AGRU)
  - 基于上面这个paper的概念和模型，提出DIEN的AUGRU，能学习到积极的学习到相对的兴趣演化

## 3 Deep Interest Evolution Network

## 3.1 基础模型回顾
先回顾一下基础模型

### Feature Representation
- 四种类型的特征（用户信息，用户行为，广告和上下文信息）。广告设为目标。 每个种类的特征都有几个字段(fields)
- 用户信息包含性别、年龄等等
- 用户行为包含用户访问过的物品编号
- 广告包含广告id，商店id等
- 上下文包含设计类型id，时间等等。
- 每个特征可以被编码成one-hot表示，例如女性可以被编码成[0,1]。
- 本质上和DIN一样，先one-hot，然后转化成embedding

### Embedding
特征的每个字段（field）对应一个embedding矩阵， 同DIN

### Multilayer Perceptron (MLP)
- 首先，将一个类别的（上述提到过的四种类型）embedding向量输入进池化操作。
- 然后将来自不同类别的所有这些池化向量连接起来。
- 最后，将连接后的向量输入MLP进行最终预测。
- 同DIN，也是最传统的deep learning 做CTR的方式

### Loss Function
- negative log-likelihood function

## 3.2 DIEM 模型
<p style="text-align: center">
    <img src="../pics/DIEN/DIEN_3.2_模型结构.png">
      <figcaption style="text-align: center">
        DIEN_模型结构
      </figcaption>
    </img>
  </p>

- DIEN由几个部分组成：
  - 第一，所有的特征都由embedding层进行转换。
  - 第二，DIEN采用两个步骤获得兴趣进化：
    - i) 兴趣抽取层基于行为序列抽取兴趣序列，兴趣进化层是对与目标项目相关的兴趣进化过程进行建模。
    - ii) 然后，将最终兴趣的表示和嵌入向量，如广告、用户信息、上下文等进行连接。
  - 最后，将连接后的向量输入MLP进行最终预测。

### Interest Extractor Layer 兴趣抽取层
- 选用GRU：
  - 电子商务系统中的用户行为丰富，即使在很短的如两周这样的时间，其点击这样的历史行为序列也很长。平衡效率和性能。
  - GRU克服了RNN模型的梯度消失问题，并且速度比LSTM快。

<p style="text-align: center">
    <img src="../pics/DIEN/DIEN_3.2_GRU公式.png">
      <figcaption style="text-align: center">
        DIEN_GRU公式
      </figcaption>
    </img>
  </p>

- 公式解读
  - σ是sigmoid激活函数
  - o是element-wise product， 
  - W_u, W_r, W_h - (n_h, n_i)
  - U_u, U_r, U_h - (n_h)
  - n_h 是隐藏层的大小
  - n_i 是输入层的大小
  - i_t 是GRU的输入
  - i_t = e_t 代表用户产生的第t个行为的embedding向量
  - h_t 是第t个隐藏状态

- 辅助loss
  - 问题：
    - h_t只能表示邻居行为的依赖性
    - 最终的label只能预测target和最终兴趣的关系
    - 历史兴趣状态没办法得到合适的监督和使用
    - 然而历史信息的每一步都可能导致直接的连续用户行为
  - 方法：
    - 用 b_t+1 的状态来监督 h_t
    - 使用下一个实际的行为作为正实例
    - 还使用从除去 click 以外的 item set 里面取样负实例
    - 本质上就是 N 对 pair 的binary classification

<p style="text-align: center">
    <img src="../pics/DIEN/DIEN_3.2_辅助loss公式.png">
      <figcaption style="text-align: center">
        DIEN_辅助loss公式
      </figcaption>
    </img>
  </p>

<p style="text-align: center">
    <img src="../pics/DIEN/DIEN_3.2_全局loss公式.png">
      <figcaption style="text-align: center">
        DIEN_全局loss公式
      </figcaption>
    </img>
  </p>

- α 是超参数，用来平衡兴趣表达和CTR最终预测
- 总结：
  - 帮助GRU的隐藏层表达兴趣
  - 降低了back propagation 的困难度，当GRU模型有长历史行为的时候
  - 提供了更多的语义信息对于embedding，构建更好的embedding matrix

### Interest Evolving Layer 兴趣进化层
- 兴趣提取层的不足：
  - 兴趣的演化并不总是平滑的，常常会伴随着兴趣漂移（Interest Drifting）现象，即用户可能在不同的兴趣点之间快速切换。
  - 只用GRU来建模这个兴趣序列，不相关的历史兴趣（漂移）可能会干扰对当前主要兴趣演化的判断。
- 兴趣进化层的优点：
  - 兴趣进化模块可以为最终兴趣的表示提供更多的相关历史信息
  - 根据兴趣进化趋势预测目标项目的点击率比较好
  - 用每一个step的兴趣信息和target计算权重，能更好的解决兴趣漂移的问题，本质上就是DIN的思路，每个兴趣对最终label都有不同的权重贡献
- 解决方式：
  - 结合了注意力机制中的局部激活能力和GRU的序列学习能力来实现建模兴趣演化的目标
  - 提出attention score在当前状态h_t和最终target之间的关系，公式如下
  - e_a 是从广告中的嵌入向量的连接
  - W - (n_h, n_a)
  - n_h 是隐状态的维数
  - n_a 是广告嵌入向量的维数
  - 这一层的GRU的输入就是上一层GRU的输出，i_t = h_t，注意这里的 i_t 是上一层的输出并且会被下面attention score更新，具体更新方式见下文详解

<p style="text-align: center">
    <img src="../pics/DIEN/DIEN_3.2_兴趣进化层权重公式.png">
      <figcaption style="text-align: center">
        DIEN__兴趣进化层权重公式
      </figcaption>
    </img>
  </p>

- 如何融合注意力机制中的局部激活能力和GRU的序列学习
  - GRU with attentional input (AIGRU)





