# 交接说明

写给接手这份工作的人。2026-10-07。

这份文档只做一件事:**把不在代码里、也不在另外三份记录里的东西写下来**。
决策的来由、踩过的坑、哪些数字能引用哪些不能 —— 这些原本只存在于对话记录里,
而对话记录带不走。

读完这份就去读 `TODO.md`,它是真正的工作计划。

---

## 1. 三份权威记录

| 文件 | 内容 | 什么时候读 |
| --- | --- | --- |
| `experiments/cost_model.md` | 形式化的资源代价模型。12 条硬约束 C1–C12、4 条传导链、能力面/搜索层的分层、7 个专题分析(§7.1–7.8) | 要理解**为什么这颗芯片需要自研工具链**,以及任何"硬件能不能做 X"的问题 |
| `experiments/TODO.md` | 工作计划。按 P0.0 / P0 / P1 / P1.5 / P2 / P3 / P4 排优先级,每条都把已有发现折进去了 | 决定下一步做什么 |
| `quant_tool/tool/paicore/snn_validation.md` | 原生 SNN 量化的端到端验证记录,含芯片侧数据与低位宽数据 | 碰 SNN 之前 |

这三份是经过反复修订的,**出现冲突时以它们为准,不以本文为准**。

---

## 2. 题目与立论

**《面向 ANN-SNN 统一类脑核的资源感知模式与精度联合优化》**

立论的起点是一个否定性结论:**MLIR / LLVM / TVM 在这里都不适用**,而这不是能力问题
是形态问题。PAICORE 2.5 的离线核是固定功能的,编译产物是 64 位配置帧而不是机器码 ——
LLVM 的指令选择与寄存器分配无对象;计算模式固定,没有调度搜索空间 —— TVM 的自动调优
无对象;单一目标、单一抽象层级 —— MLIR 的多层方言渐降无收益。真正缺的那一段是
**在硬件资源约束下做量化决策**,而这是通用工具链共同的空白。

于是论文的贡献面是三层:

1. **第三章(保底)**:把硬件手册翻译成可执行的代价模型与约束检查。
2. **第四章**:补齐"硬件支持、后端支持、但前端说不出来"的能力面落差。
   三个维度 per-channel / per-group / ANN-SNN 模式,**缺口形状完全一样**,
   这张表本身就是论点(`TODO.md` §P1.0)。
3. **第五章**:在补齐的能力面上做资源-精度联合的逐层决策搜索。

**核心的量化论据**(第三章已拿到):资源盲的 int8 量化器会选 `w8a8`,占 24 核;
而 `w4a8` 精度完全相同、只占 18 核 —— **8-bit 权重从不是正确选择,但它是默认选择**。
`w4a4` 在 15 核上就达到浮点水平,少 38% 的核。

---

## 3. 代码在哪

```
PAIBox/                         分支 feat/quant-frontend
  paibox/quant/                 量化前端(本工作的主体,已提交)
    hw.py                       硬件常量与约束检查(C1–C12 的可执行形式)
    lut.py                      256 项激活 LUT 拟合,阈值置于判决边界
    spec.py                     QuantSpec —— 前后端的声明式契约
    export.py                   float 模型 + QuantSpec → PAIBox 可降级的模型
  tests/quant/                  上面四个的测试
  experiments/                  研究脚本与记录(本文所在目录)
    deploy.py                   ANN 单配置端到端:量化→导出→编译→布放→评分
    snn_quantize.py             SNN 的同一条路,含参考实现 int8 作对照
    pareto.py                   位宽网格上的精度-核数帕累托
    sweep.py core_probe.py snn_probe.py residual_probe.py
                                单问题资源探针
    snn_layer_diff.py snn_param_diff.py snn_tick0_diff.py
                                诊断脚本,见下 §6
    models/                     外部训练的架构与其 spec

quant_tool/                     独立仓库
  tool/paicore/                 PAICORE 专用的标定与 spec 构造
    calib.py                    MSE / percentile / 2 幂格点标定器
    fuse.py                     BN 与输入归一化折叠(原地改写)
    topology.py                 FX 数据流,神经元当叶子
    spiking.py                  SpikingJelly 层注册与 spiking site 识别
    spec_builder.py             标定后的模型 → QuantSpec
    config.py                   modelopt 配置
  tool/mytorch/                 vendored 的 NVIDIA modelopt 子集(Apache 2.0)
```

提交位置:PAIBox `9bfc403`(experiments)、`b4dcd65`(per-channel IR)、
`abb84cf`(quant_tool 对接);quant_tool `461897a`。

**两个仓库是并列关系**,`quant_tool` 要在 `PYTHONPATH` 上:

```sh
cd PAIBox && uv sync
PYTHONPATH=../quant_tool uv run python experiments/deploy.py --eval
```

环境是 uv 管的 Python 3.14,`paicorelib 2.0.0`。不要复制 `.venv`,用 `uv.lock` 重建。

---

## 4. 一个容易误解的边界:quant_tool 到底做了什么

**`quant_tool` 目前的实际贡献只有 `amax`。**

它是标定与尺度记账库。量化数学 —— 整数物化、LUT 拟合、移位配置、约束检查 ——
全在 `paibox/quant`。**所有已报告的精度数字百分之百来自 PAIBox 侧的执行。**

这同时界定了"把 quant_tool 接进 PAICORE"这件事的范围:接的是**标定算法**,
不是量化执行。

**为什么不改用 `torch.ao`**(仓库里 `quantize` 分支走的就是那条路):
`BackendConfig` 只能描述融合模式与 dtype,**没有任何位置能表达"这一层改成 SNN、
阈值 θ、时间步 T"**。这与"ONNX 的 Q/DQ 表达不了这颗芯片的决策"是同一条论证 ——
交换格式必须能承载目标的决策空间,而 mode × 位宽 × 粒度这个空间 torch.ao 装不下。
另一个硬理由:那条分支只有 8-bit + per-tensor + MinMax 标定,而在这一个点上
精度与资源都没有可优化的维度,论文的立论无从展开。

两条分支的能力对比表在 `TODO.md` §"与 `quantize` 分支的关系",那里也写了
**可以直接复用的东西**(`QuantizedConvAddReLU2dOp` 与前端无关,残差不需要复刻
torch.ao 的模式识别)。

### 目标形态,以及离它还差什么

项目完整的样子是一条闭合的链:

```
torch 模型
  → 识别图      哪些是量化 site、谁是谁的前驱、激活挂在哪
  → 替换图      装观测器 / 换成芯片神经元
  → 量化        按选定的粒度与标定算法定出 scale
  → convert     产出 QuantSpec(声明式契约,硬件无关)
  → PAIBox      按 spec 物化整数权重、填 LUT、配移位,编译布放
```

**这就是现在的分工,不需要改架构** —— quant_tool 做决策,`QuantSpec` 过契约,
PAIBox 执行。缺的是每一环里的具体能力。逐环对照:

| 环节 | 现状 | 缺口 |
| --- | --- | --- |
| 识别图 | `topology.py`,FX 提取 site 拓扑序/前驱/激活路径,神经元当叶子 | **分支拓扑**。残差会响亮报错并说明缺什么(这是故意的,早先是静默串错 scale 链) |
| 替换图 | quant_tool 侧 `apply_mode` 装观测器(含 SpikingJelly 层);PAIBox 侧 `export` 换芯片神经元 | **融合阻断标注** —— per-group 路径 A 的前提,现在只能靠"假消费者" |
| 做量化 | 标定在 quant_tool,整数物化/LUT/移位在 PAIBox | **伪量化中间参照缺失**。vendored modelopt 的 `_fake_quantize` 整段被注释,所以无法做"float → 伪量化 → 芯片仿真"的三段对比,**分不清精度损失来自量化决策还是部署环节** |
| 不同粒度 | per-tensor ✓、per-channel ✓(已打通并实测) | **per-group ✗** —— `export.py` 显式抛 `UnsupportedConfig`。这就是"把 PAIBox 里还没支持的粒度实现"那一项,见 §7 |
| 不同算法 | max / percentile / MSE / Pow2Mse ✓ | histogram-KL(vendored 的方法体被注释掉了);QAT 完全没有,现在全是 PTQ |
| convert | `QuantSpec` + `LayerQuant` + `WeightQuant` + `AnnActivation` / `SnnActivation` | 表达不了 **per-group**、**残差的 $M,n$**、**逐层 ANN/SNN 模式选择** —— 三者都是第四章要补的 |

**关键路径是 per-group**:它同时卡住"替换图"(要融合阻断)、"不同粒度"
(PAIBox 侧要实现路径 A)和"convert"(`QuantSpec` 要能表达分组)三环,
是唯一一个横跨三环的缺口。

---

## 5. 已验证的数字,以及它们的口径

**口径比数字重要。** 下面每一行都注明了样本量与前置条件,因为这些数里有几个
**不能直接引用**。

### SNN:DVSGesture,7 个权重层,全测试集 288 样本

| | 软件 | 芯片仿真 |
| --- | --- | --- |
| 折叠后 float | 0.8438 | — |
| 本工具 int8 | **0.8507** | **0.8333** |
| 参考实现 int8(全局 ×127) | 0.8264 | 0.8194 |

27 核 / 63 核预算,流水线深度 14。芯片列**必须加 `--align-ticks`**(见 §6)。

低位宽(只降权重位宽,spiking site 没有激活观测器):w8 = 0.8507、w4 = 0.7951、
**w2 = 0.0868**(11 类随机水平是 0.091,即塌了)。已验证塌的是三值权重本身
而非阈值取整 —— 阈值不取整重测仍是 0.0868。

**未解释的遗留**:参考实现自报 float 0.8542,我们复现 0.8438,差 1 个点(约 3 张图)。
同一份检查点、同一份测试集、同样 `out.sum(0)` 判决。怀疑 dataloader 细节,未定位。
这不影响 int8 的对比结论(我们的 int8 高于**自己的** float 基线,也高于参考的 int8)。

### ANN:CIFAR-10,`models/simplecnn.py` + `ckpt/cifar10_cnn.pt`

6 层 conv+BN、2 层 FC、220 万参数。检查点报告 92.82%,用标准 CIFAR-10 统计量
复现 92.79%。折叠输入归一化后 **92.67%**(损失 0.12% / 12 张图,来自 padding 语义,
见下 §6)。

**w8a8 需 627 核 / 63 核预算,超 10 倍** —— 这条让"必须做选择"自动成立。

### ANN:MNIST,conv 32-64 + fc 256,帕累托前沿

九个配置的完整表在 `TODO.md` §P0,原始数据 `experiments/pareto_c32-64_fc256.json`。

> **这张表只跑了 512 样本**,同批 float 基线 0.9922 而全测试集是 0.9866。
> 所以表里"量化后精度超过浮点"是噪声假象,相邻点差异(如 0.9863 vs 0.9727)
> 在这个样本量下也不够稳。**引用前必须用全测试集重跑**(`TODO.md` P0.0 有这条)。

### 不可引用的数字

- **CIFAR-10 低位宽的核数**(176/112/122/76/58):受 `input_formats` 失配污染。
  唯一干净的是 w8a8 = 627 核。`deploy.py` 已把该失配改成硬错误,重跑即可。
- **`cost_model.md` §8 传导链里的数值**(权重 1.85×、激活 1.2×):来自随机权重 +
  粗糙量化的早期扫描,已被真实管线取代。结论的相对强弱不变,数值要换。
- **w2 的精度**:分类头也被量到 2-bit,而常规 PTQ 会把首尾层留在 8-bit。
  所以这个数不能当"2-bit 权重的能力上限"引用。

---

## 6. 踩过的坑

这一节大概是本文最值钱的部分。每一条都花掉了半天到两天。

### 必须记住的三个开关

**跑 SNN 芯片仿真必须加 `--align-ticks`。** 不加会按流水线深度丢帧,
0.8333 掉到 0.5625。根因:PAIBox 的编译器按流水线深度错开 `tick_start`(硬件上
数据每跨一个核花一拍,这是对的),但 `PAIIRGraph.step` 每拍新建 `node_outputs`、
按拓扑序在**同一拍内**把整张图算完,**没有逐跳延迟寄存器**。两者叠加的后果不是
"延迟"而是"丢帧":深度 k 的核在第 k 拍收到的是第 k 帧,前 k 帧从未被处理。
输出核 `tick_start=12`,20 帧只积分 8 帧。完整论证在 `cost_model.md` §7.8。
**这是 PAIBox 的缺陷,两条修法互斥,选哪条还没定**(`TODO.md` P0.0 有这条)。

**同一进程内多次编译会让 `input_formats` 静默失配。** 节点名来自进程全局计数器,
第二次编译时 `InputNode_0` 变成 `InputNode_1`,于是你传的 `input_formats` 键匹配不上,
输入位宽按默认值走,**只告警不报错**。所有扫描脚本必须每配置一个独立子进程。
`deploy.py` 已把它改成硬错误。CIFAR-10 那批低位宽核数就是这么被污染的。

**`--eval-batches 0` 才是全测试集。** 默认值是个小批量,早期所有表的
float 基线用全集、量化精度用前 512 张,造成了"量化提升精度"的假象。

### 量化侧

**融合必须在量化之前。** GPU 流程可以先标定后融合(per-channel 与 BN 同在输出通道轴,
实测整数码逐位不变),但 PAICORE 的 per-channel 限于 2 的幂比值,而 BN 折叠引入的
通道尺度 $k[c]$ 实测 1.81–5.04 倍且非 2 的幂 —— **连 per-channel 也救不回来**。
完整论证在 `cost_model.md` §7.5,这是"自研工具链必要性"的一条硬论据。
实现在 `tool/paicore/fuse.py` 的 `prepare()`。

**不要把整数模型交给 `export`。** `materialize()` 深拷贝后写整数权重,
`export()` 期望的是**浮点模型**(它自己要除 scale)。两个都读同一份 spec,
把整数模型喂给 `export` 等于量化两遍,权重全部饱和到 int8、累加器宽约 4 倍。
我在排查 SNN 掉点时踩了这个,白跑了一轮逐层对比。
注意 `materialize` 深拷贝而 `export` **原地改写**,顺序有讲究。

**2-bit 有符号实际是三值 `{-1,0,1}`。** `maxbound = (1 << (bits-1+unsigned)) - 1`
在 1-bit 有符号下得 0 直接退化,所以真 1-bit 需要无符号 `{0,1}` 方案。
这解释了早先 w1 与 w2 结果完全相同的现象。搜索空间里权重位宽取 `{2,4,8}`。

**标定方法决定低位宽能不能用。** MNIST w2 上:max 标定 0.1953、percentile 0.2598、
MSE **0.9043**。先前报告的"4→2 bit 精度断崖"几乎完全是 max 标定的假象。
而核收益最大的恰好是低位宽那一段,所以这条直接打开了论文的可做空间。

**标定要固定随机种子。** 训练 loader 的 `shuffle=True` 未固定种子时,`act_max`
在 1.48–1.60 之间漂,精度数字不可复现。已固定为 `quantize.SHUFFLE_SEED`。

**硬件的舍入方向是 half-up,`torch.round` 是 half-even。** 硬件判 `acc >= threshold`
等价于向上取整,两者在正中间的点上分歧。主机侧参考实现要按 half-up 对齐,
否则 LUT 填表会有系统偏差。`cost_model.md` §7.6 量化了代价:半个 LSB 占满量程
$2^{-(b^o+1)}$,8-bit 0.20% 可忽略,**2-bit 12.5% 不可忽略** —— 而核预算恰好
把可行区间压到低位宽。

### SNN 侧

**脉冲边界上的 `output_scale` 差一个 T,两种读法都"看起来合理"。**
SNN→SNN 边界上 `output_scale = 1.0`(一个脉冲给下游累加器贡献的数值就是 1,
因为突触算 `权重 × 脉冲`);`θ/T` 属于 **SNN→ANN 边界**,由需要解码速率的
消费者施加。记成 `θ/T` 的后果是模型掉到随机水平,**而每个单独的 scale 看起来
都完全正常**。我把它"修"错过一次。

**DVSGesture 的标签按字符串排序。** SpikingJelly 沿用 torchvision `DatasetFolder`
的约定,类别目录按字符串排,于是 `"10"` 落在索引 **2**(在 `"1"` 和 `"2"` 之间)。
按数字读目录名会置换 11 个类里的 9 个,精度掉到约 2/11。只有 `"0"` 和 `"1"`
映射正确 —— 这也解释了为什么只看前 48 个样本(恰好是这两个目录)会得到虚高的 93.75%。

**BN 折叠必须原地改写模块。** 重建 `nn.Conv2d` 会丢掉 SpikingJelly 的 `step_mode`
分派壳,**多步前向直接崩,而折叠本身不报任何错**。同理 `layer.Dropout` 不是
`nn.Dropout` 的子类(它继承 `MemoryModule`),`isinstance` 漏掉,要按类名剥离。

**modelopt 不认 SpikingJelly 的层。** 其替换逻辑要求子类**未重写 `forward`**,
而 SpikingJelly 全部重写了。它们的重写是纯分派壳(多步模式把 `[T,N,...]` 压进
批维后调同一个 `super().forward`),所以可以安全注册 —— 这就是
`spiking.py:register_spiking_layers` 存在的理由。不注册的话失败会延迟到
`build_spec` 报"模型从未被量化"。

**FX 追不进 SpikingJelly 的神经元。** 它们 dispatch 到 JIT 函数而 JIT 拒绝
`Proxy`,所以必须当叶子。**但"追不进去"和"是激活"是两件事** —— 我一度混为一谈,
结果 `AvgPool2d` 被当激活放行、除以 4 被静默忽略。正确做法是按有没有
`v_threshold` 识别神经元,而且**tracer 的叶子规则与激活识别规则必须一致**。

**评测必须在 T 步上累计。** 脉冲网络的答案是一个速率,读单个时间步不构成答案。

### 硬件语义

**输入归一化折叠只对无 padding 的卷积精确。** 原模型 padding 填的 0 是归一化域的 0
(等价像素 = mean),折叠后填的是码域的 0。误差完全集中在 1 像素边框(内部 1.9e-6
对边框 1.5),代价 92.79% → 92.67%。精确解法是主机侧用码值 `[125,123,114]` 预填充
到 34×34 再用无 padding 卷积 —— 成熟整数工具链(TFLite)正是用 zero-point 填充
消掉这一项,而**PAICORE 的 padding 是地址重映射、值恒为 0,这条标准解法在硬件上
不可用**。这是一条很好的"硬件差异导致标准方法失效"的例子。

**不折输入归一化会掉 52 个点且不报任何错。** 只用 `ToTensor` 是 0.4206 对 0.9279。
常数已钉在 `deploy.py` 并注明来源(用测试集反推确定的)。

**`Tmax` 不是计算上限,是读出模式的判据。** 公式正确但只管输出组,
且只决定 `STREAM`/`STEP`。计算侧时间步没有上限。这条曾被误解成硬约束。

**per-channel 不一定贵。** 折叠不是按层决定而是**按折叠组**,代价取决于折叠轴与
通道轴是否正交。卷积的折叠组沿空间展开、per-channel 沿通道分 scale,两轴正交
⇒ 组内 tau 恒定 ⇒ 折叠保留。实测神经元 SRAM 只 +0.9%、核数不变。
**"per-channel 代价 8 倍"这个早期结论是错的**,但构造一个折叠组跨通道的用例、
实测那条 8 倍分支真实存在,这件事还欠着。

**`M × 2^n` 不是纯 2 的幂。** 残差 shortcut 对齐需要 $s_x/s_\text{accum}$,
`quantize` 分支用一个 8 位权重 $M$(填进 depthwise 1×1 卷积)加乘法泄露 $2^n$
去逼近,等于手工搭出 8 位尾数的浮点。纯 2 的幂最坏偏差 41.4%,$M\times 2^n$
最坏约 **0.39%**,细约 100 倍。**这不能直接搬到 per-channel**($M$ 住在权重里需要
一条突触,而 per-channel 要的是同核内逐神经元的乘子),但 depthwise 1×1 的权重
本就逐通道 —— **每层多花一个核或许能换到 8 位尾数的 per-channel 重标定**,
那正是第五章要搜的"核数换精度"。前提未验证:该核需"膜电位进、膜电位出",
卷积能否作用于膜电位输入待查。

---

## 7. 三个能力面缺口的现状

第四章的主体。三者的**缺口形状完全一样**:硬件支持、后端支持、**前端说不出来或测不出来**。
这张表本身就是论点 —— 不是造一个量化器,而是补齐工具链能力面与硬件可表达集合之间的落差。

| 维度 | 硬件 | 后端 | 卡在哪 |
| --- | --- | --- | --- |
| per-channel | 支持(`leak_tau` 逐神经元移位) | 支持 | **已打通**;瓶颈转为"缺通道尺度离散的模型" |
| per-group | 支持(膜电位直接累加) | 支持(实测成立) | **前端**无法表达"这些路径分核" |
| ANN/SNN 模式 | 支持(`snn_mode` 逐核) | 支持(六种排布全部编译通过) | **方法**缺失:边界无速率解码、无阈值平衡、无时间步选择 |

**若三者都不成立,权重 `{2,4}` × 激活 `{4,8}` 只剩 4 个点,第五章的搜索无从谈起。**
所以这一节是关键路径,不是可选项。

### per-group:与 per-channel 性质完全不同

这是最容易想当然的一点。**不能套用 per-channel 的任何结论。**

per-channel 沿**输出轴**分 scale:一个神经元一个 scale,累加完成后整体重标定,
所以能被逐神经元字段(`leak_tau` 移位)吸收。per-group 沿**归约轴**分 scale:
单个神经元的点积**内部**就有多个 scale,而 32-bit 累加器已经把各组的贡献加在一起了,
**事后无法分离**。

设归约长度 $K_l$、分组大小 $G$,组数 $\Gamma = \lceil K_l/G \rceil$。两条实现路径:

**路径 A:沿归约轴拆核 + 膜电位级联。** 每个核只承担 scale 相同的那些组,核内累加出
部分和,由该核的 `leak_tau` 移位折算到公共 scale,再经 `add_potential`(膜电位直接
累加模式)汇总到下游神经元。核数随组数近似线性增长;而且部分和以膜电位形式在核间
传输,走工作帧 2 型,**每个 32-bit 膜电位需 4 帧**,NoC 流量是普通数据帧的 4 倍。

**路径 B:把 scale 差异折进权重位宽。** 若各组 scale 成 2 的幂关系,可统一到
$s_{\min}$ 表示,只需一个 scale、一个核,但值域被放大:
$b^{w\prime}_l = b^w_l + (k_{\max} - k_{\min})$,且必须落在 $\{1,2,4,8\}$ 内(C5)。

> **最重要的一条判断:在这颗芯片上,per-group 的收益是用权重位宽买的。**
> 而权重位宽恰恰是实测中核数的主导因素。所以**路径 B 很可能严格劣于直接均匀提高
> 权重位宽** —— 这是一个可验证的假设,无论结论如何都是第六章的消融内容。
> 基于这条,**决定只做路径 A**。

### per-group 路径 A:已实测成立,但要靠"假消费者"才能触发

两个实测结果,第二个是关键:

**多路相加会被融合掉。** `act(a(x1) + b(x2))` 形式的 2 路 / 4 路相加会融合成单个
`AccumulateOp`,落在**一个核**上。各路轴突进同一树突、共享一个累加器和一张 LUT,
**per-group 的 scale 差异无处安放**。融合对普通网络是对的(省核),但恰好是
per-group 的反面。

**阻断融合后路径 A 完整成立。** 给某路偏积加第二个消费者,绕开"单消费者 compute path
才吸收"的融合条件,就得到了想要的形状:

```
StandaloneCompOp_0   add_potential=NORMAL       ← 偏积,独立核,输出 POTENTIAL
StandaloneCompOp_1   add_potential=NORMAL       ← 偏积,独立核,输出 POTENTIAL
PotentialAddOp_0     add_potential=DIRECT_ADD   ← 膜电位累加,未被融合
StandaloneActOp_0    add_potential=DIRECT_ADD   ← 激活读取累加结果
```

偏积核无 LUT,故按 `has_lut ↔ ANN` 规则自动落在 SNN 模式;每个偏积核有自己的
`leak_tau`,所以 $2^k$ 的分组重标定可用。

**所以结论是:硬件支持、后端支持、前端无表达手段。** `fuse_to_offline_cores`
无条件吸收单消费者 compute path,目前只能靠"加一个假消费者"这种人为手段绕开。
另外 `add_potential` 虽然实现了,但由**图结构**驱动而非资源决策 ——
`_derive_input_add_potential_mode` 仅当 `StandaloneActOp` 的前驱处于 POTENTIAL 域时
才置 `DIRECT_ADD`。

**要做的前端能力,二者择一**(`cost_model.md` §7.4 末尾):

- 在 IR 节点上加融合阻断标注(如 `__no_fuse__`),让用户或搜索层钉住"这些路径分核"
- 提供专用的 per-group GEMM 算子,lowering 成 $\Gamma$ 个 `StandaloneCompOp`
  + `PotentialAddOp` + `StandaloneActOp`

这是一个边界清晰、规模适中的前端贡献,也是"不是硬件做不到,是工具链说不出来"的典型实例。

**当前状态**:`paibox/quant/export.py` 对 `Granularity.PER_GROUP` **显式抛
`UnsupportedConfig`**,并在错误信息里说明硬件限制与建议的替代。这是故意的 ——
早先它会把 per-group 的 scale 误当 per-channel 用,**静默算错**。宁可响亮报错。

---

## 8. 下一步做什么,以及为什么是这个顺序

详细清单在 `TODO.md`。这里只说**为什么**,因为优先级的理由比清单本身容易丢。

**原则:先把第三章做扎实(它是保底),再做估算器,最后冲搜索。不要先冲搜索。**

1. **P0.0 评测有效性** 排在最前,因为**第三章的图已经画出来了,但承载它的实验平台
   撑不住那些结论**。全测试集重跑、重测被污染的位宽网格、基线口径对齐 ——
   这些每一条都会直接改变已报告的数字。在这之前做任何拟合都是在拟合噪声。

2. **P0 拟合膨胀系数 $\eta_l$** 是这一批里最要紧的单条,因为它**阻塞 P2 的估算器**。
   `cost_model.md` §4.4 的 $n_l$ 是理想装箱下界,贪心 first-fit 实际更差,
   需要在多个模型/配置上采样真实核数去拟合 $\hat n_l = \lceil \eta_l n_l \rceil$。

3. **P1.0 能力面补齐**(现状见 §7)决定第五章有多少可搜内容,是关键路径。
   **其中 per-group 是第一优先** —— 三个缺口里它边界最清晰、后端已实测走通、
   缺的只是一个前端表达手段,是把"不是硬件做不到,是工具链说不出来"这条论点
   落到实处的最佳载体。ANN→SNN 转换最有新意但同样是 2–3 周,排在其后;
   per-channel 已打通,卡在缺合适的模型而非缺实现。

   建议配一个一天的前置测量(不阻塞实现,但决定论文里怎么写它的价值):
   **归约轴的权重尺度离散度**。per-channel 的教训是能力做完了才发现收益在噪声里,
   而归约轴散不散一次都没测过,且不像通道轴有 BN 的 $\gamma/\sqrt{\text{var}}$
   这样的先验。更值钱的那个假设是 **per-group 能否让 2-bit 权重从不可用变可用** ——
   权重位宽是核数主导因素,这个收益用核数算,比"精度涨零点几个点"好写得多。

4. **代价报告**(`TODO.md` P1)被升为第四章的首要交付。定位变了:既然扫描是
   用户行为而非工具功能,那么"装不下时说清该改什么"就是工具**能否被别人使用**的前提。
   用户钉住 w8a8、发现 627 核装不下,下一步调什么只能由报告告诉他。
   这里面还藏着一个漂亮的结论:`simplecnn` 的 maxLCN 顶到硬件上限 7,说明它是
   **神经元/LCN 受限**,而轴突地址 = 输入数 × 输入位宽,所以激活位宽是主导杠杆;
   MNIST 恰好相反,是权重受限。**同一份代价模型在两个模型上给出相反的建议** ——
   这本身就是第三章该有的结论,也说明报告必须报**绑定约束**而不是一句"超预算"。

5. **两个 backendv2 bug**(`TODO.md` P1.5)同时挡住我们和 `quantize` 分支的 ResNet,
   修好是能拿出去的上游贡献。两条都只在"一个产出被多个消费者分叉"时触发,
   顺序网络不产生这种拓扑,残差是第一次踩到。`op_node.py:587` 那条三行可修。

**不要当主线做的**:P4 的"前后端解耦契约"是被指派的活,压缩到够用就行,
论文里占一节,定位"实验平台建设"。

---

## 9. 仍未回答的问题

- PAIBox 的时序语义以哪一侧为基准?给仿真器加逐跳延迟,还是仿真路径上不错开
  `tick_start`?选前者要顺带确认 `tick_relative` / 工作帧的 timestep 语义。
- 折叠神经元的空间折叠规则能表达哪些 layout?(影响 $\tau_l$ 与 per-channel 代价链)
- `csc_accelerate=1` 占用 `vjt_initial` 后,是否影响任何现有部署路径?
- 多线程(`thread_number`)下各线程独立时间步,能否用来缓解 `STEP` 读出的吞吐损失?
- 卷积能否作用于膜电位输入?(决定 `M × 2^n` 能否用于 per-channel 重标定)
- per-group 路径 B 是否真的严格劣于均匀提高权重位宽?(§7 的核心假设,第六章消融项)
- per-group 路径 A 的膜电位级联,4 倍 NoC 流量在什么规模下会成为瓶颈?
- 我们的 DVSGesture float 基线 0.8438 与参考自报的 0.8542 差 1 个点,来源未定位。

---

## 10. 环境与数据

**环境**:uv 管的 Python 3.14,`uv sync` 按 `uv.lock` 重建。不要复制 `.venv`。

**数据集都不在仓库里**(`.gitignore` 排除),因为都是可获取的:

| 数据 | 位置 | 怎么拿 |
| --- | --- | --- |
| MNIST | `experiments/data/` | torchvision 自动下载 |
| CIFAR-10 | `experiments/ckpt/cifar10_data/`(npy) | 缺失时自动回退到 torchvision 下载 |
| DVS 帧 | `DVSGesture_sw_and_rt/data/DVS128Gesture/frames_number_20_split_by_number/` | **需从 IBM 获取原始数据集后用 SpikingJelly 重新积分**,约 209M |

CIFAR-10 有个口径风险:本机 npy 是从 HuggingFace parquet 导出的,torchvision 走的是
官方 python-batches 版本。测试集内容与顺序应当一致,但这是推断而非实测 ——
若精度对不上 92.67%,先查这里。

**训练权重不在仓库里**(二进制且大):`experiments/ckpt/cifar10_cnn.pt`、
`DVSGesture_sw_and_rt/checkpoints/*.pt`。这些**不可从仓库重新生成**,要单独保管。

**四个芯片 PDF 文档**经 `.git/info/exclude` 本地排除(内部文档,而 PAIBox 的 upstream
是公开仓库)。它们是 C1–C12 和 §7.8 的依据,**需要单独保管**:
`2.5芯片应用说明文档.pdf`、`2.5芯片离线核说明文档.pdf`、`钟毅_2.5神经元设计.pdf`、
`钟毅_激活函数.pdf`。

**`quantize` 分支没有克隆下来**。它的历史里有约 280M 提交进去的数据集
(MNIST 原始文件、CIFAR-10 batches、几个 `.pth`),所以只取了工作分支。
它在 GitHub 上(`origin/quantize`),`git fetch origin quantize` 即可取回。
其结论已提取进 `TODO.md`,作为代码对跑实验没用,作为论文的对照/前人工作有用。
