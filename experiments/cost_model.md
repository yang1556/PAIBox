# PAICORE 2.5 离线核资源代价模型与约束形式化

本文把量化决策到芯片资源占用的传导关系写成可计算的形式,用途有三:

1. 作为快速核数估算器的规格说明(搜索循环里不能每次都跑 CP-SAT)
2. 作为逐层决策搜索的约束集合
3. 作为论文第三章"资源模型与传导分析"的骨架

出处标注:`[手册]` = 《PAICORE 2.5 应用手册 v1.0》,`[离线核]` = 《2.5 芯片离线核说明文档》,
`[神经元]` = 《钟毅_2.5神经元设计》,`[激活]` = 《钟毅_激活函数》,`[lib]` = `paicorelib 2.0.0`,
`[PAIBox]` = `paibox/backendv2/` 实现。标注 `⚠待实测` 的是文档未明确、需要用编译实测校准的部分。

---

## 1. 硬件常量

| 符号 | 值 | 含义 | 出处 |
| --- | --- | --- | --- |
| $N_{\text{core}}$ | 63 | 单芯片离线核数(另有 17 在线核、1 RISC-V CPU,共 81) | [手册 1.5][lib] |
| $S_{\text{core}}$ | 4096 | 单核弹性 SRAM 地址数,每地址 128 bit,神经元与权重共用 | [手册 2.3.3] |
| $F_{\text{base}}$ | 512 | LCN 1 倍时的扇入基数(轴突地址数) | [手册 2.3.1][PAIBox] |
| $B_{\text{addr}}$ | 17 | 工作帧中 Time Step 位宽 + Axon 位宽之和 | [手册 2.2.3] |
| $l_{\max}$ | 7 | LCN 指数上限(LCN = $2^{l}$,最大 128×) | [手册 2.3.1] |
| $N_{\text{neu}}^{\max}$ | 4096 | `neuron_number` 上限,以半神经元为单位计 | [手册 2.3.1] |
| $\mathcal{B}$ | $\{1,2,4,8\}$ | 输入/输出/权重可选位宽 | [手册 2.3.1] |

**神经元存储开销**(单位:128-bit SRAM 地址数/神经元) [手册 2.3.3]

$$\rho(\tau) = \begin{cases} 2 & \tau = \text{full 全神经元(256 bit)} \\ 1 & \tau = \text{half 半神经元(128 bit)} \\ 1/4 & \tau = \text{folded 折叠神经元(32 bit)} \end{cases}$$

**权重每行容纳数**(128-bit 行) [手册 2.3.3]

| $b^w$ | 稠密 $W_{\text{dense}}$ | CSC $W_{\text{csc}}$ | CSC 盈亏平衡非零率 |
| --- | --- | --- | --- |
| 1 | 128 | 7 | 5.5% |
| 2 | 64 | 7 | 10.9% |
| 4 | 32 | 6 | 18.8% |
| 8 | 16 | 5 | 31.3% |

**参数值域**(超出即不可部署) [手册 2.3.3]

| 参数 | 位宽 | 范围 | 承载的量化语义 |
| --- | --- | --- | --- |
| `vjt` 膜电平 | 32/s | $\pm(2^{31}-1)$ | 累加器 |
| `threshold_pos/neg` | 32/s | $\pm(2^{31}-1)$ | 阈值 |
| `leak_v` | 20/s | $[-2^{19}, 2^{19}-1]$ | **bias 折叠目标** |
| `reset_v` | 16/s | $[-2^{15}, 2^{15}-1]$ | 复位电平 |
| `vjt_initial` | 12/s | $[-2^{11}, 2^{11}-1]$ | 初始膜电平,`csc_accelerate=1` 时被占用 |
| `leak_tau` | 6/s | $[-32, 31]$ | **仅 $2^k$ 缩放** |

**激活 LUT** [手册 2.3.2][激活]:每核一张,256 项,阈值 32/s + 激活值 8 bit。
SAR 逐次逼近,输出 8/4/2/1 bit 分别比较 8/4/2/1 次。阈值间距完全自由,
最大量化误差 $\le 0.5$ LSB,与软件四舍五入一致。

**AER 多播面积** [lib `aer_packet_area`]

$$\alpha(z,x,y) = 1 + z + (1+z)x + (1+z+x)y, \quad |z|\le 6,\ |x|\le 8,\ |y|\le 6$$

可达面积在 $[1,63]$ 内近乎稠密,缺失值仅 $\mathcal{M} = \{11,17,31,41,53,55,59\}$。

---

## 2. 决策变量

网络有 $L$ 层,层 $l$ 的决策为:

| 变量 | 取值 | 含义 |
| --- | --- | --- |
| $m_l$ | $\{\text{SNN}, \text{ANN}\}$ | 计算模式(`SNN_ANN` 核参数) |
| $b^w_l$ | $\mathcal{B}$ | 权重位宽 |
| $b^o_l$ | $\mathcal{B}$ | 输出(激活)位宽 |
| $g_l$ | $\{\text{per-tensor}, \text{per-channel}, \text{per-group}(G)\}$ | 量化粒度($G$ = 归约轴分组大小) |
| $f_l$ | $\{\text{dense}, \text{csc}, \text{shared}\}$ | 权重存储格式 |
| $e_l$ | $\{\text{lcn}, \text{add\_potential}\}$ | 扇入扩展方式 |

输入位宽不是自由变量,由前驱决定:$b^i_l = b^o_{l-1}$。

---

## 3. 网络给定量

| 符号 | 含义 |
| --- | --- |
| $A_l$ | 层 $l$ 单个输出神经元的输入元素数(扇入) |
| $N_l$ | 层 $l$ 的输出神经元总数 |
| $C_l$ | 输出通道数 |
| $K_l$ | 单个输出神经元对应的权重个数 |
| $\sigma_l$ | 共享同一套权重的神经元数(FC: $\sigma_l = 1$;Conv: $\sigma_l = H^{\text{out}}_l W^{\text{out}}_l$) |
| $s_l$ | 权重非零率 |

---

## 4. 派生量

### 4.1 LCN 与时间步(核心传导链)

轴突地址需求:

$$A^{\text{addr}}_l = A_l \cdot b^i_l = A_l \cdot b^o_{l-1}$$

LCN 指数([PAIBox] `routing.py:446` 的实现):

$$l_l = \left\lceil \log_2 \left\lceil \frac{A^{\text{addr}}_l}{F_{\text{base}}} \right\rceil \right\rceil, \qquad l_l \le l_{\max} = 7$$

### 4.1.1 LCN 与时间步:是读出模式,不是计算上限

工作帧把 17 位分给 Time Step 与 Axon 地址,比例随 LCN 变化(8:9 至 1:16)[手册 2.2.3],
于是时间步字段宽度为 $8 - l$,可表示

$$T^{\max}(l) = 2^{\,8 - l}$$

**但这个上限只作用于输出读出,不限制计算。** [PAIBox] `compile_artifacts.py:404`:

```python
max_stream_timesteps = min(1 << (8 - out_grp.lcn.value) for out_grp in output_groups)
return DECODE_MODE_STREAM if max_stream_timesteps >= timesteps else DECODE_MODE_STEP
```

要点三条:

1. 取的是**输出组的 LCN** $l_{\text{out}}$,不是全网 $\min_l$。计算核的
   `RoutingGroup.set_lcn()`(`routing.py:427`)只做位宽一致性断言,**不做时间步检查**。
2. 超出不是错误,而是把读出模式从 `STREAM` 降级为 `STEP`
   (`DecodeMode` 枚举见 `schemas/compile_artifacts.fbs:20`)。
3. 计算侧之所以无上限:核间由 `sync_all` 全局同步,帧里那 8 位承载的是
   **slot 索引 `tick_relative`**,而非全局步计数。

$$\text{readout} = \begin{cases}
\text{STREAM} & T \le 2^{\,8 - l_{\text{out}}} \\
\text{STEP} & \text{否则}
\end{cases}$$

**STEP 模式的代价是吞吐,不是可行性**:host 需在每个 `sync_all` 后读一次,
每个时间步多一次主机往返。计入 §6.2 的吞吐模型,不进 §5 的硬约束。

由此得到一条比"时间步不够用"更细的传导链:
**激活位宽 → LCN → 读出模式 → 主机往返次数 → 吞吐**。

`RuntimeParams` 已经导出了 `timesteps`、`tick_depth`(= 流水线深度)、`sync_steps`、
`decode_mode`,所以这些量不必自己推算,读 metadata 即可。

### 4.2 神经元类型

折叠不是按层决定的,而是按**折叠组**决定:`routing.py:652` 要求同组神经元的
Part 2 属性**完全相同**,任一逐神经元字段有差异即放弃折叠、退回全神经元。
而 `leak_tau` 在 `NeuronParams.__vectorized_attrs__` 中,`attrs_part2(idx)` 会
按神经元下标取值([PAIBox] `op_node.py:413`),所以逐通道移位确实进入了这个比较。
判定因而取决于**折叠轴与分 scale 轴是否正交**:

$$\tau_l = \begin{cases}
\text{folded} & \text{折叠规则可表达} \ \wedge\ \text{每个折叠组内 } \texttt{leak\_tau} \text{ 恒定} \\
\text{full} & \text{折叠组跨越了不同的 scale}\ \ (\text{Part 2 不一致}) \\
\text{half} & \text{折叠规则不可表达}
\end{cases}$$

神经元 SRAM 占用($n$ 个神经元折叠成一组时为 $3 + \lceil (n-1)/4 \rceil$ 行,
对照未折叠的 $2n$ 行,[PAIBox] `neuron.py:99`):

$$S^{\text{neu}}_l = N_l \cdot \rho(\tau_l)$$

**per-channel 的代价是条件性的,不是恒定 8 倍。** 卷积层的折叠组沿**空间**展开,
而 per-channel 沿**通道**分 scale,两轴正交,组内 $\texttt{leak\_tau}$ 恒定,折叠得以保留。
只有当折叠组跨通道时才退化为全神经元,付出 $\rho(\text{full})/\rho(\text{folded}) = 8$ 倍
(神经元参数 SRAM 占单核面积 59% [手册 1.5])。

实测(`experiments/pareto.py --granularities`,conv 32-64 + fc 256,w8a8):
per-tensor 与 per-channel 的折叠成功次数同为 260,神经元 SRAM 15174 → 15307 行
(**+0.9%**),总核数同为 24。**这条链在卷积主导的网络上几乎不产生代价。**

### 4.3 权重 SRAM

单套权重的行数:

$$\text{lines}(f_l, b^w_l) = \begin{cases}
\left\lceil K_l \cdot b^w_l / 128 \right\rceil & f_l = \text{dense} \\
\left\lceil s_l K_l / W_{\text{csc}}(b^w_l) \right\rceil & f_l = \text{csc} \\
\end{cases}$$

$$S^{\text{wgt}}_l = \left\lceil N_l / \sigma_l \right\rceil \cdot \text{lines}(f_l, b^w_l)$$

`shared` 不是独立的行数公式,而是通过 $\sigma_l > 1$ 体现。

CSC 仅当 $s_l < W_{\text{csc}}(b^w_l) \cdot b^w_l / 128$ 时优于稠密。

### 4.4 核数

理想装箱下层 $l$ 所需核数:

$$n_l = \left\lceil \frac{S^{\text{neu}}_l + S^{\text{wgt}}_l}{S_{\text{core}}} \right\rceil$$

**⚠待实测:** 这是**下界**。[PAIBox] `RoutingGroup.try_store_neuron` 是 first-fit 贪心,
且换核时 `stored_base_weight.clear()` 会丢弃跨核权重复用机会,实际值 $\ge n_l$。
需要用编译实测拟合一个膨胀系数 $\eta_l \ge 1$,使 $\hat{n}_l = \lceil \eta_l \cdot n_l \rceil$ 逼近真实值。

### 4.5 AER 放置占用

$$\text{occ}(n) = \min\{\alpha : \alpha \in \text{Range}(\alpha),\ \alpha \ge n\} = \begin{cases} n+1 & n \in \mathcal{M} \\ n & \text{否则}\end{cases}$$

取整浪费很小(实测全网 0%),不是主要矛盾——这一点与 v1 的四叉树 NoC(`1 << (n-1).bit_length()`,2 的幂取整)有本质区别。

---

## 5. 硬约束

**C1 单核容量** $\quad S^{\text{neu}}_l + S^{\text{wgt}}_l \le S_{\text{core}} \cdot n_l$,且单核 `neuron_number` $\le 4096$(半神经元当量)

**C2 芯片核预算** $\quad \sum_l \text{occ}(\hat{n}_l) \le N_{\text{core}} = 63$(单片)或 $252$(`array_2x2`)

**C3 扇入上限** $\quad l_l \le 7$,即 $A_l \cdot b^o_{l-1} \le 512 \times 128 = 65536$。超出时必须用 $e_l = \text{add\_potential}$ 级联多核累加膜电平 [手册 2.3.1]

**C4 时间步** $\quad$ **不是硬约束。** $T > 2^{8-l_{\text{out}}}$ 只是把读出模式降为 `STEP`,
代价计入吞吐(§4.1.1、§6.2)。计算侧无时间步上限。唯一的硬性下限来自流水线:
$T \ge \text{tick\_depth}$,否则输出尚未填满,仿真与硬件都返回零值

**C5 位宽核级统一** $\quad b^w_l, b^i_l, b^o_l$ 是 Core 参数中的 2-bit 字段,同核内不可混用;取值仅 $\{1,2,4,8\}$

**C6 分支位宽一致** $\quad$ 多输入层(concat/add)的所有前驱必须有相同输出位宽:$b^o_{p} = b^o_{q}$ for all $p,q \in \text{pred}(l)$ 。[PAIBox] `routing.py:436` 有断言强制

**C7 bias 值域** $\quad |\text{bias}^q_{l,c}| \le 2^{19}-1$(折叠进 `leak_v`,20/s)

**C8 激活重标定** $\quad$ 每核仅一张 LUT,同核所有输出通道共享同一条 requant∘激活曲线。
故 per-channel 激活量化不可直接表达,per-channel scale 必须折进权重

**C9 输出位宽上限** $\quad b^o_l \le 8$;SNN 模式下 $b^o_l \in \{1, 2\}$(`{0,1}` 或 `{-1,0,+1}`)

**C10 乘性重标定的可表达集合** $\quad$ **分两种情形,不是一句"仅 $2^k$"。**

*同核内、逐神经元*(per-channel 权重 scale 的场合):只有 `leak_tau` 移位,
$k \in [-32,31]$,即**只能是 2 的幂**。因为同核所有神经元共享一张 LUT(C8),
而逐神经元可用的字段里唯一的乘性机构就是这个移位。最坏舍入偏差
$\sqrt2 - 1 = 41.4\%$(实测 `pc-round` 为 38.9–41.4%,与理论上界吻合)。

*跨核、经过一条突触*(残差对齐、或愿意为某层多花一个核的场合):比例可写成
$M \cdot 2^n$,其中 $M$ 是权重码($1 \le M \le 255$,8 位无符号)、$2^n$ 是乘法泄露。
等于手工搭出 8 位尾数的浮点,每个八度内有 128–255 个格点,**最坏舍入偏差约 0.39%,
比纯 $2^k$ 细约 100 倍**。代价是那条路要占一个核。

`quantize` 分支的残差对齐用的正是后者:$s_x/s_{\text{accum}} \approx M \cdot 2^n$,
$M$ 填进 depthwise $1{\times}1$ 卷积的权重、$n$ 走乘法泄露,而 shortcut 本来就要占一个核,
所以 $M$ 是白搭的。举例:实测 `block1.2` 处若有残差,$\text{Ratio} \approx 677$,
纯 $2^k$ 舍到 512 偏差 24.4%,而 $M{=}169, n{=}2$ 得 676、偏差 0.16%。

**这留下一个待验的可能**:depthwise $1{\times}1$ 的权重本就逐通道,所以每层多花一个核
或许能换到 8 位尾数的 per-channel 重标定,把 §7.3 的"必须是 2 的幂"变成一个
**核数换精度**的权衡 —— 那正是第五章要搜的那类选择。未验证的前提是该核需
"膜电位进、膜电位出",卷积能否作用于膜电位输入待查。

任意 scale 若两条都不用,则必须由 LUT 承担(阈值位置自由,§7.6),或折进权重

**C11 初始膜电平** $\quad |\text{init}_l| \le 2^{11}-1$,且 `csc_accelerate=1` 时该字段被 `weight_address_start` 占用,不可用

**C12 模式边界** $\quad m_l = \text{SNN} \wedge m_{l+1} = \text{ANN}$ 时需在边界累积脉冲序列为数值,代价为 $T$ 步延迟;
只有 ReLU 类(非负、无上界)激活可做 ANN→SNN 转换

---

## 6. 目标函数

主目标(核预算受限场景):

$$\min_{\{m_l, b^w_l, b^o_l, g_l, f_l, e_l\}} \sum_l \text{occ}(\hat{n}_l) \quad \text{s.t.} \quad \text{Acc}(\cdot) \ge \text{Acc}_{\text{target}},\ \text{C1--C12}$$

对偶形式(核预算固定):

$$\max \text{Acc}(\cdot) \quad \text{s.t.} \quad \sum_l \text{occ}(\hat{n}_l) \le 63,\ \text{C1--C12}$$

### 6.1 面积与功耗代理

单核物理面积 0.36 mm²,功耗约 9.07 mW [手册 1.5, 2.4],故

$$\text{Area} \approx 0.36 \sum_l \text{occ}(\hat{n}_l) \ \text{mm}^2$$

功耗中神经元计算占 73%、路由占 17% [手册 1.5],神经元更新次数正比于 $T \sum_l N_l$,
路由事件数正比于非零输出个数(受 `zero_output` 与激活稀疏度影响),故

$$\text{Energy} \propto 0.73 \cdot T \sum_l N_l \ +\ 0.17 \cdot \sum_l T \cdot N_l \cdot (1 - z_l)$$

其中 $z_l$ 为层 $l$ 输出的零值比例。**激活稀疏度在此芯片上是直接的能耗收益**,这是标准量化不考虑的维度。

### 6.2 吞吐

$$\text{Throughput} \propto \frac{1}{T \cdot D}$$

$D$ 为流水深度。[神经元] 给出的 LeNet 实测印证了核数-吞吐权衡:同一网络 20 核 / 86 fps 对 56 核 / 650 fps。

[PAIBox] `compute_pressure.py` 已有一个计算压力代理指标
$\text{pressure} = \text{fold\_count} \times b^i \times b^w \times \text{weight\_slots}$,
并被 `pressure_unroll.py` 用来拆分高压力核。这个指标可直接作为 $D$ 的估计,不必另造。

---

## 7. 能力面与搜索层的分离

量化工具必须能忠实执行用户指定的配置,**即使它在资源上是浪费的**。这不只是易用性:
第六章的基线(均匀 int8、纯精度驱动的混合精度、人工配置)都是用户指定的配置,其中多数
资源次优。只能产出最优解的工具构造不出基线。让基线与本方法走同一条代码路径,也消除了
"基线实现被削弱"这一类质疑。

因此工具分两层,职责不可混:

**可行性层(能力面)** 回答"这个配置在硬件上能不能表达",即 §5 的 C1--C12。这一层
不做任何优化判断:能表达就照做,不能就报错并指出是哪条硬件约束。

**搜索层** 在可行集合里优化 §6 的目标函数。

### 7.1 决策变量的三态

§2 的每个决策变量处于三种状态之一:

| 状态 | 含义 | 在搜索中的形式 |
| --- | --- | --- |
| free | 由搜索决定 | ILP 中的自由 0-1 变量组 |
| pinned | 用户钉死 | 固定该变量组的取值,解剩余变量 |
| ranged | 用户约束范围 | 追加线性约束,如 $b^w_l \ge 4$ |

用户钉住的决策**就是给搜索层加约束,不是特例分支**。固定变量再求解是 ILP 的原生能力,
所以代码里不应出现"用户模式"与"搜索模式"两条路径。

### 7.2 浪费 vs 不可行

这两件事对用户的意义不同,工具的反应也必须不同。

**浪费但可行** —— 照做,并把代价显式报出来。典型是 per-channel 的逐神经元 `leak_tau`:
可以表达,但当折叠组跨通道时迫使 $\tau_l = \text{full}$,付出 8 倍神经元 SRAM(§4.2)。
代价是否真的发生取决于折叠轴与通道轴是否正交,所以**必须实测而不能预设**——
在卷积主导的网络上实测仅 +0.9%。工具应报告"该选择使总核数从 $n$ 增至 $n'$"。

**不可表达** —— 报错,说明硬件原因,并给出最近的可行替代。典型是 **per-channel 激活量化**:
每核仅一张 LUT(C8),同核所有输出通道必须共享同一条 requant∘激活曲线。这不是浪费,
是表达不出来;替代方案是把 per-channel scale 折进权重,或令每核只放一个通道。

按这个二分法归类 §5 的约束:

| 约束 | 类别 | 违反时的行为 |
| --- | --- | --- |
| C1 单核容量 | 代价 | 计入 $n_l$,不报错 |
| C2 芯片核预算 | 不可行 | 报错:超出 63 核 |
| C3 扇入上限 | 不可行,有替代 | 提示改用 $e_l = \text{add\_potential}$ |
| C4 时间步 | 代价 | $T > 2^{8-l_{\text{out}}}$ 降为 `STEP` 读出,报告吞吐代价;$T < \text{tick\_depth}$ 报错 |
| C5 位宽取值域 | 不可行 | 报错:仅支持 $\{1,2,4,8\}$ 且核内统一 |
| C6 分支位宽一致 | 不可行 | 报错:指出冲突的前驱对 |
| C7 bias 值域 | 不可行 | 报错:超 `leak_v` 20/s 范围 |
| C8 激活重标定 | 不可表达 | 报错:per-channel 激活无硬件载体 |
| C9 输出位宽上限 | 不可行 | 报错:ANN $\le 8$,SNN $\in \{1,2\}$ |
| C10 乘性重标定 | 近似,代价分档 | 同核内仅 $2^k$(最坏 41.4%);跨核可用 $M\cdot2^n$(最坏 0.39%,+1 核)。见 §7.3 |
| C11 初始膜电平 | 不可行 | 报错:超 12/s,或与 `csc_accelerate` 冲突 |
| C12 模式边界 | 语义 | 需转换方法;编译通过不等于数值正确 |

**"报错时指出是哪条硬件约束"是通用量化工具给不了的能力**,也是本工具能力面刻画的价值所在。

量化粒度按此二分法的归属:per-tensor 无额外代价;per-channel 属"浪费但可行"(代价条件性,
最坏 8 倍神经元 SRAM,§7.3);per-group 视实现路径而定(§7.4),路径 B 受 C5 位宽取值域约束,路径 A 受
`add_potential` 策略是否实现约束;**per-channel 激活属"不可表达"**(C8)。

### 7.3 per-channel 权重的三种模式

per-channel 权重 scale 使每个输出通道的累加器单位为 $s_w[c] \cdot s_{\text{in}}$,于是每通道
需要不同的 LUT 比率 $r[c]$。但 LUT 是核级共享的(C8),一张表装不下多个比率。唯一出路是
用逐神经元字段吸收差异,而逐神经元可用的只有 `leak_tau`(移位,仅 $2^k$)与 `leak_v`
—— **两者都在全神经元 Part 2 中**。

结论:**per-channel 权重量化要求通道间 scale 比值为 2 的幂。** 它是否进一步迫使全神经元,
取决于折叠组是否跨通道(§4.2)——卷积层通常不跨,因此并非必然付出全神经元代价。

用户请求 per-channel 时,工具提供三种行为供选择:

**严格** —— 将 per-channel scale 约束到 2 的幂比值求解,报告由此引入的量化误差。

**近似** —— 用无约束 per-channel scale 求解,再四舍五入到最近的 2 的幂,报告偏差。

**拒绝** —— 偏差超过用户阈值时报错。

这三种行为的精度-资源差异本身是第六章可测的内容。三者均已实现并可测量
(`PerChannelPolicy.STRICT` 配 `Pow2MseCalibrator` 为严格,`APPROXIMATE` 配
`MseCalibrator` 为近似,`export.py` 通过 `_channel_shift_kwargs` 把比值写成
逐神经元 `leak_tau_shift`,`leak_multi_sequence = BEFORE_COMPARE` 使移位发生在查表前)。

首轮实测(conv 32-64 + fc 256,a8,512 样本)显示**通道间 scale 的动态范围极窄**:
逐层移位仅落在 $[-3, 0]$,每层只有 2-4 个不同取值。也就是说该网络的通道尺度差异
不足 3 个八度,per-channel 本就没有多少可赚的余量,精度差异全部落在采样噪声内。
**per-channel 的收益上界由通道尺度离散度决定**,这个量应当在决定粒度前先测。

### 7.4 per-group(GEMM 分组)量化的两条实现路径

per-group 与 per-channel 的性质**完全不同**,不能套用 §7.3 的结论。

per-channel 沿**输出轴**分 scale:一个神经元一个 scale,累加完成后整体重标定,所以能被
逐神经元字段吸收。per-group 沿**归约轴**分 scale:单个神经元的点积**内部**就有多个
scale,而 32-bit 累加器已经把各组的贡献加在一起,**事后无法分离**。

设层 $l$ 的归约长度为 $K_l$,分组大小 $G$,则组数 $\Gamma = \lceil K_l / G \rceil$,
各组 scale 为 $s_w[\gamma]$。

**路径 A:沿归约轴拆核 + 膜电平级联**

每个核只承担 scale 相同的那些组,核内累加出部分和,由该核的 `leak_tau` 移位把部分和
折算到公共 scale,再经 `add_potential`(膜电平直接累加模式)汇总到下游神经元。

代价:

$$n_l^{A} \ge \Gamma \cdot \left\lceil \frac{S^{\text{neu}}_l + S^{\text{wgt}}_l / \Gamma}{S_{\text{core}}} \right\rceil$$

即核数随组数近似线性增长。此外部分和以膜电平形式在核间传输,走工作帧 2 型,
**每个 32-bit 膜电平需 4 帧**(手册 2.2.3),NoC 流量是普通数据帧的 4 倍。

**路径 B:把 scale 差异折进权重位宽**

若各组 scale 成 2 的幂关系 $s_w[\gamma] = 2^{k[\gamma]} s_{\min}$,可把所有组统一到
$s_{\min}$ 表示:

$$W_q'[\gamma] = W_q[\gamma] \cdot 2^{\,k[\gamma] - k_{\min}}$$

此时只需一个 scale、一个核,但值域被放大,所需权重位宽变为

$$b^{w\prime}_l = b^w_l + \underbrace{(k_{\max} - k_{\min})}_{\text{scale 跨度(log}_2\text{)}}$$

而 $b^{w\prime}_l$ 必须落在 $\{1,2,4,8\}$ 内(C5)。

**这给出一个重要判断:在这颗芯片上,per-group 的收益(更细的 scale)是用权重位宽买的。**
而权重位宽正是实测中核数的主导因素(§8 链 1)。所以路径 B 下的 per-group 很可能
**严格劣于直接均匀提高权重位宽** —— 这是一个可验证的假设,无论结论如何都是第六章的内容。

**选择规则**

$$g_l = \text{per-group}(G) \Rightarrow \begin{cases}
\text{路径 B} & k_{\max} - k_{\min} \le 8 - b^w_l \ \wedge\ \text{scale 可对齐到 } 2^k \\
\text{路径 A} & \text{否则(代价按上式计入 } n_l\text{)} \\
\text{报错} & e_l \ne \text{add\_potential} \ \wedge\ \text{路径 B 不可用}
\end{cases}$$

**路径 A 的可行性(已实测)**

`add_potential` 的策略**是实现了的**,但由**图结构**驱动而非资源决策:
`passes.py:1853 _derive_input_add_potential_mode` 仅当 `StandaloneActOp` 的前驱处于
POTENTIAL 域时置 `DIRECT_ADD`;`add_ops.py:243` 为 `PotentialAddOp` 置 `DIRECT_ADD`。

实测两种情形:

**多路相加会被融合掉。** `act(a(x1) + b(x2))` 形式的 2 路 / 4 路相加融合成单个
`AccumulateOp`,落在**一个核**上(`n_comps=2/4`、`n_weights=2/4`、`add_potential=NORMAL`,
LCN 随总轴突数从 1 涨到 2)。各路轴突进同一树突、共享一个累加器和一张 LUT,
**per-group 的 scale 差异无处安放**。融合对普通网络是对的(省核),但恰是 per-group 的反面。

**阻断融合后路径 A 完整成立。** 给某路偏积加第二个消费者以绕过
"单消费者 compute path 才吸收"的融合条件后,得到:

```
StandaloneCompOp_0   add_potential=NORMAL       ← 偏积,独立核,输出 POTENTIAL
StandaloneCompOp_1   add_potential=NORMAL       ← 偏积,独立核,输出 POTENTIAL
PotentialAddOp_0     add_potential=DIRECT_ADD   ← 膜电位累加,未被融合
StandaloneActOp_0    add_potential=DIRECT_ADD   ← 激活读取累加结果
```

偏积核无 LUT,故按 `has_lut ↔ ANN` 规则自动落在 SNN 模式。每个偏积核有自己的
`leak_tau`,因此 $2^k$ 的分组重标定是可用的。

**结论:硬件支持、后端支持,前端无表达手段。** `fuse_to_offline_cores` 无条件吸收
单消费者 compute path,目前只能靠"加一个假消费者"这种人为手段绕开。

**因此路径 A 需要新增前端能力**,二者择一:

- 在 IR 节点上加融合阻断标注(如 `__no_fuse__`),让用户或搜索层能钉住"这些路径分核"
- 提供专用的 per-group GEMM 算子,lowering 成 $\Gamma$ 个 `StandaloneCompOp`
  + `PotentialAddOp` + `StandaloneActOp`

这是一个边界清晰的前端贡献,规模适中,且是 §7.2 意义上"能力面缺口"的典型实例:
不是硬件做不到,是工具链说不出来。

### 7.5 算子融合必须前置:GPU 依赖的交换律在这颗芯片上失效

这一条是"为什么需要自研量化工具"最干净的论据,因为它不是"通用工具做得不够好",
而是**通用工具的一条正确性前提在这颗芯片上不成立**。

**通用流程为什么可以后融合。** NVIDIA 的路径是:带 BN 训练 → modelopt 插入量化器并标定
→ 导出带 Q/DQ 的 ONNX → TensorRT 构建引擎时才融合 conv+BN。标定发生在**未融合**的模型上。
这之所以安全,是因为卷积权重默认 per-output-channel 量化,而 BN 折叠正是逐输出通道乘
$k[c] = \gamma[c]/\sqrt{\text{var}[c]+\epsilon}$。两者作用在**同一根轴**上,于是 $k[c]$ 被
scale 完整吸收,整数码不变:

$$W'[c] = k[c]\,W[c],\quad s'_w[c] = k[c]\,s_w[c] \ \Longrightarrow\ \frac{W'[c]}{s'_w[c]} = \frac{W[c]}{s_w[c]}$$

所以**融合与量化的可交换性,取决于量化粒度是否覆盖了融合所作用的轴**。这也解释了
vendored modelopt 里为什么根本没有 conv-BN 融合,只把 `nn.BatchNorm*` 列进
`_default_disabled_quantizer_cfg`(不量化、忽略它)——它把融合留给了下游的 TensorRT。
而我们的下游是 PAIBox,它假设交进来的模型**已经是芯片语义**,不做融合。这一格是空的。

**实测(`ckpt/cifar10_cnn.pt`,6 层 conv+BN,92.82%):**

| 卷积 | $k[c]$ 跨度 | per-channel 码是否不变 | per-tensor 最大码差 | per-tensor 码变动比例 |
| --- | --- | --- | --- | --- |
| 0 | 5.04× | 是 | 40 | 94.1% |
| 1 | 2.49× | 是 | 40 | 69.5% |
| 2 | 2.18× | 是 | 30 | 64.0% |
| 3 | 1.81× | 是 | 25 | 78.1% |
| 4 | 2.21× | 是 | 45 | 64.0% |
| 5 | 2.57× | 差 1(浮点舍入) | 24 | 62.9% |

per-tensor 下 `max|W|` 最多变动 **4.41×**,于是 62%–94% 的权重码不同、最大码差 40
(码域仅 ±127)。先标定的 $s_w$ 量的是折叠前的权重,却被用来除折叠后的权重。

**关键转折:PAICORE 上连 per-channel 也救不回来。** 通道间 scale 比值在**同核内**只能由
每个神经元的 `leak_tau` 移位表达,即只能是 2 的幂(C10 第一档)。而实测 $k[c]$ 的跨度是
1.81–5.04 倍、逐通道取连续值,不是 2 的幂。所以 per-channel scale 只能吸收 $k[c]$ 舍入到
2 的幂之后的部分,残差无处安放。(C10 第二档的 $M\cdot2^n$ 要额外占一个核,不属于
"先标定后融合"这条路径能白拿的东西。)

$$\boxed{\text{在 PAICORE 上,先融合再标定是唯一正确的顺序 —— per-tensor 与 per-channel 皆然。}}$$

这与 C8(每核一张 LUT)、C10(缩放仅 $2^k$)同属一条主线:**硬件把"任意逐通道乘法"
这个通用量化工具默认拥有的能力拿掉了**,于是一系列在 GPU 上无需考虑的前提都要重新检验。

**对工具架构的结论:** 融合属于量化入口的职责(与 PyTorch `fuse_modules`、ONNX Runtime
的做法一致),否则用户交一个带 BN 的模型进来会**静默**拿到错的 scale。但它属于
`tool/paicore/` 而非通用的 `tool/mytorch/`——"哪些算子必须折"是目标相关的。
实现于 `tool/paicore/fuse.py` 的 `prepare()`,含三个等价变换:BN 折叠、输入归一化折叠
(仿射,沿**输入**通道,故不产生逐通道输出尺度差,修正项全进 bias)、推理无操作剥离
(`Dropout`,lowering 无对应节点)。顺序上归一化折叠最后做,因为它作用于第一层卷积,
而 BN 折叠可能已经把那一层换掉了。

顺带,折叠还是 per-channel 这一维能否成立的前提:同一实测显示折叠后逐通道尺度离散度为
1.58–4.09 个八度,而无 BN 的 MNIST 模型只有 0–0.27 个八度(§7.3)。**BN 的 $k[c]$ 就是
通道尺度差的来源**,这证实了先前"per-channel 无收益是网络无 BN 所致"的猜测。

### 7.6 LUT 阈值的放置决定舍入语义,其代价随位宽放大

累加器出来之后理论上要做的是

$$q_{\text{out}} = \text{clamp}\Big(\text{round}\big(\text{acc}\cdot r\big) + Z_{\text{out}}\Big),\qquad r = \frac{s_w s_{\text{in}}}{s_{\text{out}}}$$

三件事叠在一起:乘 $r$ 是**重缩放**(累加器单位 → 输出码单位),`clamp(·,0,·)` 是**激活**,
`round` 加截断是**再量化**。芯片算不了 $r$ 这个浮点系数,通用做法是把数据搬回主机做
"反量化 → 激活 → 再量化"三步;而这里三步被折进同一张阈值表,运行时只剩
`acc >= threshold` 的整数比较(SAR 逐次逼近,$b^o$ 位输出比较 $b^o$ 次)。

**这一点两条实现路径一致,不一致的是怎么填那张表。**

**均匀网格 + 桶内取值(`quantize` 分支)。** 只算定义域边界
$\text{max\_val} = (s_{\text{out}}/s_{\text{accum}})\times 255$,交给 PAIBox 原有的
`LutReLU` 用 `linspace` 均匀铺 256 个阈值。于是桶 $i$ 覆盖
$[\,i/r,\ (i+1)/r\,)$ 而整桶输出 $i$ —— 这是**截断**,而真值在桶的上半段应当给 $i+1$。
偏差是**半个输出 LSB,且是系统性的**(恒向下)。

**阈值置于判决边界(本工作 `fit_affine_lut`)。** 把阈值放在 `round` 恰好跨码的位置,
即 $\lceil (k-0.5)/r \rceil$,于是 $q_{\text{out}} = \text{round}(\text{acc}\cdot r)$ 逐点精确,
无系统偏差。代价是必须知道硬件的舍入方向:硬件判 `acc >= threshold` 等价于
half-**up**,而 `torch.round` 是 half-even,两者在正中间的点上分歧,所以主机侧参考
实现要按 half-up 对齐(§1 [激活])。

**代价随位宽放大。** 半个 LSB 占满量程的比例是 $2^{-(b^o+1)}$:

| $b^o$ | 半 LSB 占满量程 |
| --- | --- |
| 8 | 0.20% |
| 4 | 3.1% |
| 2 | **12.5%** |

8-bit 下可忽略,2-bit 下不可忽略。而核预算恰好把可行区间压到低位宽
(`simplecnn` 在 w8a8 需 627 核 / 63 核预算),**所以这半个 LSB 正好落在论文要用的那一段**。

这条与非均匀格点(§1 [激活]:阈值间距完全自由,对数/分位点格点与均匀格点硬件代价相同)
是同一件事的两个方向:**阈值位置是一个免费的自由度,通用工具链因为 GPU 只会乘加移位
而被迫放弃它**,而这里放弃它要付出可量化的代价。第六章应把"均匀网格 + 截断"作为
一个消融项,与"判决边界 + round"在 $b^o \in \{8,4,2\}$ 上对比。

### 7.7 代价报告的最小要求

任何配置(无论来自搜索还是用户指定)编译后都应输出:

- 总核数 $\sum_l \text{occ}(\hat n_l)$ 与预算占比
- 逐层 $n_l$、$\tau_l$、$l_l$;以及输出组的 $l_{\text{out}}$ 与由此定出的读出模式
- 每个决策相对"该层最省核选择"的核数增量,即该选择的资源溢价
- 触发的约束类别(若有近似或替代)

最后两项是让用户理解"我的选择花了多少钱"的关键,也是搜索层敏感度分析(§9)的数据来源。

### 7.8 流水线延迟在前端不可见:错开的活动窗口在瞬时传播的仿真器里等于丢帧

前面几节讨论的都是空间维度的代价。SNN 把时间也变成了语义的一部分,而时间维度上
前后端目前对不上账。

**两种时序模型。** 硬件侧,一个值物理上每跨一个核花一拍,深度 $k$ 的核在第 $k$ 拍
才等到第 0 帧;编译器据此把 `tick_start` 错开成 $1,2,\dots$,让每个核在它的数据到达时
恰好开窗。仿真侧,`PAIIRGraph.step` 每拍新建 `node_outputs`,按拓扑序在**同一拍内**
把整张图算完,前驱的值取自当拍——**没有逐跳延迟寄存器**。

两者叠加的后果不是"延迟",而是**丢帧**:深度 $k$ 的核在第 $k$ 拍收到的是第 $k$ 帧,
第 $0$ 到 $k-1$ 帧不是迟到而是从未被处理。丢失量随深度线性累积,

$$\text{有效帧数}_l = T - \text{tick\_start}_l$$

而 SNN 的判决是 $T$ 拍上的脉冲计数,证据量按同一比例衰减。

**实测(DVSGesture,7 个 spiking 层,27 核,流水线深度 14)。** 输出核
`tick_start = 12`,20 帧里只积分 8 帧:

| 活动窗口 | 芯片仿真精度 | 对软件整数模型 |
| --- | --- | --- |
| 编译器错开($1,2,\dots,13$) | 0.5625 | −0.288 |
| 全部对齐到第 0 拍 | **0.8333** | −0.017 |

软件整数参考 0.8507,折叠后 float 0.8438,全测试集 288 样本。对齐后残差 1.7 个点,
量级与 §7.6 的舍入平局方向一致(硬件 `acc >= thres` 是 half-up,`torch.round` 是
half-even;第一层第 0 拍即差 3 脉冲 / 1331)。

**这不是量化问题,定位链如下。** 把参考实现自己的 int8 检查点(全局 $\times 127$)
接进同一条部署路径,掉点幅度相同(软件 0.8264 → 芯片 0.5781),说明与 scale 无关。
逐元素比对进一步排除了算式:第 1 层累加器逐元素相同(过阈值数 1334);第 2 层
(首个吃池化输出的层)芯片累加器恰好是软件的 4 倍、阈值 $328 = 4\times 82$、
过阈值数两边都是 7448,偏差为 0——即 AvgPool 的延迟除法补偿是精确的。
神经元语义也对得上:SNN 默认负阈值取 FLOOR(只作地板、不发负脉冲),与
SpikingJelly `IFNode` 一致。唯一对不上的是哪一拍收到哪一帧。

**一个次要因素。** `tick_duration` 编译时留成 0,含义是"永不关闭"。错开窗口时
运行长度必须是 $T + \max_l \text{tick\_start}_l$ 才能让信号走到输出,于是尾部有
一段没有输入的拍;而**卷积偏置每拍都落在膜电平上,与输入是否到达无关**,各核就
靠偏置自己充电、凭空发放(第 1 层总脉冲 683009 对软件 319968,多出的 36 万全在第
20 拍之后)。把窗口按 $T$ 关上可回收约 9 个点。窗口对齐后运行长度就等于 $T$,
这段尾巴不存在,该因素自动消失。偏置清零的消融可反证这条通路:此时软件 0.1667 与
芯片 0.1719 完全吻合(网络本身失效,但等价性恢复)。

**结论与论文位置。** 这是第四章"能力面补齐"的一个典型样本:机制在硬件与 IR 里都
存在(`tick_start` / `tick_duration` 是真实的核级配置字段),后端也正确地填了值,
但**仿真器的数据流模型与这些字段的硬件含义不一致**,于是前端拿不到可信的精度数。
两条修法互斥:给仿真器加逐跳延迟(与硬件一致),或在仿真路径上不错开 `tick_start`
(保持瞬时传播语义)。在此之前,`experiments/snn_quantize.py` 用 `--align-ticks`
取后者,读出偏移由实际活动窗口反推而非另算,以免二者再次漂移。

诊断脚本:`snn_layer_diff.py`(逐层逐拍脉冲)、`snn_param_diff.py`(神经元参数与
活动窗口)、`snn_tick0_diff.py`(逐元素累加器)。

---

## 8. 四条关键传导链(论文第三章的核心图)

注意核数**不只**是形状与位宽的函数:后端按稀疏度在稠密与 CSC 权重存储间选择,
而有多少权重码落在零上取决于 scale、scale 取决于标定。同一架构 w8a8 用 4 批标定
样本测得 627 核、用 2 批测得 643 核。所以解析估算与随机权重探针都带几个百分点的松弛。

**链 1(主导):权重位宽 → 权重 SRAM → 核数**

$$b^w_l \uparrow \ \Rightarrow\ \text{lines} \uparrow \ \Rightarrow\ S^{\text{wgt}}_l \uparrow \ \Rightarrow\ n_l \uparrow$$

实测:4 层小 CNN 上权重 8-bit → 1-bit 使核数从 24 降到 13(1.85×),
且 24 核中有 14 核贴着 4096 行 SRAM 的 90% 以上。

**链 2(次要):激活位宽 → 轴突地址 → LCN → 扇出 → 核数,并挤压时间步**

$$b^o_{l-1} \uparrow \ \Rightarrow\ A^{\text{addr}}_l \uparrow \ \Rightarrow\ l_l \uparrow \ \Rightarrow\ \begin{cases} n_l \uparrow \\ \text{读出模式 } \text{STREAM} \to \text{STEP}\ (\text{仅 } l_{\text{out}}) \end{cases}$$

实测:激活 8-bit → 1-bit 核数仅 24 → 20(1.2×),但 LCN 从 L6 降到 L3。
**在权重受限的层上,这条链几乎不产生核数收益。**

**链 4(最强):计算模式 → 输出位宽 → 轴突需求 → LCN → 核数**

$$m_l = \text{SNN} \ \Rightarrow\ b^o_l = 1 \ \Rightarrow\ A_{l+1}\cdot b^o_l \ \text{降 8 倍} \ \Rightarrow\ l_{l+1} \downarrow\ \Rightarrow\ n_{l+1} \downarrow$$

一个 8-bit ANN 输出每个输入占 8 个轴突地址,1-bit 脉冲只占 1 个。轴突需求是
$A_l \cdot b^o_{l-1}$,LCN 跟着轴突需求走,所以**脉冲化的生产者最多把消费者的 LCN 砍 8 倍**。

实测(`experiments/snn_probe.py`,`simplecnn` w8a8,前 $k$ 层脉冲化,仅测核数):

| 脉冲层数 | 核数 | 预算 | 增量 |
| --- | --- | --- | --- |
| 0(全 ANN) | 643 | 1021% | — |
| 1 | 546 | 867% | −97 |
| 2 | 527 | 837% | −19 |
| 3 | 385 | 611% | **−142** |
| 4 | 353 | 560% | −32 |
| 5 | 105 | 167% | **−248** |
| 6 | 104 | 165% | −1 |
| 7 | 104 | 165% | 0 |

**6.2 倍,远强于链 1(权重位宽,约 1.6 倍)与链 2(激活位宽)。** LCN 集合从
$\{2,6,7\}$ 变成 $\{0,3,5,6,7\}$,即部分组脱离了 128× 上限。
但**单靠模式不够**:104 核仍超 63 核预算 65%,必须与位宽联合。

**收益极度不均匀,且可由绑定约束预测。** `block3.0` 一层贡献 248 核,
`block2.0` 贡献 142 核,而 `block3.2`、`head.1` 各贡献 1 核和 0 核:

- `block3.0` 的消费者 `block3.2` 扇入 $256\times9=2304$,8-bit 需 18432 个轴突地址
  (LCN 16×),1-bit 仅需 2304(LCN 2×)—— **LCN 受限,收益巨大**
- `head.1` 是 $\text{Linear}(4096{\to}256)$,100 万权重 —— **权重受限**,砍输入位宽无用

**这是"双敏感度不相关"论点(§9)的核数侧实测**:核数敏感度在层间差两个数量级,
而且可以从"该层是 LCN 受限还是权重受限"事先判断。这同时说明代价报告必须报
**绑定约束**而不是一句"超预算"(§7.7)。

注意该探针只测核数:阈值取自各层标定出的输出范围(阈值平衡的占位),
且 `build_spec` 把脉冲层的 `output_scale` 置为 1.0(正确值应为 $\theta/T$),
所以**这些图的数值是无意义的**。布放不读 LUT 阈值,故核数仍然是真实转换会得到的数。

**链 3:量化粒度 → 神经元类型 → 神经元 SRAM → 核数**

$$g_l = \text{per-channel} \ \Rightarrow\ \text{逐神经元 } \texttt{leak\_tau} \ \Rightarrow\ \begin{cases}
\tau_l \text{ 不变} & \text{折叠轴} \perp \text{通道轴} \\
\tau_l = \text{full} \Rightarrow S^{\text{neu}}_l \times 8 & \text{折叠组跨通道}
\end{cases}$$

实测:conv 主导网络落在第一种情形,折叠次数不变(260 → 260),神经元 SRAM +0.9%,核数 0 变化。
**这条链被证伪为"恒定 8 倍代价";真实形态是条件性的,且在常见卷积网络上几乎不触发。**
仍待构造折叠组跨通道的用例来实测第二种情形。

---

## 9. 逐层敏感度定义

论文的核心论点是**精度敏感度与核数敏感度不相关**。形式化定义:

对层 $l$ 与配置扰动 $\delta$(如 $b^w_l: 8 \to 4$):

$$\text{Sens}^{\text{acc}}_l(\delta) = \text{Acc}(\mathbf{c}) - \text{Acc}(\mathbf{c} \oplus \delta_l)$$

$$\text{Sens}^{\text{core}}_l(\delta) = \sum_k \text{occ}(\hat{n}_k(\mathbf{c})) - \sum_k \text{occ}(\hat{n}_k(\mathbf{c} \oplus \delta_l))$$

注意 $\text{Sens}^{\text{core}}_l$ 是**全网求和**而非仅层 $l$,因为 $b^o_l$ 会经 C6 与链 2 影响后继层。

初步实测($\delta$ = 激活 $8\to1$,4 层 CNN):$\text{Sens}^{\text{core}}$ 分别为 3、1、0、0 核。
**后两层牺牲精度换不到任何核收益**,这是逐层搜索的动机。

---

## 10. 待实测校准清单

| 项 | 为什么需要实测 |
| --- | --- |
| 膨胀系数 $\eta_l$ | §4.4 的 $n_l$ 是理想装箱下界,贪心 first-fit 实际更差 |
| ~~$\tau_l$ 的判定规则~~ | **已解决**:见 §4.2。折叠按组决定,条件是组内 Part 2 属性全等(`routing.py:652`);`leak_tau` 参与该比较,故 per-channel 的代价取决于折叠轴与通道轴是否正交,实测卷积网络上仅 +0.9% 而非 8 倍 |
| $\sigma_l$ 的实际取值 | 权重共享在 PAIBox `get_weight.py` 里的实际粒度 |
| ~~ANN↔SNN 混合可编译性~~ | **已解决**:`AAAA`/`SSSS`/`AASS`/`SSAA`/`ASAS`/`SASA` 六种排布全部通过前端与后端(`experiments/mixed_mode_probe.py`)。模式校验是逐节点的(`passes.py:1627`),无全图约束。注意编译通过不等于数值正确:边界无速率解码 |
| ~~$e_l = \text{add\_potential}$ 的代价~~ | **已解决**:见 §7.4。机制完整(偏积分核 + `PotentialAddOp` + `DIRECT_ADD` 实测成立),但由图结构驱动;前端缺"分核"的表达手段,`fuse_to_offline_cores` 会吸收单消费者路径 |
| ~~$T^{\max}$ 是否被检查~~ | **已解决**:见 §4.1.1。公式 $2^{8-l}$ 正确但只作用于输出组,且只决定 `STREAM`/`STEP` 读出模式,非可行性约束 |

---

## 11. 与 PAIBox 现状的差距

现状:`propagate_data_format` 按"正确且值域可表示"选最窄位宽,**代价函数为空**。
`route_solver.py` 的 CP-SAT 目标只有软性曼哈顿距离,核数从不进目标函数。
核数在 `try_store_neuron` 的贪心装箱中被动产生。

本模型要补的是:把 §6 的目标函数和 §5 的约束显式化,让前端决策可以查询 $\hat{n}$。

时间维度上还有一处更硬的差距(§7.8):`tick_start` 由后端按流水线深度正确错开,
但 `PAIIRGraph.step` 在同一拍内穿透整张图,两种时序模型不一致,SNN 的仿真精度
因此按深度成比例丢帧(DVSGesture 上 0.83 → 0.56)。这不是代价模型的问题,是
能力面的正确性问题,需要先定下哪一侧是基准语义。
