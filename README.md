# MoleOOD：baseline 与动态 Prompt 持续学习最终交付

本目录只保留基础复现、最终方法的必要代码、数据、初始检查点和最终结果。
中间实验、旧日志、TTA、Notebook 和开发说明已移至旁边的 `../MoleOOD_历史实验归档/`。
**当前结论：功能原型已实现，但这轮实验没有证明动态 Prompt 比不更新模型或固定池更好。**

## 1. 目录与结果位置

```text
MoleOOD/
├── README.md                     本说明（唯一的项目总 README）
├── main.py                       无 TTA 的 MoleOOD baseline
├── run_dynamic.py                最终方法和对照的一键入口
├── evaluate_dynamic.py           保存模型的纯评价入口
├── requirement.txt               运行依赖
├── models/                       baseline 所需分子图、子结构和损失模块
├── continual/
│   ├── core.py                   数据/模型/指标公共工具
│   ├── protocol.py               数据隔离及初始模型来源校验
│   └── dynamic.py                动态 Prompt 检索、分配、复用和容量管理
├── configs/dynamic_prompt.json   最终固定实验配置
├── data/
│   ├── ec50/lbap_core_ec50_scaffold_brics.json
│   └── continual_seed42/          source、划分记录及最终数据流输入
├── log/
│   ├── lbap_core_ec50_scaffold_brics/baseline_no_tta/  baseline 成绩与模型
│   ├── cl_source_head_prepared_seed42/               最终方法必需的初始模型
│   └── dynamic_delivery/                            最后 12 组结果、模型和曲线
└── tests/test_dynamic.py          最终方法功能测试
```

`.git`、`.gitignore` 和 IDE 设置是项目元数据，保留。运行 Python 时可能重新生成
`__pycache__`，它不属于实验结果。

`log/cl_source_head_prepared_seed42/` 虽然不是最后一轮的结果目录，但其中的
`best_model.pth` 和 `args.json` 是最后一轮加载、复核和恢复所必需的，不能移走。
历史实验文件里的原路径按当时状态保存，不要继续照其中的旧命令运行。

## 2. 题目和本次方法在做什么

“动态持续学习”指分子数据陆续到来，数据组成会变化；模型应适应新数据，并尽量
保持处理历史数据的能力。这里的“环境”指数据分布或特征群组，不是 Conda 环境。

本实现采用**有监督**设定：学习时知道分子的性质标签 `cls_label`，不知道环境编号
或任务 ID。环境编号仅供离线组织数据和评价，不输入 Prompt 检索或分配模块。
本实现不包含原方法图中“无标签、遮挡原子恢复”的分支，不属于 TTA 或原版 L2P 的严格复现。

## 3. baseline 的原理

`main.py` 使用本地 PyG 改版的 MoleOOD，流程是：

1. 先训练环境推断模型和条件预测模型，辅助学习不变性目标。
2. 训练完整分子图编码器、BRICS 子结构编码器和性质预测头。
3. 用 OOD 验证集 AUC 选择检查点，并记录对应 OOD 测试集 AUC。

当前实现不是注意力融合：完整图经 4 层 GIN 得到 128 维表示，子结构分别经另一个
4 层 GIN 编码后求均值，再与完整图表示**相加**，送入二分类预测头：

```text
z_graph = 完整图编码器(G)
z_sub   = mean(子结构编码器(s)，s 属于 BRICS 子结构集合)
logits  = 性质预测头(z_graph + z_sub)
```

baseline 的训练目标包括按推断环境聚合的分类损失和偏离条件预测损失的约束。
环境推断器实际读取图与性质标签 q(e|G,y)，不应把它当作不依赖性质标签的在线路由器。
最终动态实验的数据分组使用的是 source 表示聚类，并未使用该推断器在线分配 Prompt。

全量 baseline 用 seed=2022、batch=128、环境辅助训练 20 轮、主模型 50 轮。
最佳验证轮次编号 39（第 40 轮），测试 ROC-AUC 为 **68.03%**。
论文 Table 2 的 EC50-Scaffold 是 66.69 ± 0.34%，原截图为 66.91%。本次是一次
本地改版运行，不能据此声称严格复现其多种子统计结果或证明优于论文。

## 4. 数据究竟怎么划分

### 4.1 原始数据

原文件是已包含 BRICS 子结构的 DrugOOD EC50-Scaffold JSON：

| 原始 split | 数量 | 用途 |
|---|---:|---|
| train | 2,570 | baseline 训练；持续学习的 source/适应/环境评价都从这里再划分 |
| ood_val | 2,532 | baseline 和初始 MoleOOD 的检查点选择 |
| ood_test | 2,533 | 原 OOD 测试记录；不用于参数更新或挑训练轮次 |
| iid_val | 580 | 初始预测头的独立开发验证，负类 17、正类 563 |
| iid_test | 4,446 | 可用于方法固定后的额外纯评价，不在最后 12 组默认流程里运行 |

### 4.2 持续学习划分

只重新划分原来的 2,570 条 train，数据划分 seed=42：

```text
2,570 条原 train
├── source：1,028 条（负类 68、正类 960）
│   └── 只用这些样本训练初始模型，不能提前用后续适应样本训练
└── 剩余：1,542 条
    ├── 在线适应：1,155 条（允许使用性质标签）
    └── 固定环境评价：387 条（不参与反向传播）
```

相同 canonical isomeric SMILES 的记录归为一组，不跨 source、适应、环境评价三个部分。
该分组保留立体化学，不等于完整的去盐、互变异构体标准化。

用初始 MoleOOD 的 source 表示拟合 KMeans，然后给后续数据分组。最初有 3 个聚类，
一个环境的评价负样本只有 1 个；最终将它按 source 聚类中心距离并入最近环境。
合并保留了原训练/评价成员，**没有把已经评价过的样本转入训练**。
最终仍保留原编号 0、2，并不是环境编号漏了一个：

| 最终环境 | 适应负类 | 适应正类 | 评价负类 | 评价正类 |
|---|---:|---:|---:|---:|
| 0（原 0 + 1） | 32 | 841 | 11 | 281 |
| 2（原 2） | 45 | 237 | 15 | 80 |
| 合计 | 77 | 1,078 | 26 | 361 |

### 4.3 初始预测头准备

持续学习的初始模型与全量 baseline 不同：先在 source 上训练 MoleOOD，再固定编码器，
仅用 source 的加权交叉熵训练预测头，用 iid_val 的平衡准确率优先、AUC 次之选择检查点。
选中第 23 轮，iid_val AUC 73.33% → 82.27%，平衡准确率 50.00% → 70.42%。
这是开发验证成绩，不是最后的环境评价或持续学习收益。

所保留的初始模型在 387 条固定环境评价数据上的初始宏平均 AUC 为 **71.00%**。
数据划分和适应种子使用 42，与全量复现的 2022 属于不同实验设置，不能混作同条件对照。

### 4.4 三种数据流

三种流使用相同适应样本和相同环境评价集，只改变适应顺序：

- **突变 abrupt**：环境 A → B。
- **渐变 gradual**：相邻环境样本在时间上混合，分为 8 个观察窗口。
- **重复 repeat**：A → B → A → B；每个环境的适应分子分成两部分，回来时遇到新的
  分子，而不是回放同一批训练样本。

每条适应记录只学习一次，每个 batch 只更新一次。窗口/环境信息用于组织和评价，
模型不接收阶段编号，不在边界重置模型或优化器。

## 5. Prompt 是怎么加入的

**采用“余弦相似度检索 + 原子嵌入加法 Prompt”。**

### 5.1 query、key 和 Prompt 各是什么

- **query q(G)**：不带 Prompt 的冻结编码器产生的分子表示，完整图表示与子结构均值相加，
  再做 L2 归一化；维度为 128。不使用性质标签或环境编号。
- **key k_j**：每个 Prompt 的检索原型。最初通过 source query 的 KMeans 中心初始化；
  新 Prompt 的 key 来自持续新颖 query 的均值方向。key 建好后固定，不通过梯度持续移动。
- **Prompt p_j**：真正可学习的 128 维向量。按相似度选择一个 key 后，取出对应 p_j。
  它可以看作该 key 对应的“内容”，但不是经过 W_V 投影的注意力 value。

检索是硬 Top-1：

```text
q = normalize(z_graph + z_sub)
j = argmax_j cosine(q, k_j)       # 只比较已激活的槽位
p = PromptMemory[j]              # 直接取出对应的可学习向量
```

### 5.2 注入位置

每个原子的类型先通过原子嵌入网络，变成 128 维向量 h_v；**在第一层 GIN 消息传递
之前**加上该分子选中的同一个 p_j：

```text
新的原子表示：h'_v = h_v + p_j
完整图分支：  z_graph_prompted = GIN(所有 h'_v，原图的化学键)
最终预测：    logits = 固定性质预测头(z_graph_prompted + z_sub)
```

同一分子的所有原子共享所选 Prompt，不改变原子数和化学键，不新增虚拟节点。
子结构分支不注入 Prompt，继续提供冻结的子结构均值。编码器参数虽然固定，
但带 Prompt 的前向分支保留输入梯度，梯度可以穿过 GIN 回到 p_j。

```text
干净分子 + BRICS 子结构 → 冻结编码器 → query → 与 keys 比较 → 选择 Prompt
                                                                 ↓
原子嵌入 ─────────────────────────────────────────────────────── 加法
                                                                 ↓
                                               冻结 GIN → 图表示
                                                                 + 冻结子结构均值
                                                                 ↓
                                                      固定预测头 → 活性预测
```

## 6. 动态新增、复用与容量管理

动态方法从 2 个活跃 Prompt 开始，最多 6 个。参数槽位提前预分配，逻辑上按需激活，
避免在线扩张优化器；实际预留参数和缓冲区有硬上限。

1. 当前 batch 到来先预测，再学习。
2. 与已有 key 足够接近：复用对应 Prompt。
3. 最近 key 的余弦距离超过 source 距离分布第 95 百分位门限，视为新颖样本。
4. 连续 2 个 batch 至少 30% 样本新颖，且新颖 query 的平均一致性至少 0.8，才尝试分配。
   新颖缓冲区最多 128 个向量，分配之间至少隔 5 个 batch。
5. 有空槽：复制最近 Prompt 作为起点，创建固定的新 key，并记录创建时的 Prompt 锚点。
6. 容量满：LRU 替换最近最少使用的非 source Prompt，清除该槽位旧 Adam 状态。
   初始 source 槽位不被替换，近 3 个 batch 使用过的槽位暂不替换。
7. 无可替换槽位：记录 capacity_deferred，复用已有 Prompt，不突破容量。

容量管理采用“替换”这一种明确策略，本版没有合并或剪枝。保护 source 槽位免于替换
不等于其预测能力绝对不变；这些 Prompt 仍可更新。容量、遗忘和复用之间的取舍需要实验评价。

fixed/dynamic 方法均冻结完整图编码器、子结构编码器、BN 统计量和性质预测头，
只更新本 batch 选中的 Prompt。每个槽位拥有独立的 Adam 状态，未选中的槽位不被动量误更新。

监督损失为：

```text
L = 类别加权交叉熵
  + 0.1 × KL(冻结 source 模型在当前输入上的预测 || 带 Prompt 的预测)
  + 1.0 × mean((选中 Prompt − 创建时锚点)^2)
```

source 类别权重为 N/(2*n_c)：负类约 7.559，正类约 0.535；不读取未来或评价集的
类别频率。后两项用于限制更新干扰，并不预先保证抗遗忘有效。

## 7. 怎么运行

当前环境为 WSL `Ubuntu-20.04` 的 Conda `piptorch`（Python 3.10、PyTorch 2.6.0+cu124，
RTX 4060 Laptop GPU）。在该环境与本项目目录执行。所有命令写成一行。

### 7.1 全量 baseline（已有结果，不必重复运行）

```bash
python -u main.py --device 0 --seed 2022 --batch_size 128 --epoch_ast 20 --epoch_main 50
```

输出会自动创建新目录，避免覆盖已有结果。当前原始 baseline 的终端输出在
`log/lbap_core_ec50_scaffold_brics/baseline_no_tta/seed_2022_20261113_014218_388420/console.log`。

### 7.2 最终方法：一键固定 12 组

三种流 × 四种方法（不更新、全参数顺序微调、固定 6 个 Prompt、动态 2→6 个 Prompt）。
**这批已经完成，不需要现在重跑。** 当前结果在 `log/dynamic_delivery/`。
如果只是重新查看结果，直接打开该目录的 `SUMMARY.md`。

若有明确需要完整复现，用新的输出目录运行：

```bash
python -u run_dynamic.py --config configs/dynamic_prompt.json --output log/dynamic_reproduction 2>&1 | tee log/dynamic_reproduction_console.log
```

程序不搜索超参数、不按评价成绩挑轮次、不自动增加实验。source 特征只用于初始化和
校准；后续模型每次只接收当前分子与标签，评价不会触发学习、分配或替换。

只读检查当前数据和初始模型，不训练：

```bash
python run_dynamic.py --config configs/dynamic_prompt.json --check
```

恢复一次确实中断的原默认目录运行（完成的项目会跳过）：

```bash
python -u run_dynamic.py --config configs/dynamic_prompt.json --resume 2>&1 | tee -a log/dynamic_delivery/console.log
```

仅从已保存的最终模型重新评价：

```bash
python evaluate_dynamic.py --run log/dynamic_delivery/abrupt/seed_42/dynamic --device cuda:0
```

功能测试只使用 CPU 合成数据，不执行真实数据的正式训练：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_dynamic.py -v
```

依赖见 `requirement.txt`。当前环境已配置好，不需要为整理目录重新安装依赖。
matplotlib 用于自动出图，缺少时仍保存 CSV/JSON。代码不会自动安装包。

## 8. 最后 12 组结果与明确结论

最终宏平均 ROC-AUC（%），seed=42：

| 数据流 | 不更新 | 全参数顺序微调 | 固定 Prompt 池 | 动态 Prompt |
|---|---:|---:|---:|---:|
| 突变 | **71.00** | 52.32 | 70.77 | 70.29 |
| 渐变 | **71.00** | 56.44 | 70.82 | 70.44 |
| 重复 | **71.00** | 60.49 | 70.96 | 70.38 |

- 本轮没有观察到动态方法的收益：它在三个流上均低于不更新和固定池。
- Prompt 方法比本轮全参数微调更稳定，但各方法的更新范围与约束也不同，
  不能把差异单独归功于动态分配；不更新仍是最强对照。
- 动态模型始终只有 2 个活跃 Prompt，新增和替换均为 0。
  55/1,155 个适应样本被判新颖（约 4.76%），没有 batch 达到 30% 触发条件，
  因而新增和满容量管理未在真实实验中触发。相关功能只在合成测试中验证。
- 动态激活 2 个、固定池激活 6 个，不代表物理 Prompt 参数节省三倍：两者都预留 6 个槽位。

**可支持的报告结论：实现了动态 Prompt 持续学习原型，但当前 EC50 模拟流没有提供
动态分配改善预测或知识保持的实验证据。不能写成“方法已经证明有效”。**

这轮只有一个适应种子、两个聚类环境，评价负样本分别只有 11/15 个，且评价集已多次
用于开发。source 从整个原 train 中随机分出，可能已覆盖后续相似分布，因此这里是
分布组成/顺序变化的实例流，不能声称后续都是首次出现的新环境。
IC50 可用同格式输入扩展，但本次没有完成其数据和实验。此处到此收尾，不自动追加调参。

## 9. 结果文件怎么读

`log/dynamic_delivery/` 中：

- `COMPLETE.json`：确认固定 12 组全部完成。
- `SUMMARY.md`、`summary.csv`、`summary.json`：总体对照。
- `*_auc.png`、`*_memory.png`：三种流的 AUC 与活跃 Prompt 数曲线。
- `streams/`：三种实际使用的数据顺序和数据来源指纹。
- 各 `场景/seed_42/方法/` 下的 `metrics.json`：各环境 AUC、平衡准确率、两类召回率、混淆矩阵。
- `switching.json`：环境变化/回归前后实际评价时点的下降与恢复，不是瞬时切换测量。
- `events.json`：新增、替换、拒绝与容量不足事件；空列表也是合法结果。
- `result.json`：最终指标、参数量、时间和存储；注意区分激活量、预留量、优化器与整个文件大小。
- `latest.pt`：模型、优化器、检测缓冲区、Prompt 世代和进度；支持从完成的数据段末恢复。
- `prompt_memory.pt`：不含基础模型的 Prompt 记忆。
- `console.log`：本批终端输出。

遗忘指标是在线已接触环境的历史最高评价 AUC 减当前 AUC，再平均；不把未来环境尚未
适应时的下降称为旧知识遗忘。`prequential_auc` 是每批适应样本在更新前的预测成绩，
不是独立评价成绩。时间平均指标对已记录评价点等权，不是连续时间积分。
日志和完整检查点随时间增长，不属于有限 Prompt 记忆的容量承诺。

## 10. 归档和可复核性

归档目录保留原相对结构、旧 README、旧代码和旧结果；`文件校验清单.json` 包含
移动文件的 SHA256。整理只迁移文件和公共校验函数，没有改动 baseline、最终算法、
模型权重、最终配置、数据划分或现有实验结果。

因最终记录含绝对 WSL 路径，保留了运行依赖的现有路径。移动整个项目到别的机器时
需要另行处理路径和指纹，不能把旧的绝对路径直接当作跨机器配置。
