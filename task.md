# UNIGEN+ : Modernizing Universal Domain Generalization for Sentiment Classification

## COMP0087 Group Project — 完整执行方案

---

## 一、项目定位与核心研究问题

### 统一研究问题（Research Question）

> **"To what extent can modern LLMs and improved prompt strategies enhance the domain generalizability of the UNIGEN framework, and does this generalizability extend beyond binary sentiment classification?"**

这个研究问题自然地将三个方向组合成一个连贯的叙事：

| 实验模块 | 对应方向 | 核心假设 |
|---------|---------|---------|
| **模块1：复现基线** | 复现原文 | 验证UNIGEN原始结果的可复现性 |
| **模块2：现代LLM生成器** | 方向A | 更强的LLM（LLaMA-3.2, Mistral等）配合优化的超参数，能显著提升生成数据质量和TAM性能 |
| **模块3：Prompt策略** | 方向C | 不同prompt设计（instruction-style、diversity-inducing等）对domain-invariant数据生成有显著影响 |
| **模块4：多分类扩展** | 方向B | UNIGEN框架可以从二分类推广到多分类任务，且保持domain generalizability |

### 为什么这个组合好？

1. **逻辑递进**：先复现 → 再改进生成器 → 再改进prompt → 再扩展任务，层层深入
2. **对比清晰**：每个模块都有明确的baseline和变量控制
3. **保底策略**：即使模块3/4不成功，模块1+2也足够写一篇好report
4. **分工自然**：5个人可以并行推进不同模块

---

## 二、具体实验设计

### 模块1：复现UNIGEN基线（必做，Week 1-3）

**目标**：复现论文Table 2中RoBERTa-based TAM的结果

**步骤**：
1. Clone代码仓库 https://github.com/c-juhwan/unigen
2. 用GPT2-XL生成1,000k数据（使用论文的universal prompt）
3. 执行pseudo-relabeling和SUNGEN去噪
4. 训练DistilBERT和RoBERTa TAM
5. 在7个测试集上评估

**关键数据集**（均为公开数据集，不涉及社交媒体原始数据）：
- SST-2, IMDB, Rotten Tomatoes（电影领域）
- Amazon Reviews（产品领域）
- Yelp Reviews（餐厅领域）
- CR（电子产品领域）
- Tweet Sentiment（需确认是否符合UCL数据限制——如不符合可用替代数据集）

**⚠️ 数据集注意事项**：
- coursework明确禁止使用social media posts数据
- Tweet数据集（Rosenthal et al., 2017）来自Twitter，可能违反规定
- **建议**：用Financial PhraseBank或其他非社交媒体数据集替代Tweet
- 提前跟TA确认数据集合规性

**预期结果**：接近论文报告的数值（允许±2%偏差）

---

### 模块2：现代LLM作为数据生成器（核心创新，Week 3-6）

**目标**：系统性地比较不同LLM生成器对UNIGEN性能的影响

**实验变量**：

| 生成器模型 | 参数量 | 类型 | 来源 |
|-----------|-------|------|------|
| GPT2-XL（原文） | 1.5B | Autoregressive | OpenAI |
| LLaMA-3.2-1B | 1B | Autoregressive + Instruction-tuned | Meta |
| LLaMA-3.2-3B | 3B | Autoregressive + Instruction-tuned | Meta |
| Mistral-7B-v0.3 | 7B | Autoregressive | Mistral AI |
| Qwen2.5-1.5B | 1.5B | Autoregressive | Alibaba |
| Phi-3.5-mini | 3.8B | Autoregressive | Microsoft |

**关键改进（相对于原文Section 4.5.4的不足）**：
- 原文用完全相同的超参数测试不同PLM，导致不公平比较
- 我们为每个模型单独调优 top-k、top-p、temperature
- 对instruction-tuned模型，测试instruction-style prompt（原文仅用completion-style）

**超参数搜索空间**：
- top-k: [20, 40, 60]
- top-p: [0.8, 0.9, 0.95]
- temperature: [0.7, 0.8, 1.0]
- τ_RE (relabeling temperature): [0.05, 0.1, 0.2]

**控制变量**：
- TAM架构固定为RoBERTa-base
- 生成数据量固定为1,000k
- 训练超参数与原文一致

**分析维度**：
1. 各域平均准确率对比
2. 生成数据的多样性分析（distinct-n, self-BLEU）
3. 生成数据的noise rate对比（relabeling前后标签一致性）
4. 不同模型生成数据的domain分布可视化

---

### 模块3：Prompt策略研究（Week 4-7）

**目标**：研究prompt设计对domain-invariant数据生成的影响

**Prompt变体设计**：

```
# P1: Original (论文原始prompt)
"The text in [positive/negative] sentiment is: "

# P2: Instruction-style (适配instruction-tuned模型)
"Generate a short text that expresses [positive/negative] sentiment about any topic: "

# P3: Diversity-inducing (鼓励多样性)
"Write a [positive/negative] opinion about something. Be creative and cover any subject: "

# P4: Contrastive prompt (强调对比)
"Here is an example of text with clearly [positive/negative] sentiment: "

# P5: Topic-diverse prompt (显式引导话题多样性)
"The [positive/negative] review or comment is: "

# P6: Chain-of-thought style
"Thinking about what makes text [positive/negative], here is an example: "
```

**实验设计**：
- 固定生成器为最优模型（由模块2确定）和GPT2-XL
- 每种prompt生成100k数据，比较数据质量
- 选出top 3 prompt分别生成1,000k数据，训练TAM并评估

**分析维度**：
1. 不同prompt生成数据的topic分布（LDA主题分析）
2. 数据多样性指标
3. noise rate
4. 下游TAM性能

---

### 模块4：多分类任务扩展（Week 5-8）

**目标**：验证UNIGEN能否扩展到超越二分类的任务

**任务选择**：

| 任务 | 数据集 | 类别数 | 领域多样性 |
|------|-------|-------|----------|
| 5-star情感分类 | Amazon Reviews (5-class), SST-5 | 5 | 产品/电影 |
| 主题分类 | AG News | 4 | 新闻 |
| 情感细粒度 | GoEmotions (选取子集) | 多类 | 通用 |

**Prompt设计（多分类版本）**：
```
# 5-star 情感
"The text with [1-star/2-star/3-star/4-star/5-star] rating sentiment is: "

# 主题分类
"The [World/Sports/Business/Technology] news article is: "
→ 这个本身就是domain-specific的，测试UNIGEN的"universal"是否仍然成立

# Universal多分类
"The text classified as [label] is: "
```

**核心研究问题**：
- 随着类别数增加，noise rate如何变化？
- Pseudo-relabeling在多分类中是否同样有效？
- Supervised contrastive learning在多类别下的效果如何？

---

## 三、时间线与分工（5人组）

### 角色分配建议

| 角色 | 主要职责 | 人数 |
|------|---------|------|
| **A: 复现与基础设施** | 搭建代码框架、复现原文结果、维护实验pipeline | 1人 |
| **B: LLM生成器实验** | 模块2的数据生成、超参数调优 | 1人 |
| **C: Prompt与数据分析** | 模块3的prompt实验、所有模块的数据质量分析 | 1人 |
| **D: 多分类扩展** | 模块4的数据集准备、实验设计与执行 | 1人 |
| **E: 训练与评估** | 所有TAM的训练、评估、ablation study | 1人 |

> 注意：角色不是完全隔离的，尤其在report撰写阶段所有人都需要参与。

### 周计划

```
Week 1 (1月26日周)：
  ├── 全组：阅读论文 + clone代码 + 环境配置
  ├── A：跑通原始代码，理解pipeline
  └── 全组：确定research question，提交group formation

Week 2 (2月2日周) — Progress Report 1：
  ├── A：开始复现GPT2-XL数据生成
  ├── B：调研可用LLM，下载模型权重
  ├── C：设计prompt变体
  ├── D：准备多分类数据集
  └── E：搭建统一的训练/评估脚本

Week 3 (2月9日周)：
  ├── A：完成数据生成 + relabeling
  ├── B：开始LLaMA/Mistral生成实验
  ├── E：训练GPT2-XL基线的TAMs
  └── 全组：review复现结果，调整计划

Week 4 (2月16日周) — Progress Report 2：
  ├── A：完成基线复现，报告结果偏差
  ├── B：完成至少3个LLM的数据生成
  ├── C：开始prompt实验（小规模100k）
  ├── D：开始多分类prompt设计和数据生成
  └── E：训练不同LLM生成数据的TAMs

Week 5 (2月23日周)：
  ├── B：完成所有LLM生成器实验
  ├── C：分析prompt实验结果，选top 3
  ├── D：生成多分类数据
  ├── E：继续训练 + 开始ablation study
  └── 全组：中期讨论，确定report框架

Week 6 (3月2日周) — Progress Report 3：
  ├── C：完成top 3 prompt的1,000k生成 + TAM训练
  ├── D：训练多分类TAMs
  ├── E：汇总所有实验结果
  └── 全组：开始写report（Introduction + Related Work）

Week 7 (3月9日周)：
  ├── 全组：补充实验 + error analysis
  ├── C/D：数据可视化（t-SNE、topic分布等）
  └── 全组：写report（Methods + Experiments）

Week 8 (3月16日周) — Progress Report 4：
  ├── 全组：写report（Results + Discussion）
  └── 查漏补缺，跑需要的额外实验

Week 9 (3月23日周)：
  ├── 全组：完成report初稿
  └── 内部review + 修改

Week 10 (3月30日周)：
  ├── 全组：report修改 + 润色
  └── 准备附录

Week 11 (4月6日周)：
  ├── 全组：最终修改
  └── LaTeX格式检查

Week 12 (4月13-17日)：
  ├── 最终检查
  └── 4月17日 16:00 提交
```

---

## 四、Report结构建议（8页正文）

### 页面分配

| 部分 | 建议页数 | 内容要点 |
|------|---------|---------|
| Abstract | 0.3页 | 研究问题 + 方法概述 + 核心发现 |
| Introduction | 1页 | 背景 → UNIGEN的局限 → 你们的研究问题 → 主要贡献 |
| Related Work | 1页 | Dataset generation, Domain generalization, Prompt engineering |
| Methods | 1.5页 | 4个模块的方法描述，重点在创新部分（模块2-4） |
| Experimental Setup | 1页 | 数据集、模型、超参数、评估指标 |
| Results & Discussion | 2.5页 | 主表 + ablation + 分析 + 可视化 |
| Conclusion | 0.7页 | 总结 + limitations + future work |

### 核心表格设计

**Table 1**（最重要）：不同LLM生成器 × 不同TAM的跨域平均性能
```
| Generator | TAM | SST-2 | IMDB | Rotten | Amazon | Yelp | CR | Avg |
|-----------|----------|-------|------|--------|--------|------|----|-----|
| GPT2-XL   | RoBERTa  | xx.x  | xx.x | ...    |        |      |    |     |
| LLaMA-3.2 | RoBERTa  | xx.x  | xx.x | ...    |        |      |    |     |
| Mistral-7B| RoBERTa  | xx.x  | xx.x | ...    |        |      |    |     |
| ...       | ...      | ...   | ...  | ...    |        |      |    |     |
```

**Table 2**：Prompt策略对比

**Table 3**：多分类任务结果

**Figure 1**：t-SNE可视化（参照原文Figure 2，但增加你们的改进结果）

**Figure 2**：生成数据多样性分析（topic分布图）

---

## 五、技术实现要点

### 代码架构建议

```
unigen-plus/
├── configs/               # 超参数配置文件
│   ├── generation/        # 各LLM的生成配置
│   └── training/          # TAM训练配置
├── data/
│   ├── generated/         # 生成的合成数据
│   └── benchmarks/        # 评估数据集
├── src/
│   ├── generation/        # 数据生成模块
│   │   ├── generator.py   # 统一的生成器接口
│   │   ├── relabeler.py   # Pseudo-relabeling
│   │   └── prompts.py     # Prompt模板管理
│   ├── training/          # TAM训练模块
│   │   ├── trainer.py     # 统一训练器（CE + SCL）
│   │   ├── memory_bank.py # Denoising memory bank
│   │   └── losses.py      # 损失函数
│   ├── evaluation/        # 评估模块
│   │   ├── evaluator.py   # 统一评估接口
│   │   └── analysis.py    # 数据质量分析工具
│   └── utils/
├── scripts/               # 实验运行脚本
├── notebooks/             # 分析和可视化
└── results/               # 实验结果
```

### 关键技术注意事项

1. **数据生成**：
   - GPT2-XL用HuggingFace的`AutoModelForCausalLM`
   - LLaMA/Mistral等较大模型考虑4-bit量化（`bitsandbytes`），3090/4090可以跑7B模型
   - 生成1,000k数据大约需要3-6小时（取决于模型大小和GPU）

2. **Instruction-tuned模型的适配**：
   - GPT2-XL是纯completion模型，用原始prompt没问题
   - LLaMA-3.2-Instruct等需要用chat template
   - 例如：`"[INST] Generate a text with positive sentiment: [/INST]"`

3. **Pseudo-relabeling**：
   - 需要PLM能做zero-shot classification
   - 对instruction模型，relabeling prompt可能也需要调整

4. **Supervised Contrastive Learning**：
   - 多分类时P(i)和A(i)的构建要注意类别平衡
   - 类别数增加时可能需要增大memory bank

5. **评估**：
   - 所有实验跑5个random seeds，报告mean ± std
   - 用paired t-test或McNemar's test验证显著性

---

## 六、风险管理与保底策略

| 风险 | 可能性 | 影响 | 应对策略 |
|------|-------|------|---------|
| 复现结果偏差大 | 中 | 高 | 仔细对照原文超参数；偏差本身也是有价值的发现 |
| 大模型生成太慢 | 低 | 中 | 使用量化；减少候选模型数量 |
| 现代LLM反而不如GPT2-XL | 中 | 中 | 这本身是有趣的negative result，分析原因即可 |
| 多分类效果差 | 中 | 低 | 分析为什么差（noise rate↑？label空间太大？），作为discussion |
| Tweet数据集不允许使用 | 高 | 低 | 提前准备替代数据集（Financial PhraseBank等） |
| 时间不够 | 中 | 高 | 优先级：模块1 > 模块2 > 模块3 > 模块4 |

---

