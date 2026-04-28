# The Algorithmic Exploitation Ratchet: A Quantitative Execution Plan

## 1. 论文定位与投稿目标

**核心叙事 (The Narrative):**
We provide the first quantitative, causal evidence that algorithmic engagement rewards act as a "ratchet mechanism," driving family vlog channels toward increasingly exploitative content involving children—an effect that is absent in adult-only creator channels.

**投稿目标推荐:**
1. **ICWSM 2027 (首选)** 
   - **Deadline:** May 15, 2026 (Round 1) / Sept 15, 2026 (Round 2)
   - **Why:** 完美的契合度。ICWSM 是 Computational Social Science 的最高殿堂，极其偏爱这种具有重大社会意义、方法论扎实（Causal + NLP）的平台研究。
2. **WWW 2027 (备选 1)**
   - **Deadline:** ~Oct 2026
   - **Why:** Web Science 领域的顶会，对社会影响力和算法审计（Algorithm Auditing）非常感兴趣。
3. **AAAI 2027 (备选 2)**
   - **Deadline:** ~Aug 2026 (AI for Social Impact Track)
   - **Why:** 专门的社会影响力 track 非常适合这种跨学科工作。

---

## 2. 数据收集计划 (Data Collection)

**目标:** 构建第一个大规模、纵向的 "Kidfluencer Content Evolution Dataset"。

### 2.1 频道选择 (Channel Selection)
我们需要构建两组对比频道：
- **Treatment Group (Family Vlogs):** 50-100 个包含儿童出镜的家庭频道。
  - *来源:* SocialBlade 排行榜，维基百科 "List of family vloggers"，以及文献中提到的争议频道（如 The ACE Family, 8 Passengers, DaddyOFive, Ryan's World, JesssFam 等）。
- **Control Group (Adult-only Vlogs):** 50-100 个纯成人出镜的生活方式/旅行/美食频道，且粉丝量级和活跃年份与 Treatment Group 匹配。
  - *来源:* Casey Neistat, Mark Wiens, Peter McKinnon 等。

### 2.2 数据抓取 (Data Extraction)
- **工具:** YouTube Data API v3 (必须用官方 API 获取精确的 view counts，`yt-dlp` 的 flat-playlist 拿不到 view count)。
- **获取字段:** 
  - `video_id`, `published_at`, `title`, `description`, `view_count`, `like_count`, `comment_count`, `duration`.
- **规模预期:** 100-200 个频道 $\times$ 平均 1000 个视频 = 100,000 - 200,000 个视频的元数据。

---

## 3. 方法论设计 (Methodology)

### 3.1 剥削方向的量化 (Exploitation Drift Score)
我们采用完全无监督的 Diachronic Embedding 方法来量化内容漂移，避免人工标注的主观性和成本。

1. **文本表示:** 将视频的 `title + description` 通过 `sentence-transformers` (如 `all-MiniLM-L6-v2`) 映射为 384 维向量 $v_i$。
2. **定义"剥削"锚点 (Anchors):**
   - 收集 100 个已知极端剥削案例的视频标题（如 8 Passengers 被捕前的标题，或 DaddyOFive 的虐童恶作剧标题）作为 $E_{exploitative}$。
   - 收集 100 个健康/教育类儿童视频标题（如 Sesame Street）作为 $E_{healthy}$。
3. **计算方向向量:** $\vec{d} = mean(E_{exploitative}) - mean(E_{healthy})$。
4. **计算 Drift Score:** 对任意视频向量 $v_i$，其剥削漂移分数为它在方向向量上的余弦相似度投影：$Score_i = \cos(v_i, \vec{d})$。

### 3.2 算法激励的因果检验 (Causal Analysis)
我们需要证明"爆款视频"（Treatment）导致了"后续内容剥削分数的上升"（Outcome）。

1. **定义 Treatment (Viral Hit):** 
   - 对每个频道，计算其历史平均播放量 $\mu_v$ 和标准差 $\sigma_v$。
   - 如果视频 $i$ 的播放量 $V_i > \mu_v + 2\sigma_v$，且其 $Score_i$ 高于近期平均水平，则标记为一个 "Exploitative Viral Hit" ($T=1$)。
2. **定义 Outcome (Content Shift):**
   - 观察 Treatment 之后 $k$ 个视频（如 $k=10$）的平均 Drift Score，与 Treatment 之前 $k$ 个视频的平均分之差 $\Delta Score$。
3. **因果推断模型 (Difference-in-Differences / Interrupted Time Series):**
   - 比较 Treatment Group (Family Vlogs) 和 Control Group (Adult Vlogs) 在经历类似 "Viral Hit" 后的 $\Delta Score$ 差异。
   - **核心假设:** 家庭频道在尝到"极端内容"的甜头后，会系统性地将后续内容向剥削方向偏移（Ratchet Effect），而成人频道不会（或者程度显著更轻）。

---

## 4. 实验执行时间线 (Timeline)

总计约 6-8 周，完全适合一个人独立执行。

### Week 1: 数据集构建
- [ ] 申请 YouTube Data API quota。
- [ ] 整理 100 个 Family Vlogs 和 100 个 Control Vlogs 的 Channel ID 列表。
- [ ] 编写 Python 脚本，拉取所有频道的完整视频历史（包含 view counts）。
- [ ] 数据清洗，去除 Shorts（通常少于 60 秒）和直播回放，专注于长视频。

### Week 2: NLP 向量化与锚点定义
- [ ] 收集 100 个极端案例标题和 100 个健康案例标题，定义 $\vec{d}$。
- [ ] 用 `sentence-transformers` 跑完全部 20 万个视频的 embedding。
- [ ] 计算所有视频的 Exploitation Drift Score。
- [ ] **Validation:** 随机抽样 200 个视频，用 GPT-4 做 zero-shot 打分，计算与 Drift Score 的 Pearson 相关系数，以证明方向向量的有效性。

### Week 3: 时间序列与因果分析
- [ ] 识别所有频道的 "Viral Hits"。
- [ ] 运行 Interrupted Time Series (ITS) 或 DiD 模型。
- [ ] 绘制内容漂移的时间序列图（类似于 Pilot Study，但加上 view count 的标注）。
- [ ] 提取统计显著性结果（p-values, effect sizes）。

### Week 4: 结果分析与对照组对比
- [ ] 对比 Family Vlogs 和 Adult Vlogs 的 Ratchet Effect 差异。
- [ ] 深入分析 (Deep Dive) 3-5 个典型频道的演变史（如 The ACE Family）。
- [ ] 整理图表（Bar charts, Time series plots, Causal effect plots）。

### Week 5-6: 论文撰写
- [ ] **Introduction:** 引入 Kidfluencer 剥削的社会背景，指出缺乏量化证据的痛点。
- [ ] **Related Work:** 区分社科定性研究和 NLP 内容分析，强调本研究的独特性。
- [ ] **Dataset & Methodology:** 详细描述数据收集、锚点定义和因果模型。
- [ ] **Results:** 展示统计结果，强调 Ratchet Effect 的存在。
- [ ] **Discussion:** 探讨算法设计对创作者行为的反向塑造（Reverse Shaping），并为政策制定（如 Coogan Law 的数字化扩展）提供建议。

---

## 5. 潜在风险与应对 (Risks & Mitigations)

1. **API 限制:** YouTube API 有每日 quota 限制。
   - *应对:* 提前申请提高 quota，或者用多个 Google 账号的 API key 轮换抓取。如果 API 实在不够，可以退回用 `yt-dlp` 抓标题，用其他第三方工具（如 SocialBlade 历史数据）补充 view count。
2. **方向向量的解释性:** 审稿人可能质疑 $\vec{d}$ 是否真的代表"剥削"。
   - *应对:* 加入 Week 2 的 GPT-4 交叉验证环节。如果无监督分数的排序与 GPT-4 的判断高度一致，方法论就立得住。
3. **因果关系的内生性:** 播放量高可能是因为平台推荐，也可能是因为外部事件。
   - *应对:* 承认这是 observational causal inference 的局限性，在 Limitations 中坦诚讨论，重点强调这是"第一个大规模的量化证据"，为未来的受控实验提供基础。

## 6. 为什么这个项目对 NIW 完美契合？

- **独立性:** 整个 pipeline（数据抓取、NLP、因果推断）由你一人完成，且**完全不涉及 Meta 的数据或 IP**。
- **社会价值:** 直接回应美国当前最热的数字监管议题（儿童在线隐私与劳动保护）。
- **学术定位:** 跨越了 CS 和 Social Science 的鸿沟，属于真正的 AI for Social Good，极易获得推荐人的高度评价。
