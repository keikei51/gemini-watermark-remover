# 核心水印去除算法调研与重构方案

日期：2026-06-08

## 背景

这次 `20260608` 样例暴露的问题不是单张图的偶发参数，而是算法架构正在偏离原始数学模型：

- 水印去除本质上是已知白色 logo 与 alpha map 的反向 alpha composite。
- 如果位置、尺寸、alphaMap、alphaGain 都正确，理论上不需要 inpainting、edge cleanup、background cleanup 等后处理。
- 如果位置或 alpha 错了，后处理会把错误候选伪装成“残差较低”，反而污染候选排序。

因此下一轮重构目标是：主流程回到 `位置候选 + alpha 候选 + 反解评分 + 早退`，把后处理从默认核心链路中移除或关停。

补充决策：当前阶段放弃自动搜索。默认主流程只尝试固定组合表里的位置/尺寸/alpha 组合；不命中就跳过，不用 preview/adaptive/local/size sweep 去兜底。

## 外部调研结论

### 1. 同类 Gemini 项目的主线都是 reverse alpha

`remove-ai-watermarks` 明确描述 Gemini / Nano Banana sparkle 使用：

```text
watermarked = alpha * logo + (1 - alpha) * original
original = (watermarked - alpha * logo) / (1 - alpha)
```

它的核心点有三个：从纯黑背景提取 alpha map、用 NCC 检测位置和尺度、对 Gemini sparkle 适配不同 opacity。它也有 residual inpaint，但文档把它定位为边缘残留清理，而不是主算法。参考：[remove-ai-watermarks README](https://github.com/wiltodelta/remove-ai-watermarks)。

`GeminiWatermarkTool` 也采用同样的确定性反解，并将 Gemini 3.5 的变化抽象为 profile 变化：alpha map 与位置公式改变，数学不变。它的检测部分是三阶段 NCC：spatial NCC、gradient NCC、statistical variance，并有 confidence threshold 来跳过非水印图。它也提供 inpaint / denoise，但文档说明这是 resized / recompressed 场景下的残留补救。参考：[GeminiWatermarkTool README](https://github.com/allenk/GeminiWatermarkTool)。

Rust crate `gemini-watermark-removal` 的公开说明同样很简洁：通过 calibrated `48x48` 和 `96x96` alpha masks 做 visible Gemini watermark 的 reverse alpha blending，并明确不能去除 SynthID。参考：[docs.rs gemini-watermark-removal](https://docs.rs/crate/gemini-watermark-removal/0.1.1)。

结论：同类 Gemini 项目的共识是，Gemini 可见水印应优先建模为确定性 alpha composite 问题。后处理可以存在，但不应成为默认成功路径。

### 2. 学术方向也支持“先定位与 matte，再恢复”

Google Research / CVPR 2017 的 `On the Effectiveness of Visible Watermarks` 把一致水印去除建模为 multi-image matting：估计 watermark foreground、alpha matte 和原图背景。它强调一致性水印可被高精度估计和恢复，并明确区分“反转水印过程”与“inpaint 合成猜测”。论文还指出，非常小的 watermark / alpha 误差都会在恢复中变成可见 artifact。参考：[项目页](https://watermark-cvpr17.github.io/) 与 [CVPR PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Dekel_On_the_Effectiveness_CVPR_2017_paper.pdf)。

深度学习类可见水印去除，例如 `Split then Refine`、`WDNet`、`FE-WRNet`，通常用于未知水印、复杂水印、多域泛化。它们共同强调 localization / mask / decomposition，而不是直接全图 repaint。参考：

- [Split then Refine / deep-blind-watermark-removal](https://github.com/vinthony/deep-blind-watermark-removal)
- [Visible Watermark Removal via Self-calibrated Localization and Background Refinement](https://arxiv.org/abs/2108.03581)
- [WDNet: Watermark-Decomposition Network](https://arxiv.org/abs/2012.07616)
- [FE-WRNet](https://www.mdpi.com/2076-3417/15/22/12216)

结论：对于未知水印，神经网络/后处理有价值；对于 Gemini 这种已知形状、已知颜色、候选位置有限的水印，核心应是准确定位和 alpha 反解。

### 3. 原始导出与二次压缩要分层处理

AuraTuner 的技术说明把边界讲得比较清楚：deterministic restoration 最适合原始导出、固定透明 overlay；截图、缩放、转发、重压缩会破坏像素级反解。参考：[Inverse Alpha Watermark Cleanup](https://auratuner.com/blog/inverse-alpha-watermark-restoration)。

这对本项目的含义是：

- Gemini 原始下载图：默认应走核心反解，目标接近完美。
- 不支持截图、缩放、转发、重压缩等降级输入作为默认目标。
- 如果输入不是原始 Gemini 导出，算法可以返回 `no-watermark-detected` 或 `unsupported-degraded-input`，而不是为了“看起来去掉”进入后处理。
- 如果将来需要处理降级输入，应作为独立 profile 或单独产品能力，而不是混入默认核心。

## 我们当前算法核心回顾

### 1. 已经正确的核心

`src/core/blendModes.js` 是正确的数学核心：

```js
original = (watermarked - alpha * LOGO_VALUE) / (1 - alpha)
```

它支持 `alphaGain`，并对低 alpha 噪声、接近 1 的 alpha 做安全限制。这部分应保留为唯一像素恢复核心。

`src/core/geminiSizeCatalog.js` 已经把 Gemini 尺寸和水印配置做成 catalog，这是正确方向。它比简单 `width > 1024` 规则更符合 Gemini 离散尺寸输出。

`src/core/candidateSelector.js` 里已经有有用的指标：

- `originalSpatialScore`
- `originalGradientScore`
- `processedSpatialScore`
- `processedGradientScore`
- `nearBlackIncrease`
- `texturePenalty`

这些指标应该继续保留，但要重新分层：原始证据用于选 anchor，处理后残差用于选 alpha / 验证，损伤指标用于拒绝候选。

### 2. 当前架构问题

当前候选选择把 detection 和 restoration 混在一起：

- `pickBetterCandidate()` 同时处理 catalog、preview-anchor、local shift、validationCost、improvement。
- `validationCost` 里包含处理后残差、near black、texturePenalty，这些适合判断某个 alpha 是否好，但不适合让弱位置候选抢掉强 anchor。
- `source.includes(...)` 字符串判断太多，业务语义藏在 source 拼接里，不如 `provenance` 稳定。
- 为修样例不断增加局部 veto，例如保护 `48px + 96px margin` catalog，这说明排序模型不统一。

`src/core/watermarkProcessor.js` 里后处理链路过重：

- alpha recalibration
- over-subtraction recalibration
- catalog dark fine tune
- weak alpha fine tune
- preview background cleanup
- preview edge cleanup
- subpixel outline refine
- small preview refinement

其中 alpha recalibration / fine tune 属于核心参数搜索，可以保留并统一；background cleanup、edge cleanup、outline refine、small preview refinement 更像视觉补丁，应从默认主流程移走。

### 3. 历史算法债务

从 `f9f6ae9` 的历史版本看，之前还存在大量 `alphaGain > 1` 候选和 multipass 链路：

- `1.05` 到 `2.60` 的强 alpha 候选会把“重复处理/过扣”引入搜索空间。
- `removeRepeatedWatermarkLayers` 代表把同一区域反复剥离，但如果原图只有一次 alpha composite，多次反解没有数学依据。

这部分近期已经被压缩，但文档上要明确：默认核心不再接受 `alphaGain > 1` 和 multipass，除非有新的生成链路证据证明 Gemini 确实叠加了多层可见水印。

## 新核心架构

### 总流程

```text
buildPositionCandidates(image)
  -> 按先验频率排序
  -> 对每个 position candidate:
       scoreOriginalEvidence(candidate)
       如果原始证据明显不足，跳过或降级
       for alphaGain in prioritizedAlphaGains(candidate):
           inverseAlphaRemove()
           scoreResidualAndDamage()
           如果 earlyAccept(candidate, alphaGain):
               return result
       对当前位置做局部 alpha fine tune
  -> 如果没有早退，按分层 rankingKey 选最优
```

核心原则：

- 不同位置候选之间，先比较 `originalEvidenceTier`，再比较 processed residual。
- 同一位置候选内部，才让 alphaGain 和 residual 决定胜负。
- 后处理不参与默认评分，不修改像素，不改变候选排名。
- 固定组合优先于自动识别：Gemini 不太可能无限增加水印布局，优先维护一个小而高频的有限候选集。
- 有限尝试保护性能：候选枚举必须有硬上限，不能为了追求极端自动识别而扫大范围位置、尺寸、alpha。

### 位置候选排序

位置候选应该显式分组并带 `sourcePriority`：

| 优先级 | 候选族 | 说明 |
|---:|---|---|
| 0 | exact official catalog current | 精确 Gemini 尺寸当前配置 |
| 1 | known current variants | 已确认高频变体，如 `48px + 96px margin` |
| 2 | exact official legacy | 旧 `96px + 64px margin`，证据门控 |
| 3 | confirmed exception catalog | 如 `2816x1536 -> 96px + 192px margin` |
| 4 | near-official projected catalog | 近似官方尺寸的缩放投影 |
| 5 | fixed preview-sized variants | 已确认 preview 小尺寸组合，但必须来自固定表 |
| 6 | local shift around strong seed | 当前默认关闭，只作为历史/实验路径 |
| 7 | size jitter around strong seed | 当前默认关闭，只作为历史/实验路径 |
| 8 | preview-anchor fallback | 当前默认关闭，不参与生产主链路 |
| 9 | adaptive fallback | 当前默认关闭，不参与生产主链路 |

注意：`preview-anchor` 不能再因为 processed residual 轻微优势压过强 catalog。

候选数量应受控。推荐默认只尝试：

- catalog exact/current 组合
- catalog confirmed variants
- legacy / exception 组合
- 围绕强证据候选的小范围 local shift
- 最后才是有限 preview-anchor fallback

不做全图滑窗，不做大范围尺寸搜索，不做无限 alpha sweep。当前生产目标是固定组合链路，而不是自动发现未知位置。

### Alpha 候选排序

全局候选保持小而有序：

```js
[0.6, 1, 0.7, 0.85, 0.55]
```

原因：

- `0.6` 是 202606 新链路高频弱 alpha。
- `1` 是标准 alpha。
- `0.7 / 0.85 / 0.55` 是弱 alpha 附近的离散备选。
- `0.9` 不应放全局；它应从 `1` 过扣后的局部 fine tune 自然出现。

局部 fine tune 规则：

```text
bestAlpha around +/- 0.02 / +/- 0.04
只在同一 anchor 内执行
只改变 alphaGain，不改变位置
```

位置和 alpha 是组合搜索，数量增长很快。因此 alpha 搜索也必须有限：

- 高频 alpha 放前面，命中早退就结束。
- 只有强位置证据的候选才允许局部 fine tune。
- 弱 fallback 候选不做细扫，避免错误 anchor 靠 alpha 微调混入最优结果。

### 评分分层

不要再用单个 `validationCost` 决定所有事情。建议拆成四个结构化评分。

#### 1. originalEvidence

用于判断该位置是否真的有水印。

输入：

- `originalSpatialScore`
- `originalGradientScore`
- alpha-band luminance signature
- alphaMap 梯度结构匹配

输出：

```js
{
  tier: 'strong' | 'medium' | 'weak' | 'none',
  spatial,
  gradient,
  score
}
```

建议门槛：

- `strong`: spatial >= 0.3 且 gradient >= 0.12，或 gradient 极强。
- `medium`: spatial / gradient 任一有明显正信号，但不足 direct match。
- `weak`: 只有轻微信号。
- `none`: 跳过或只作为 fallback。

#### 2. residual

用于判断 alpha 反解后水印是否消失。

输入：

- `abs(processedSpatialScore)`
- `max(0, processedGradientScore)`
- `suppressionGain`
- alpha-band halo visibility
- before/after delta map 与水印模板的关系

输出：

```js
{
  cleared: boolean,
  spatialResidual,
  gradientResidual,
  suppressionGain,
  score
}
```

#### before/after 差异图的使用方式

可以引入 `delta = before - after`，但它不能单独作为“水印已去除”的证明。

原因是：只要我们按某个候选执行反解，差异图天然会带有该候选 alphaMap 的形状；错误 anchor 也可能产生“看起来符合候选模板”的 delta。因此 delta 更适合作为辅助指标：

1. **修改范围约束**：主要差异应集中在 alphaMap 有效区域，alpha 很低的外圈不应出现大量改动。
2. **方向约束**：白色水印去除通常让像素变暗，`before - after` 应多数为非负；大量反向变化说明候选或 alpha 异常。
3. **幅度一致性**：对给定 alphaGain，理论改变量近似为：

```text
expectedDelta = alpha * (255 - before) / (1 - alpha)
```

实际 `before - after` 应与 expectedDelta 在 alpha 有效区域保持一致。
4. **残差联合判断**：最终仍必须看 after 图上的残余水印相关性，尤其是 `processedSpatialScore` 和 `processedGradientScore`。

更合理的判断不是“差异越小越好”，而是：

```text
原图有水印证据
反解产生了符合 alpha composite 的差异
反解后图像不再有水印形状相关性
反解没有明显伤图
```

2026-06-08 实验结论：

- `recompose(after + watermark) ~= before` 不能作为主评分。因为 `after` 是用同一组 alpha 反解出来的，同 alpha 重合成在数学上天然接近 `before`，无法区分 `alpha 0.65 / 0.7 / 0.74` 这类视觉差异。
- `diffTemplateCorrelation` 和 `diffGradientCorrelation` 也不能单独排序。实际样例中它们在多个 alpha 候选间几乎恒定，只能说明“差异形状来自该 alpha”，不能证明 after 已干净。
- 更有用的是 diff 派生的 artifact 指标：after 图上的 `spatial / gradient residual`、alpha band 的暗/亮 halo、以及 newly clipped pixel ratio。
- 当前代码已在 `src/core/restorationMetrics.js` 落地 `assessRemovalDiffArtifacts()`，用于把这些指标归一到 alpha 候选视觉成本；它是候选评分的一部分，不是唯一判据。

#### 3. damage

用于拒绝过扣或伤图。

输入：

- `nearBlackIncrease`
- `texturePenalty`
- `tooDark`
- `tooFlat`
- `negative residual / sign flip`

输出：

```js
{
  safe: boolean,
  penalty,
  reason
}
```

#### 4. rankingKey

候选排序用分层 key，而不是加权混合：

```js
[
  sourcePriority,
  -originalEvidenceTier,
  damageSafe ? 0 : 1,
  residualScore,
  alphaPriorityIndex,
  damagePenalty
]
```

这里的关键是：`sourcePriority` 和 `originalEvidenceTier` 在前，避免错误位置被微小 residual 优势带飞。

### 早退条件

早退必须同时满足：

```text
sourcePriority 足够高
originalEvidenceTier >= strong
residual.cleared === true
damage.safe === true
```

建议初始阈值：

```text
abs(processedSpatialScore) <= 0.04
max(0, processedGradientScore) <= 0.12
suppressionGain >= 0.25
nearBlackIncrease <= 0.03
texturePenalty 不触发 hardReject
```

自动 fallback 候选当前默认不进入主流程。即使将来重新开启，也不能直接早退，除非固定 catalog / standard 全部无有效证据。

早退同样是性能保护机制。固定高频组合如果通过阈值，应立即返回；不要为了寻找理论上更低的 residual 继续搜索低频候选。

## 与现有代码的映射

建议新增或重构为以下模块：

```text
src/core/watermarkCandidatePipeline.js
  buildPositionCandidates()
  buildAlphaCandidates()
  evaluateCoreCandidate()
  rankCoreCandidates()
  selectBestCoreCandidate()

src/core/watermarkScoring.js
  scoreOriginalEvidence()
  scoreResidual()
  scoreDamage()
  shouldEarlyAccept()

src/core/watermarkProcessor.js
  只负责协调 pipeline、生成 meta、保留 public API
```

现有模块保留：

- `blendModes.js`: 核心反解。
- `geminiSizeCatalog.js`: 候选先验。
- `adaptiveDetector.js`: spatial / gradient correlation 和 alphaMap resize/warp 工具。
- `restorationMetrics.js`: damage / texture / halo 指标，但不要让它直接驱动 anchor 抢占。

后处理函数处理方式：

- 先保留代码但默认不调用。
- 增加显式 debug / experimental flag 才能打开。
- 新核心稳定后删除 dead code。

## 测试策略

### 样例回归

必须覆盖：

- `20260607.png`
- `20260607-2.png`
- `20260608-2.png`
- `20260608-3.png`
- `20260608-4.png`
- `20260608-5.png`
- `2816x1536` known exception
- 官方尺寸 0.5K / 1K / 2K / 4K 合成 fixture

每个样例断言：

- selected anchor family
- `position`
- `alphaGain`
- `sourcePriority`
- `originalEvidenceTier`
- `earlyExit` 是否发生
- residual / damage 指标

### 反例测试

必须覆盖：

- 干净图不应处理。
- 白色高亮图案不应被误判为 watermark。
- 弱 preview-anchor 不应压过强 catalog。
- 错误位置 residual 偶然较低也不能胜出。
- `alphaGain > 1` 默认不进入候选。
- multipass 默认不进入主流程。

### 可视化输出

继续保留 `.artifacts` contact sheet，但它只能作为人工验证，不作为算法排名依据。

## 实施顺序

1. 抽出 `watermarkScoring.js`，把 original / residual / damage 分开。
2. 默认关闭自动搜索：preview-anchor sweep、adaptive、local shift、size jitter、template warp 都不参与主链路。
3. 抽出候选对象和 `rankingKey`，替换 `pickBetterCandidate()` 的局部 veto 链。
4. 把 alpha search 固定在每个 anchor 内部执行。
5. 增加 before/after delta 辅助评分，但只作为 residual / damage 的组成部分，不作为唯一判据。
6. 关停默认后处理：background cleanup、edge cleanup、subpixel outline、small preview refine。
7. 设置候选数量上限和早退阈值，保护页面实时处理性能。
8. 用现有样例重跑，校准 early accept 阈值。
9. 再决定删除哪些后处理死代码。

## 当前结论

新方向不是“少做一点”，而是更严格：

- 只相信数学反解。
- 只让 alpha 搜索修 alpha。
- 只让候选排序选位置。
- 只让后处理作为显式降级工具，不参与默认核心。
- 只支持原始 Gemini 导出图，不为降级输入牺牲主流程简洁性。
- 只做有限候选尝试，用高频固定组合和早退保护性能。

这能让未来 Gemini 改位置、改 alpha、改尺寸时，我们增加候选和先验，而不是继续叠视觉补丁。

## 2026-06-08 重构复盘

这次重构确认了一个核心事实：失败样例不应该用后处理逐张修，而应该回到水印模型本身。Gemini 可见水印仍然更像一个固定 logo、固定 alpha map、有限位置和有限 alpha gain 的反向 alpha composite 问题。

### 已确认的发现

1. Gemini 链路确实存在变化。

   `20260607` 和 `20260608` 样例说明，水印位置、边距和有效 alpha gain 都可能随 Gemini 链路变化。问题不是单纯位置偏移，也不是单纯 alpha 强度变化，而是不同输出尺寸和生成链路会落到不同固定组合。

2. `maxPasses` 与 alpha gain 候选重复。

   多次去水印本质上是在模拟更强 alpha，但它比显式枚举 alpha gain 更难解释、更难评分，也更容易制造暗边和纹理损伤。即使真实存在重复水印，多 pass 也不能保证完美反解，因此不应作为默认主链路。

3. 后处理不应作为成功路径。

   background cleanup、edge cleanup、subpixel outline、残影修补等逻辑容易把错误候选伪装成较低残留，污染评分。对于原始 Gemini 导出图，只要位置和 alpha 正确，核心反解应接近完美；如果不完美，优先怀疑候选和评分，而不是追加修图补丁。

4. 固定组合优先是合理假设。

   Gemini 不太可能引入无限位置和无限 alpha。更可维护的策略是维护一个高频固定组合表，把最常见的 `position + alpha` 放前面，低频组合必须有证据门控。自动搜索、二分查找和大范围 sweep 暂时放弃。

5. 评分算法比候选数量更关键。

   候选可以保持有限，但评分必须能区分“水印残留”和“过度扣除”。之前的问题往往不是没有候选，而是局部残留、暗 halo、背景纹理和裁剪像素没有被统一建模，导致个别样例选错 alpha 或 anchor。

6. before/after diff 有价值，但不能朴素使用。

   实验确认，`after + watermark ~= before` 的重合成误差在多个 alpha 候选上都可能很小，区分度不足。`before - after` 与模板的相关性也会随 alpha 近似等比例变化，不能单独决定胜负。更有用的是从 diff 中派生视觉伪影指标，例如暗 halo、新增裁剪像素、残留梯度和局部纹理异常。

### 当前代码落地

当前代码已把 diff 思路落为 `assessRemovalDiffArtifacts()`，但它不是唯一判据，而是为候选评分提供一组辅助指标：

- `recomposeError`: 诊断用，说明候选是否符合 alpha composite 形状。
- `diffTemplateCorrelation`: 诊断用，说明被移除区域是否像水印模板。
- `diffGradientCorrelation`: 诊断用，辅助观察差异边界。
- `negativeDiffRatio`: 观察是否出现反向差异。
- `newlyClippedRatio`: 衡量反解是否制造新的黑/白裁剪。
- `halo`: 衡量 alpha band 周边是否出现暗边或亮边。
- `visualArtifactCost`: 综合伪影成本，参与 alpha 微调排序。

这条线的结论是：diff 可以帮助发现“去掉了什么”和“是否过度去除”，但不能替代 residual score 和候选先验。

### 当前样例状态

本轮固定核心样例输出在：

```text
.artifacts/fixed-core-sample-check/processed
.artifacts/fixed-core-sample-check/contact-sheet.png
.artifacts/fixed-core-sample-check/summary.json
```

关键样例当前结果：

- `20260608-3.png`: `48px / r96 b96 / alpha 0.7`
- `20260608-4.png`: `48px / r96 b96 / alpha 0.95`
- `20260608-5.png`: `48px / r96 b96 / alpha 0.64`
- `20260520-3.png`: 使用 legacy 96 alpha map，避免被新版 alpha 误判。
- `5-4.webp`: 已恢复为支持样例，依赖固定核心强证据和 near-black 窄例外。

已通过验证：

```text
pnpm test
pnpm build
git diff --check
```

其中 `git diff --check` 只有 Windows 换行提示，没有空白错误。

## 后续演进方向

下一阶段不建议继续逐图微调，而应把固定组合链路做成可观测、可维护、可回归的系统。

### P0: Candidate ranking report

为每张样例输出 top N 候选，而不是只输出最终结果。每个候选至少记录：

- watermark size
- right / bottom margin
- alpha gain
- alpha map profile
- original evidence
- residual score
- diff artifact cost
- damage / halo / clipping 指标
- 是否 early accept
- 失败或胜出的主要原因

目标是让失败定位变成可解释问题：

- 候选没覆盖：补 catalog。
- 候选覆盖但排序错：改评分。
- 候选和评分都对但视觉仍差：检查 alpha map 或输入是否降级。

### P1: 人工 gold set manifest

为样例目录增加人工标注，记录：

- 是否应该处理。
- 是否已知不支持。
- 期望 anchor family。
- 期望 alpha gain 范围。
- 是否允许弱残留。
- 是否是特殊背景风险样例。

这能把“测试当前代码行为”升级为“测试人类视觉预期”。新增 Gemini 链路样例时，应先进入 gold set，再调整 catalog 或评分。

### P2: Catalog 显式知识库

把尺寸、位置、alpha、alpha map profile 和来源说明组织成更显式的 catalog：

- 高频组合排前。
- 低频组合必须 evidence-gated。
- 每个组合记录来源样例或 Gemini 链路。
- 新增组合必须附带回归样例。
- 不让 local shift、size jitter、adaptive sweep 抢占固定 catalog。

Gemini 未来变化时，主要更新 catalog，而不是改核心算法。

### P3: 评分分层

评分应拆成三类，避免一个综合分吞掉所有语义：

- `originalEvidence`: 原图中是否真的有 Gemini 水印。
- `residualScore`: 处理后是否还有水印形状。
- `artifactCost`: 是否制造了暗边、裁剪、纹理坍塌或局部异常。

最终排序可以使用统一 ranking key，但 debug meta 必须保留分项指标，便于判断是残留问题还是损伤问题。

### P4: 可视化调试页

在 contact sheet 之外，增加可展开的候选排序视图：

- 原图 / 最终图。
- top N 候选缩略图。
- 每个候选的评分表。
- alpha band halo 可视化。
- before/after diff 可视化。
- residual heatmap。

这会显著降低之后调参和判断失败样例的成本。

### P5: 质量型发版

固定核心、ranking report、gold set 和 catalog 稳定后，再做一次质量型版本发布。发版前应固定流程：

```text
pnpm test
pnpm build
生成 sample artifact
人工检查 contact sheet
检查 candidate ranking report
更新 changelog
打包 release
```

## 长期原则

- 不支持降级输入作为默认目标。
- 不恢复默认 multipass。
- 不恢复默认后处理。
- 不做无限自动搜索。
- 不用单一 diff 指标决定胜负。
- 新位置和新 alpha 必须通过样例进入 catalog。
- 算法核心保持透明水印反解，复杂度转移到 catalog、评分和观测工具。
