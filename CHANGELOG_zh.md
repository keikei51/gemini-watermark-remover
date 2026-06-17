# 更新日志

## 1.0.25 - 2026-06-17

### 水印移除

- 修复 2026-06-16 Gemini 样例图：优先保留右下角 canonical 96px 证据，避免弱的向内候选抢占，并防止深色纹理背景被过度去除。
- 为浅色、非目录位置的残留加入更保守的 smooth-prior 清理，降低白色星形残留，同时避免变成暗色伪影。

### 质量

- 增加该 bug 图和有界弱残留结果的回归覆盖。
- 重新验证外部 Gemini watermark 样例集，190 张中 188 张通过，且没有新增失败样例。
- 由于此前 npm `1.0.24` tarball 已不可变且早于本次修复，发布 `v1.0.25` patch 版本承载该修复。

## 1.0.24 - 2026-06-16

### 视频

- 改进 Veo 文字水印在稳定低对比度竖屏视频中的检测表现，覆盖新增的西藏风景样例。
- 对低对比度文字水印继续保持保守判定：必须在默认模板位置反复出现，并且至少有一帧达到强证据，才会接受为可信检测。

### 质量

- 新增间歇性低对比度 Veo 文字证据的回归测试。
- 重新通过本地视频 UI preset 导出器验证新增的 720p、720x1280 和 Veo 文字样例集。
- 已发布 `v1.0.24` 的 npm 包、GitHub Release 资产、userscript 产物和 Chrome 插件备用 zip。
- 官网已同步到 `1.0.24` 包和 release 资产。官网侧另做了一次 follow-up：为视频 runtime 脚本 URL 加上内容版本参数，避免站点部署后固定 runtime 路径继续命中旧 JavaScript 缓存。

## 1.0.23 - 2026-06-14

### 视频

- 修复 issue #77 报告的浏览器视频 ONNX runtime 尺寸不匹配问题：小尺寸水印改走 104px FDnCNN 模型，标准或未知水印继续走 200px 模型。
- 新增固定 shape ROI 规划和 resize fallback，确保所有视频水印候选送入 ONNX runtime 前都符合所选模型的输入尺寸。
- 将 runtime padding fallback 下沉到视频导出层，让直接调用视频导出接口时也能避开同类固定 shape 不匹配。
- 修复本地 Before / After 视频预览对竖屏视频的裁切问题，确保右下角水印区域在审查时保持可见。
- 收紧 Veo 文字水印检测的搜索范围，并加入协作式进度让出，避免检测期间页面长时间假死。

### 质量

- 补充覆盖视频水印目录的回归测试，包括 standard、inset、compact、scaled、portrait、4K、超大 8K 和极小画布 ROI 场景。
- 增加发版安全检查，防止视频页和官网 runtime bundle 退回固定 200px 模型或写死 64px padding 的旧路径。
- 明确公开发布口径：当前图片默认链路和受门控的图片改进可以发布；视频清理在发版 gate 提升前仍按 review-scoped 能力描述。

## 1.0.22 - 2026-06-14

### 水印移除

- 支持 2026-06-13 样例暴露的新 Gemini near-official 大边距锚点。
- 新增证据门控的小尺寸锚点重定位，用于处理可见 fixed-local 残留，并为高置信 48px 大边距残留加入更强的 mid-alpha 调整。
- 对剩余样例中虽然分数更低但会产生明显暗边伪影的候选保持拒绝，避免为过拟合牺牲画面安全。

### 质量

- 补充 2026-06-13 锚点、小尺寸锚点重定位、stronger mid-alpha 选择的回归测试。
- 重新验证外部 Gemini watermark 样例集，189 张中 186 张通过，且没有新增失败样例。

## 1.0.21 - 2026-06-12

### SDK / CLI

- 新增 `@pilio/gemini-watermark-remover/video` SDK 入口，支持本地视频去水印，可注入自定义处理器，并提供基于 Playwright 预览页的默认处理路径。
- 在 Node SDK 中重新导出视频辅助函数，并让 CLI 自动识别 `.mp4`、`.webm`、`.mov` 输入进入视频流程。
- 新增视频预览页、去噪后端、超时时间、低置信度导出等 CLI 参数。

### 视频

- 新增本地视频预览导出器使用的浏览器 AI 清理路径，并记录相邻帧复用遥测，提升重复水印区域的处理效率。

## 1.0.20 - 2026-06-09

### SDK

- 将 `sharp` 从强制运行时依赖调整为 optional peer，避免浏览器消费者在不使用 Node 图片 codec 时安装 native 依赖。
- 补充 CLI 文档：使用内置文件解码/编码路径时需要额外安装 `sharp`。

## 1.0.19 - 2026-06-09

### SDK

- 将最新 Gemini 水印候选识别改进作为新的 npm SDK 版本发布，因为 `1.0.18` 已经存在于 npm。
- 保持 SDK 发布入口不变，让 Pilio 可以依赖公开包，而不是继续维护 vendored copy。

### 质量

- 复用已经验证通过的 `1.0.18` 算法构建作为这次 npm-only 发布的基线。

## 1.0.18 - 2026-06-08

### 水印移除

- 将 Gemini 固定核心去水印链路重构为按优先级选择位置和 alpha 候选，不再依赖 multipass 或视觉后处理作为默认成功路径。
- 新增基于 before/after diff 派生的伪影评分，让 alpha 选择同时考虑残留边缘、halo 和新增裁剪像素，但不把 diff 当成唯一判据。
- 新增 2026-06-08 回归样例，覆盖最新 Gemini 48px 水印变化和弱 alpha 调整结果。

### Chrome 插件

- 将官方 Chrome 插件发布包移动到顶层 `release/` 目录，同时保留 `dist/extension` 作为本地调试版未打包插件。

### 质量

- 记录固定核心算法复盘和后续演进方向，包括 candidate ranking report、gold set manifest 与 catalog 驱动维护。
- 重新完成全量测试、生产构建、样例 artifact 生成和插件发布包生成验证。

## 1.0.17 - 2026-06-07

### 水印移除

- 支持新观察到的 Gemini 48px 水印右下 96px 边距锚点。
- 新增按优先级评估的 alpha 强度参数组：新弱 alpha 链路优先尝试 60% 强度，必要时再回退到标准 100% 链路。
- 继续对旧版 96px 水印和 192px 边距候选做证据门控，确保旧输出和全尺寸输出仍能安全处理。

### 质量

- 新增 2026-06-07 回归样例，覆盖弱 alpha 的 48px / 96px 锚点输出。
- 重新验证当前样例 benchmark，23 张样例全部通过。

## 1.0.16 - 2026-05-21

### Chrome 插件

- 在插件弹窗底部显示当前版本号，方便确认已安装的构建版本。
- 复用 Gemini 复制动作后的全屏图片会话状态，避免刷新后的 `blob:` 图片再次触发页面可见的 Processing。
- 将复制 fallback 的处理结果写入全尺寸会话缓存，后续复制和下载可以复用已处理图片。

### 质量

- 补充全屏弹窗动作提示、剪贴板 fallback 缓存、全屏刷新图复用相关回归测试。

## 1.0.15 - 2026-05-20

### 水印移除

- 支持新观察到的 Gemini 2K 输出所使用的新版 96px 水印 alpha 模板，以及右下 192px 边距锚点。
- 收紧候选选择逻辑：当标准 48px/96px 锚点已经干净时，避免被残留边缘更重的小尺寸 preview-anchor 候选抢走。
- 保留 preview-anchor 自身的 warp 与 edge-cleanup 后处理路径，避免真实 Gemini 预览图回归。

### 质量

- 补充新版 Gemini 水印位置、2026-05-20 样例图，以及备用 96px alpha 模板的回归覆盖。
- 重新验证当前样例集的全量测试、生产构建和样例 benchmark。

## 1.0.14 - 2026-05-03

### 油猴脚本

- 将 userscript 自动更新元信息切换到 GitHub Release `latest/download` 固定链接，确保更新端点由已发布的 release 资产控制。

### 质量

- 更新回归测试，锁定基于 release 的 userscript 自动更新地址。

## 1.0.13 - 2026-05-03

### 油猴脚本

- 新增托管版 `@downloadURL` 和 `@updateURL` 元信息，让 userscript 管理器可以从官方脚本固定链接自动更新。

### 质量

- 补充托管 userscript 自动更新元信息的回归测试。

## 1.0.12 - 2026-04-29

### Chrome 插件

- 将插件弹窗默认文案调整为英文，适配 Chrome Web Store 首次提交。
- 优化插件弹窗视觉设计，采用更接近 Apple 风格的间距、柔和面板、统一蓝色强调、内联操作图标和 GitHub 反馈入口。
- 将 Manifest V3 元数据中的插件名称简化为 `Gemini Watermark Remover`。

### 质量

- 重新构建 Chrome 插件发布产物，并复验插件构建、兼容适配层和当前清理覆盖。

## 1.0.11 - 2026-04-17

### Chrome 插件

- 新增 Manifest V3 Chrome 插件构建，通过 Tampermonkey 兼容适配层打包共享 userscript runtime。
- 新增插件弹窗，包含启用开关、官网入口、通用去水印入口和 GitHub issue 反馈入口。
- 新增版本化插件发布包流程，可生成 zip、sha256 校验文件和 `latest-extension.json`，用于 GitHub Release 和官网下载。

### SDK

- 新增公共 `runtime-browser` 入口，作为无副作用的浏览器 blob 处理器，供下游页面项目直接复用。
- 新增公共 `runtime-userscript` 入口，提供窄包装的 userscript runtime 接口，显式暴露 initialize/process/remove/dispose。
- 为两个 runtime 入口补齐类型声明，让打包后的 TypeScript consumer 可以直接导入。

### 工具链

- 更新了 package exports 与发布白名单，`pnpm pack` 现在会正确包含 runtime 入口及其所需的共享实现文件。
- 补充了隔离 consumer smoke 覆盖，验证 runtime 子路径可导入，并显式拒绝 `@pilio/gemini-watermark-remover/src/...` 这种深层私有导入。
- 在中英文 README 中补充 Chrome 插件安装入口，并在发版清单中加入插件产物检查。

### 质量

- 新增 runtime 回归测试，覆盖浏览器入口的无副作用导入、默认处理选项、脱离实例调用，以及 userscript worker fallback 行为。
- 新增插件构建、兼容层、弹窗、发布元数据和 README 顺序相关回归覆盖。
- 已重新完成面向 page/runtime/sdk/package-consumer 的验证，并对 `1.0.11` 做过一次新的发布 dry run。

## 1.0.10 - 2026-04-07

### 油猴脚本

- 对被动的 preview 请求拦截改为 fail-open：当 request-layer 预览处理失败时，让 Gemini 继续显示原始页面图片，而不是把页面链路卡死。
- 加固了 Gemini 全屏复制链路：当已处理 object URL 失效时，不再回退到会被 CSP 拦截的 `fetch(blob:...)`，而是改为直接重处理 Gemini 自己写入剪贴板的图片数据。
- 稳定了全屏预览替换：全屏 dialog 中的 blob 图片会优先复用 session store 里记住的 preview source 绑定，并且在页面替换队列里优先于普通预览图处理。

### 质量

- 补充了 stale fullscreen clipboard object URL、共享 image session 的 fullscreen preview source 复用，以及 fullscreen 优先队列行为的回归测试。
- 已重新完成一次全量自动化测试、生产构建，以及固定 profile 上的 Tampermonkey userscript freshness 校验。

## 1.0.9 - 2026-03-31

### 油猴脚本

- 从本地应用流程中移除了 Gemini 原图来源确认，是否为 Gemini 图改由用户自行判断。
- 简化了跳过处理时的状态文案，改为“未检测到可移除水印”，不再假装系统能可靠判断 Gemini 来源。
- 删除了已废弃来源识别链路对应的 `exifr` 依赖。

### 工具链

- 为本地 dev 静态服务加上禁用浏览器缓存的响应头，降低当前 `pnpm dev` 实际端口（从 `http://127.0.0.1:4173/` 起探测）继续命中旧 bundle 的概率。

### 质量

- 新增回归测试，确保应用不再导入 Gemini 来源判断 helper，语言包也不再保留已删除的来源确认文案。
- 已重新完成一次全量自动化测试、样例 benchmark 与生产构建验证。

## 1.0.8 - 2026-03-31

### 油猴脚本

- 修复了去元数据输入图的 Gemini 来源确认逻辑：当 EXIF 缺失时，改为回退读取图片真实尺寸，而不是只依赖 EXIF 宽高字段。
- 扩充了 Gemini 尺寸目录，覆盖当前项目样例里实际存在的长图和宽图输出规格。
- 调整了来源未确认时的提示文案，不再把“证据不足”错误表述成“不是 Gemini”。

### 工具链

- 移除了 `benchmark:samples` 和 `export:samples` 的本地浏览器依赖，两条脚本现在都直接走 Node 解码与编码链路。
- 调整本地回归夹具与测试，使现有的 WebP 样例集成为当前发布基线。

### 质量

- 补充了无 EXIF 来源回退，以及纯 Node 样例解码/导出链路的回归测试。
- 已重新完成全量自动化测试、SDK smoke 验证、样例 benchmark/export，以及生产构建验证。

## 1.0.7 - 2026-03-31

### 油猴脚本

- 改进了接近官方尺寸的纵向图和预览尺寸 Gemini 图片的水印锚点恢复，在默认锚点轻微漂移时更容易选中正确区域。
- 当首轮处理已经足够压低水印残留时，更早停止有害的额外 pass，减少把边缘重新做坏的风险。
- 继续让 preview-anchor 走更便宜的边缘清理路径，不再回到高开销且经常不采纳的子像素细化扫掠。

### 质量

- 为锚点恢复、pass 停止条件，以及发布元数据一致性补充了回归测试。
- 补充了本轮发布使用的 single-pass 与 multipass 取舍说明文档。

## 1.0.6 - 2026-03-30

### 油猴脚本

- 围绕共享的 image-session 和 `actionContext` 管线，统一了 Gemini 预览图、全屏图、复制与下载动作。
- 复用了不同使用面的 processed session 资源，让全屏复制和下载更稳定地拿到同一份去水印结果。
- 从当前运行路径里移除了过时的 userscript 意图别名，减少发布前的遗留兼容分支。

### 质量

- 为 `actionContext`、共享 image-session 解析，以及发布清理后的 userscript hook 行为补充了针对性覆盖。
- 已重新完成一次全量自动化测试与生产构建验证。

## 1.0.2 - 2026-03-20

### 油猴脚本

- 将 Gemini 页面图片替换链路继续拆成更小的共享 helper，分别覆盖处理准备、mutation 分发、source 分派和结果落盘，主流程更薄更清楚。
- 简化原图获取规则：预览图统一走 rendered capture，下载图统一走 background fetch，内联地址保留 direct fetch。
- 简化下载拦截缓存策略，只保留进行中的请求去重，不再持久缓存已处理响应。

### 质量

- 为 preview/original source 分派、候选图片收集、mutation 调度，以及自写 processed blob 识别补充了更细的回归测试。
- 已重新完成全量自动化测试与生产构建验证。

## 1.0.1 - 2026-03-19

### 油猴脚本

- 新增 Gemini 页面内预览图替换链路，页面图片现在可以在手动下载前先完成去水印处理。
- 预览图抓取在可用时优先走 `GM_xmlhttpRequest`，避免 userscript 沙箱里回退到普通 `fetch` 后触发 CORS 失败。
- 预览图处理期间新增克制的 `Processing...` 遮罩，处理失败时保持原图可见，不会把页面图片替换成空白状态。
- 加固了处理中遮罩的生命周期，避免旧的淡出回调误删新一轮处理状态。

### 共享显示链路

- 保持页面内图片替换行为与 userscript 预览处理链路、处理中体验一致。

### 质量

- 新增 userscript 版本同步测试，以及处理中遮罩生命周期边界测试。
- 已完成完整自动化测试与生产构建验证。
