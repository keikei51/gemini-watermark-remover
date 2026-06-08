# 发版清单

## 发布面

当前仓库有四个发布面：

- 网站构建产物：`dist/`
- 油猴脚本产物：`dist/userscript/gemini-watermark-remover.user.js`
- `package.json`、`src/core/`、`src/sdk/` 对应的 package/sdk 源码与元数据
- Chrome Web Store 商店页，以及备用安装包：`release/gemini-watermark-remover-extension-v<version>.zip`

## 发布前检查

在仓库根目录执行：

```bash
pnpm install
pnpm test
pnpm build
pnpm package:extension
```

预期结果：

- 所有测试通过
- `dist/` 下的网站构建产物已按当前代码重新生成
- `dist/userscript/gemini-watermark-remover.user.js` 已重新生成
- `package.json` 中的 package/sdk 入口仍与实际发布源码布局一致
- 生成后的 userscript 元数据使用当前 `package.json` 版本号
- `release/` 下已重新生成 Chrome 插件 zip、sha256 文件和 `latest-extension.json`，用于 GitHub Release 和手动安装备用入口
- `dist/extension` 下的未打包插件是本地测试版；正式发布 manifest 只写入 `release/` 里的 zip

## 版本元数据

- 提升 `package.json` 版本号
- 保持 `build.js` 中 userscript 的 `@version` 来自 `pkg.version`
- 在 `CHANGELOG.md` 和 `CHANGELOG_zh.md` 中新增对应版本记录

## 人工验证

- 在 Tampermonkey 或 Violentmonkey 中安装或更新生成后的 userscript
- 验证本地安装版本时，针对固定 profile 运行一次 `pnpm probe:tm:freshness`
- 验证 Gemini 页面预览图替换链路正常
- 验证 Gemini 原生复制/下载动作仍返回去水印后的结果
- 验证预览图处理失败时页面原图仍保持可见
- 从 `dist/extension` 加载未打包本地 Chrome 插件，验证弹窗开关、Gemini 在线工具链接、通用去水印链接和 GitHub 反馈链接；确认扩展卡片显示为 `Gemini Watermark Remover Local`
- 确认线上 Chrome Web Store 商店页指向：
  `https://chromewebstore.google.com/detail/gemini-watermark-remover/cjlmnfcfnofnglkphbcdclbpimdjkmdf`
- 如果本次要发布 sdk/package，发包前再做一次 package smoke 检查

## 发布

- 提交版本相关改动
- 创建与版本号一致的 git tag，例如 `v1.0.1`
- 基于该 tag 创建 GitHub Release，并上传 `dist/userscript/gemini-watermark-remover.user.js`
- 上传 `release/gemini-watermark-remover-extension-v<version>.zip`、对应 `.sha256.txt` 和 `latest-extension.json` 到 GitHub Release，作为手动安装备用包
- 将 Chrome 插件包提交到 Chrome Web Store，或确认已审核通过的商店页正在提供目标版本
- 只有本次涉及 package 对外接口时，才同步发布 sdk/package

GitHub Release 命令示例：

```bash
gh release create v<version> \
  dist/userscript/gemini-watermark-remover.user.js \
  release/gemini-watermark-remover-extension-v<version>.zip \
  release/gemini-watermark-remover-extension-v<version>.zip.sha256.txt \
  release/latest-extension.json \
  --repo GargantuaX/gemini-watermark-remover \
  --title "v<version>" \
  --notes "<release notes>" \
  --latest
```

## 官网同步

官网项目位于独立本地目录：`D:\Project\geminiwatermarkremover.io`。

GitHub Release 发布后：

1. 在官网项目中运行 `pnpm run userscript:build`。
   - 该命令会重新构建当前上游仓库。
   - 然后把 `dist/userscript/gemini-watermark-remover.user.js` 复制到 `public/userscript/gemini-watermark-remover.user.js`。
2. 从 GitHub Release 下载准确的 Chrome 插件备用资产到官网项目：
   - `gemini-watermark-remover-extension-v<version>.zip`
   - `gemini-watermark-remover-extension-v<version>.zip.sha256.txt`
   - `latest-extension.json`
3. 将这些文件复制到 `public/downloads/`。
4. 更新 `src/i18n/chrome-extension-content.ts`，确保 Chrome 插件主 CTA 指向 Chrome Web Store，备用包元数据与 `latest-extension.json` 保持一致。
5. 从 `public/downloads/` 删除旧版本 zip 和 checksum 文件。
6. 在官网项目中运行 `pnpm test` 和 `pnpm run build`。
7. 使用 `pnpm run deploy:cf-workers` 部署官网。

`pnpm run deploy:cf-workers` 可能已经成功完成 Cloudflare 部署，但最后报告 Sentry release finalize 错误。如果 Wrangler 打印了当前 version ID，并且线上站点验证通过，应先把官网部署视为已发布，再单独排查 Sentry。

## 发布后检查

- 确认浏览器里已安装的 userscript 显示正确版本号
- 确认 GitHub Release latest userscript 返回最新产物：
  `https://github.com/GargantuaX/gemini-watermark-remover/releases/latest/download/gemini-watermark-remover.user.js`
- 确认官网返回最新 userscript 产物：
  `https://geminiwatermarkremover.io/userscript/gemini-watermark-remover.user.js`
- 确认 Chrome Web Store 商店页可访问：
  `https://chromewebstore.google.com/detail/gemini-watermark-remover/cjlmnfcfnofnglkphbcdclbpimdjkmdf`
- 确认官网 Chrome 插件主 CTA 指向 Chrome Web Store，同时仍提供最新备用 zip，且校验值一致
- 确认 `https://geminiwatermarkremover.io/downloads/latest-extension.json` 返回最新插件版本、文件名、体积和 sha256
- 临时性的验证记录放到 release note 或 PR 里，不继续堆在仓库文档中
