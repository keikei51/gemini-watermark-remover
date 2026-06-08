# Release Checklist

## Scope

This repository currently ships four release surfaces:

- website build in `dist/`
- userscript bundle in `dist/userscript/gemini-watermark-remover.user.js`
- package/sdk source and metadata from `package.json`, `src/core/`, and `src/sdk/`
- Chrome Web Store listing plus fallback package in `release/gemini-watermark-remover-extension-v<version>.zip`

## Preflight

Run these locally from the repo root:

```bash
pnpm install
pnpm test
pnpm build
pnpm package:extension
```

Expected result:

- all tests pass
- website artifacts in `dist/` are regenerated for the current build
- `dist/userscript/gemini-watermark-remover.user.js` is regenerated
- package/sdk entrypoints in `package.json` still match the published source layout
- generated userscript metadata uses the current `package.json` version
- Chrome extension release zip, sha256 file, and `latest-extension.json` are regenerated in `release/` for GitHub Release and manual fallback installs
- the unpacked extension in `dist/extension` is a local test build; the official release manifest is written only into the zip in `release/`

## Release Metadata

- bump `package.json` version
- keep `build.js` userscript `@version` sourced from `pkg.version`
- add dated entries to `CHANGELOG.md` and `CHANGELOG_zh.md`

## Manual Verification

- install or update the generated userscript in Tampermonkey/Violentmonkey
- run `pnpm probe:tm:freshness` against the fixed profile when validating the local install
- verify Gemini page preview replacement works
- verify native Gemini copy/download still returns processed output
- verify preview processing failure leaves the original page image visible
- load the unpacked local Chrome extension from `dist/extension` and verify the popup toggle, Gemini online link, general watermark link, and GitHub feedback link; confirm the extension card is labeled `Gemini Watermark Remover Local`
- verify the live Chrome Web Store listing points to:
  `https://chromewebstore.google.com/detail/gemini-watermark-remover/cjlmnfcfnofnglkphbcdclbpimdjkmdf`
- if you publish the sdk surface, run a final package smoke check before uploading

## Publish

- commit release changes
- create a git tag matching the package version, for example `v1.0.1`
- create a GitHub Release from that tag and upload the built userscript from `dist/userscript/gemini-watermark-remover.user.js`
- upload `release/gemini-watermark-remover-extension-v<version>.zip`, its `.sha256.txt` file, and `latest-extension.json` to GitHub Release as the manual fallback package
- submit the Chrome extension package to Chrome Web Store, or confirm the already-approved listing is serving the intended version
- publish the sdk package only if this release includes package-facing changes

Example GitHub Release command:

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

## Official Website Sync

The public website is maintained in the separate local project `D:\Project\geminiwatermarkremover.io`.

After the GitHub Release is published:

1. Run `pnpm run userscript:build` in the website project.
   - This rebuilds this upstream repository.
   - It copies `dist/userscript/gemini-watermark-remover.user.js` to `public/userscript/gemini-watermark-remover.user.js`.
2. Download the exact Chrome extension fallback assets from the GitHub Release into the website project:
   - `gemini-watermark-remover-extension-v<version>.zip`
   - `gemini-watermark-remover-extension-v<version>.zip.sha256.txt`
   - `latest-extension.json`
3. Copy those files to `public/downloads/`.
4. Update `src/i18n/chrome-extension-content.ts` to keep the primary Chrome extension CTA pointed at the Chrome Web Store and the fallback package metadata matched to `latest-extension.json`.
5. Remove stale older extension zip and checksum files from `public/downloads/`.
6. Run `pnpm test` and `pnpm run build` in the website project.
7. Deploy with `pnpm run deploy:cf-workers`.

`pnpm run deploy:cf-workers` may finish the Cloudflare deployment successfully and then report a Sentry release finalization error. If Wrangler prints a current version ID and the live site verifies correctly, treat the website deployment as published, then investigate Sentry separately.

## Post-Release

- confirm the installed userscript reports the expected version
- confirm the GitHub Release latest userscript serves the latest bundle:
  `https://github.com/GargantuaX/gemini-watermark-remover/releases/latest/download/gemini-watermark-remover.user.js`
- confirm the official website serves the latest userscript bundle:
  `https://geminiwatermarkremover.io/userscript/gemini-watermark-remover.user.js`
- confirm the Chrome Web Store listing is reachable:
  `https://chromewebstore.google.com/detail/gemini-watermark-remover/cjlmnfcfnofnglkphbcdclbpimdjkmdf`
- confirm the official website points the primary Chrome extension CTA to Chrome Web Store and still serves the latest fallback zip with matching checksum
- confirm `https://geminiwatermarkremover.io/downloads/latest-extension.json` reports the latest extension version, file, size, and sha256
- keep any ad hoc verification notes in the release PR or tag notes, not in source docs
