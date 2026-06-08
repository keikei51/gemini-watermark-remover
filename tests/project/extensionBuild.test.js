import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';

async function readText(relativePath) {
  return readFile(new URL(`../../${relativePath}`, import.meta.url), 'utf8');
}

async function readBinary(relativePath) {
  return readFile(new URL(`../../${relativePath}`, import.meta.url));
}

test('production build should emit a MV3 extension that packages the shared userscript runtime', async () => {
  const build = spawnSync('pnpm', ['build'], {
    cwd: new URL('../..', import.meta.url),
    encoding: 'utf8',
    shell: process.platform === 'win32'
  });

  assert.equal(build.status, 0, build.stderr || build.stdout);

  const manifest = JSON.parse(await readText('dist/extension/manifest.json'));
  assert.equal(manifest.manifest_version, 3);
  assert.equal(manifest.name, 'Gemini Watermark Remover Local');
  assert.equal(manifest.short_name, 'GWR Local');
  assert.equal(manifest.action.default_title, 'Gemini Watermark Remover Local');
  assert.match(manifest.version_name, new RegExp(`^${manifest.version.replaceAll('.', '\\.')}\\+local\\.`));
  assert.match(manifest.description, /local test build/);
  assert.equal(manifest.icons['16'], 'assets/icon-16.png');
  assert.equal(manifest.icons['32'], 'assets/icon-32.png');
  assert.equal(manifest.icons['48'], 'assets/icon-48.png');
  assert.equal(manifest.icons['128'], 'assets/icon-128.png');
  assert.equal(manifest.action.default_icon['16'], 'assets/icon-16.png');
  assert.equal(manifest.action.default_icon['32'], 'assets/icon-32.png');
  assert.equal(manifest.action.default_icon['48'], 'assets/icon-48.png');
  assert.equal(manifest.action.default_icon['128'], 'assets/icon-128.png');
  assert.ok(manifest.permissions.includes('storage'));
  assert.ok(manifest.permissions.includes('activeTab'));
  assert.deepEqual(
    manifest.content_scripts.map((script) => script.world || 'ISOLATED'),
    ['MAIN', 'ISOLATED']
  );
  assert.equal(manifest.background.service_worker, 'service-worker.js');
  assert.equal(manifest.action.default_popup, 'popup.html');
  assert.ok(manifest.host_permissions.includes('https://*.googleusercontent.com/*'));

  assert.equal(existsSync(new URL('../../dist/extension/content-main.js', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/isolated-bridge.js', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/service-worker.js', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/popup.html', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/popup.css', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/popup.js', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/assets/icon-16.png', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/assets/icon-32.png', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/assets/icon-48.png', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/assets/icon-128.png', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/assets/logo-shape.svg', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension/assets/github.svg', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../dist/extension-local', import.meta.url)), false);

  const contentMain = await readText('dist/extension/content-main.js');
  assert.match(contentMain, /Gemini Watermark Remover/);
  assert.match(contentMain, /GM_xmlhttpRequest/);

  const popupHtml = await readText('dist/extension/popup.html');
  assert.match(popupHtml, /<html lang="en">/);
  assert.match(popupHtml, /https:\/\/geminiwatermarkremover\.io\//);
  assert.match(popupHtml, /https:\/\/pilio\.ai\/image-watermark-remover/);
  assert.match(popupHtml, /Online Gemini watermark remover/);
  assert.match(popupHtml, /Remove any image watermark/);
  assert.match(popupHtml, /Report an issue on GitHub/);
  assert.match(popupHtml, /https:\/\/github\.com\/GargantuaX\/gemini-watermark-remover\/issues/);
  assert.match(popupHtml, /assets\/github\.svg/);
  assert.match(popupHtml, /assets\/logo-shape\.svg/);
  assert.match(popupHtml, /enable-toggle/);
  assert.match(popupHtml, /popup\.js/);

  const popupScript = await readText('dist/extension/popup.js');
  assert.match(popupScript, /globalThis\.chrome/);
  assert.match(popupScript, /storage\?\.local/);
  assert.match(popupScript, /tabs\?\.reload/);
  assert.match(popupScript, /gwrEnabled/);
});

test('extension package should replace the local debug manifest with the official release manifest', async () => {
  const packageBuild = spawnSync('pnpm', ['package:extension'], {
    cwd: new URL('../..', import.meta.url),
    encoding: 'utf8',
    shell: process.platform === 'win32'
  });
  assert.equal(packageBuild.status, 0, packageBuild.stderr || packageBuild.stdout);

  const latest = JSON.parse(await readText('release/latest-extension.json'));
  assert.equal(latest.name, 'gemini-watermark-remover-extension');
  assert.equal(latest.source, 'dist/extension');
  assert.equal(existsSync(new URL('../../release/latest-extension-local.json', import.meta.url)), false);

  const zipText = (await readBinary(`release/${latest.file}`)).toString('utf8');
  assert.match(zipText, /"name": "Gemini Watermark Remover"/);
  assert.match(zipText, /"short_name": "GWR"/);
  assert.match(zipText, /"default_title": "Gemini Watermark Remover"/);
  assert.doesNotMatch(zipText, /Gemini Watermark Remover Local/);
  assert.doesNotMatch(zipText, /local test build/);
  assert.doesNotMatch(zipText, /version_name/);
});

test('production build should preserve packaged extension release artifacts', async () => {
  const packageBuild = spawnSync('pnpm', ['package:extension'], {
    cwd: new URL('../..', import.meta.url),
    encoding: 'utf8',
    shell: process.platform === 'win32'
  });
  assert.equal(packageBuild.status, 0, packageBuild.stderr || packageBuild.stdout);

  const latestBefore = await readText('release/latest-extension.json');
  const latest = JSON.parse(latestBefore);

  const build = spawnSync('pnpm', ['build'], {
    cwd: new URL('../..', import.meta.url),
    encoding: 'utf8',
    shell: process.platform === 'win32'
  });
  assert.equal(build.status, 0, build.stderr || build.stdout);

  assert.equal(existsSync(new URL(`../../release/${latest.file}`, import.meta.url)), true);
  assert.equal(existsSync(new URL(`../../release/${latest.file}.sha256.txt`, import.meta.url)), true);
  assert.equal(existsSync(new URL('../../release/latest-extension.json', import.meta.url)), true);
  assert.equal(existsSync(new URL('../../release/latest-extension-local.json', import.meta.url)), false);
  assert.equal(await readText('release/latest-extension.json'), latestBefore);
});
