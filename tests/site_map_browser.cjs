/* Run: node tests/site_map_browser.cjs (requires Playwright and Chromium).
 * Starts an ephemeral server at the production /estuary/site-map/ prefix.
 * Optional: SITE_MAP_URL for an existing server; PLAYWRIGHT_CHROMIUM_EXECUTABLE
 * for system Chrome; SITE_MAP_SCREENSHOTS for QA images.
 * Automated tests use a fixture tile, avoiding traffic to the public OSM service.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const http = require('node:http');
let url = process.env.SITE_MAP_URL;
let server;
const features = JSON.parse(fs.readFileSync(path.join(__dirname, '../site-map/estuary-sites.geojson'))).features;
const screenshotDir = process.env.SITE_MAP_SCREENSHOTS;
const normalize = value => value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLocaleLowerCase();

(async () => {
  if (!url) {
    const root = path.resolve(__dirname, '..');
    server = http.createServer((request, response) => {
      let pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      if (!pathname.startsWith('/estuary/')) { response.writeHead(404).end(); return; }
      if (pathname.endsWith('/')) pathname += 'index.html';
      const file = path.resolve(root, pathname.slice('/estuary/'.length));
      if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
      const types = { '.html':'text/html', '.js':'text/javascript', '.css':'text/css', '.geojson':'application/geo+json', '.json':'application/json', '.png':'image/png' };
      fs.readFile(file, (error, data) => {
        if (error) { response.writeHead(404).end(); return; }
        response.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'text/plain' });
        response.end(data);
      });
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    url = `http://127.0.0.1:${server.address().port}/estuary/site-map/`;
  }
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || chromium.executablePath() });
  try {
    const context = await browser.newContext();
    await context.route('https://tile.openstreetmap.org/**', route => route.fulfill({
      status: 200, contentType: 'image/svg+xml',
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="#e1eceb"/></svg>',
    }));
    const page = await context.newPage();
    await page.setViewportSize({ width: 1440, height: 1000 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await page.locator('#status').filter({ hasText: '66 of 66 estuaries' }).waitFor();
    const rendered = await page.locator('.site-button').evaluateAll(buttons => buttons.map(button => ({
      id: Number(button.dataset.siteId), name: button.firstChild.textContent,
    })));
    assert.equal(rendered.length, 66);
    assert.equal(new Set(rendered.map(item => item.id)).size, 66);
    assert.equal(new Set(rendered.map(item => item.name)).size, 66);
    assert.deepEqual(rendered.sort((a,b) => a.id-b.id), features.map(f => ({ id:f.properties.site_id, name:f.properties.site_name })).sort((a,b)=>a.id-b.id));
    assert.equal(await page.locator('.estuary-marker').count(), 66);
    assert.equal(await page.locator('link[rel="canonical"]').getAttribute('href'), 'https://kyle-dorman.github.io/estuary/site-map/');
    for (const href of ['https://doi.org/10.5281/zenodo.20753031', 'https://doi.org/10.31223/X5C228', 'https://github.com/kyle-dorman/estuary']) {
      assert.equal(await page.locator(`nav a[href="${href}"]`).count(), 1);
    }
    for (const href of ['estuary-sites.geojson', 'provenance.json', 'vendor/leaflet/LICENSE']) {
      assert((await page.request.get(new URL(href, url).href)).ok(), href);
    }
    for (const query of ['sAn', 'Péñasquitos', '  RIVER  ']) {
      await page.locator('#search').fill(query);
      const expected = features.filter(f => normalize(f.properties.site_name).includes(normalize(query.trim()))).length;
      assert.equal(await page.locator('#site-list li:not([hidden])').count(), expected);
      assert.equal(await page.locator('.estuary-marker').count(), expected);
    }
    await page.locator('#search').fill('no-such-estuary-123');
    assert.equal(await page.locator('#site-list li:not([hidden])').count(), 0);
    assert.equal(await page.locator('.estuary-marker').count(), 0);
    assert(await page.locator('#empty').isVisible());
    await page.locator('#clear').click();
    assert.equal(await page.locator('#search').inputValue(), '');
    assert.equal(await page.locator('.estuary-marker').count(), 66);
    assert(await page.locator('#empty').isHidden());
    await page.locator('#search').fill('   ');
    assert.equal(await page.locator('#site-list li:not([hidden])').count(), 66);
    await page.locator('#search').fill('Carmel');
    const selected = features.find(f => f.properties.site_name === 'Carmel River');
    await page.locator(`[data-site-id="${selected.id}"]`).click();
    await page.locator('.leaflet-popup-content h3').filter({ hasText: 'Carmel River' }).waitFor();
    assert.equal(await page.locator('.estuary-marker.selected').count(), 1);
    assert.equal(await page.locator('.site-button[aria-pressed="true"]').getAttribute('data-site-id'), String(selected.id));
    assert.equal(await page.evaluate(() => map.getZoom()), 12);
    const popup = await page.locator('.leaflet-popup-content').innerText();
    for (const value of [String(selected.id), selected.properties.pmep_region, selected.properties.cmecs_class, 'WGS 84']) assert(popup.includes(value), value);
    await page.locator('#show-all').click();
    assert.equal(await page.locator('#search').inputValue(), '');
    assert.equal(await page.locator('.estuary-marker.selected').count(), 0);
    await page.locator('.leaflet-popup').waitFor({ state: 'detached' });
    assert((await page.evaluate(() => map.getZoom())) < 12);
    await page.locator('#search').fill('Carmel');
    await page.locator(`[data-site-id="${selected.id}"]`).focus();
    await page.keyboard.press('Enter');
    await page.locator('.leaflet-popup-content h3').filter({ hasText: 'Carmel River' }).waitFor();
    await page.locator('#show-all').click();
    await page.locator('.leaflet-popup').waitFor({ state: 'detached' });
    await page.locator('#search').fill('Russian');
    await page.locator('.estuary-marker').click();
    await page.locator('.leaflet-popup-content h3').filter({ hasText: 'Russian River' }).waitFor();
    assert.equal(await page.locator('.site-button[aria-pressed="true"]').count(), 1);
    await page.locator('#show-all').click();
    await page.locator('.leaflet-popup').waitFor({ state: 'detached' });
    await page.locator('#search').fill('Russian');
    await page.locator('.estuary-marker').focus();
    await page.keyboard.press('Enter');
    await page.locator('.leaflet-popup-content h3').filter({ hasText: 'Russian River' }).waitFor();
    assert.equal(await page.locator('.site-button[aria-pressed="true"]').count(), 1);
    assert.equal(await page.evaluate(() => map.getZoom()), 12);
    await page.locator('#search').fill('Carmel');
    assert.equal(await page.locator('.site-button[aria-pressed="true"]').count(), 0);
    assert.equal(await page.locator('.estuary-marker.selected').count(), 0);
    await page.locator('.leaflet-popup').waitFor({ state: 'detached' });
    await page.locator('#show-all').click();
    if (screenshotDir) {
      fs.mkdirSync(screenshotDir, { recursive: true });
      await page.screenshot({ path: path.join(screenshotDir, 'desktop.png'), fullPage: true });
    }
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('#search').fill('Carmel');
      await page.locator(`[data-site-id="${selected.id}"]`).click();
      await page.locator('.leaflet-popup-content h3').filter({ hasText: 'Carmel River' }).waitFor();
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `overflow at ${width}px`);
      assert(await page.locator('.leaflet-popup-content').isVisible());
      await page.waitForFunction(() => getComputedStyle(document.querySelector('.leaflet-popup')).opacity === '1');
      const bounds = await page.locator('.leaflet-popup').boundingBox();
      const mapBounds = await page.locator('#map').boundingBox();
      assert(bounds.x >= mapBounds.x && bounds.x + bounds.width <= mapBounds.x + mapBounds.width, `popup clipped at ${width}px: ${JSON.stringify(bounds)}`);
      const controls = await page.locator('.leaflet-control-zoom').boundingBox();
      const overlaps = bounds.x < controls.x + controls.width && bounds.x + bounds.width > controls.x && bounds.y < controls.y + controls.height && bounds.y + bounds.height > controls.y;
      assert(!overlaps, `popup overlaps zoom controls at ${width}px`);
      if (screenshotDir) await page.screenshot({ path: path.join(screenshotDir, `mobile-${width}.png`), fullPage: true });
      await page.locator('#show-all').click();
    }
    assert.deepEqual(errors, []);
    await page.close();
    const tileFailure = await context.newPage();
    await tileFailure.route('https://tile.openstreetmap.org/**', route => route.abort());
    await tileFailure.goto(url);
    await tileFailure.locator('#map-notice').filter({ hasText: 'Background tiles could not load' }).waitFor();
    await tileFailure.locator('#search').fill('Carmel');
    assert.equal(await tileFailure.locator('.estuary-marker').count(), 1);
    await tileFailure.close();
    const leafletFailure = await context.newPage();
    await leafletFailure.route('**/vendor/leaflet/leaflet.js', route => route.abort());
    await leafletFailure.goto(url);
    await leafletFailure.locator('#status').filter({ hasText: '66 of 66 estuaries' }).waitFor();
    await leafletFailure.locator('#map-notice').filter({ hasText: 'The map could not load' }).waitFor();
    await leafletFailure.locator('#search').fill('Carmel');
    assert.equal(await leafletFailure.locator('#site-list li:not([hidden])').count(), 1);
    await leafletFailure.locator(`[data-site-id="${selected.id}"]`).click();
    assert.equal(await leafletFailure.locator('.site-button[aria-pressed="true"]').count(), 1);
    await leafletFailure.close();
    const dataFailure = await context.newPage();
    await dataFailure.route('**/estuary-sites.geojson', route => route.fulfill({ status: 500, body: 'unavailable' }));
    await dataFailure.goto(url);
    await dataFailure.locator('#status').filter({ hasText: 'Site data could not load' }).waitFor();
    assert(await dataFailure.locator('#search').isDisabled());
    assert(await dataFailure.locator('#clear').isDisabled());
    assert(await dataFailure.locator('#show-all').isDisabled());
    await dataFailure.close();
    console.log('PASS: 66 unique sites, search, selection, popups, links, 390/320px layouts, and tile/Leaflet/data failures.');
  } finally { await browser.close(); if (server) await new Promise(resolve => server.close(resolve)); }
})().catch(error => { console.error(error); if (server) server.close(); process.exitCode = 1; });
