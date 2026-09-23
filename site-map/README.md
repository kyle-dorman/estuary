# California estuary site map

A static, searchable discovery interface for the paper **Seasonal and interannual
dynamics of estuary connectivity along the California coast**. The authoritative
archived dataset remains [Zenodo record 20753031](https://doi.org/10.5281/zenodo.20753031),
version **1.0.0**, covering 2017–2025. The map does not report current mouth states.

Expected public URL: https://kyle-dorman.github.io/estuary/site-map/

## Data source and reproduction

The authoritative input is `shared_data/sites.geojson` inside
`california_estuary_connectivity_shared_data_v1.0.0.zip` from that Zenodo record.
The archived file is byte-identical to the local public export
`/Volumes/x10pro/estuary/shared_data/sites.geojson` at implementation time.
No raw working metadata is distributed by the map.

The repository's `scripts/share_data/prepare_shared_data.py`, function `clean_sites`,
produces that export from `ca_data_w_empa_pmep_usgs.geojson`: it selects `skipped == False`,
retains `Site code` as `site_id`, takes `Site name` with `Estuary_Name` fallback,
and constructs WGS 84 points from `point_geom_long` and `point_geom_lat`.
Use the archived export to reproduce this map, rather than rerunning the full release
pipeline against mutable working data.

`scripts/site_map_data.py` verifies the pinned archive SHA-256 and source checksum,
copies all 66 sites and their exact coordinates, sorts by stable integer site ID,
sets GeoJSON `Feature.id` to that ID, and selects only these already-public fields:

- `site_id`, `site_name`
- `pmep_region`, `cmecs_class`, `estuary_area_ha` (nullable)

No names or coordinates were manually retyped. GeoJSON coordinates are
`[longitude, latitude]` in decimal degrees, WGS 84 / EPSG:4326 with CRS84 axis order.
They represent study mouth locations, not boundaries or precise current mouth positions.
Leaflet receives `[latitude, longitude]` at display time. Popup coordinates are rounded
for readability; the GeoJSON retains the original precision. Area is from the released
PMEP metadata and is not calculated from the point geometry.

Download the archive without modifying the release files:

```sh
curl -fL 'https://zenodo.org/records/20753031/files/california_estuary_connectivity_shared_data_v1.0.0.zip?download=1' -o /tmp/estuary-v1.0.0.zip
python3 scripts/site_map_data.py --source-zip /tmp/estuary-v1.0.0.zip
python3 scripts/site_map_data.py --check
```

Generation needs only Python's standard library. `provenance.json` records the archive,
source member, SHA-256 checksums, archived MD5, exact inventory, fields, and license.
The archive MD5 was independently matched to the live Zenodo API on 2026-09-23.
Attribution follows the three creators in that live record (its packaged README lists
only two). Regeneration never edits or extracts over the release package.

## Preview and checks

From the repository root:

```sh
python3 scripts/site_map_data.py --check
python3 -m unittest discover -s tests -p 'test_site_map_data.py'
node --check site-map/app.js
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://127.0.0.1:8000/site-map/ (fetching GeoJSON requires HTTP, not `file://`).
The browser regression script is `tests/site_map_browser.cjs`; see its header for
invocation. Playwright is only a development/testing dependency. The production site
has no package install, build step, backend, database, analytics, or API key.

To install the isolated browser test dependency and run the self-hosted test:

```sh
npm install --prefix /tmp/estuary-browser-tests --no-package-lock playwright@1.62.1
/tmp/estuary-browser-tests/node_modules/.bin/playwright install chromium
NODE_PATH=/tmp/estuary-browser-tests/node_modules node tests/site_map_browser.cjs
```

Verification on 2026-09-23 passed: seven data unit tests, exact comparison with all
66 archived features, targeted Ruff and JavaScript syntax checks, and the browser
suite against both the repository and the packaged artifact at `/estuary/site-map/`.
Desktop, 390px, and 320px screenshots were inspected. Mouse/keyboard selection,
popup bounds, case/accent search, empty results, resets, and tile/library/data failure
states were checked. All three study resource links returned HTTP 200. The public
map URL returned 404, as expected before deployment. Repository-wide non-mutating
Ruff reported 332 existing findings outside this change; no unrelated files were fixed.

Search is case- and accent-insensitive; filtering updates both the list and markers.
Blank search restores all sites; an unmatched query shows an explicit empty state.
Selecting a list item or marker highlights both and opens a popup. Clear/Show all
resets the selection and fits the complete inventory. The list supports keyboard use.

Repository lint uses Ruff. For this change, run its non-mutating checks rather than
the repo-wide `lint.sh` fixer:

```sh
.venv/bin/ruff check scripts/site_map_data.py tests/test_site_map_data.py
.venv/bin/ruff format --check scripts/site_map_data.py tests/test_site_map_data.py
```

## Publication (owner action required)

No push, Pages setting change, or remote deployment was performed for this implementation.
GitHub's public repository API reported `has_pages: false` and default branch `main`
on 2026-09-23; the expected URL was not yet live.

1. Review and commit the map, generator, tests, README change, and
   `.github/workflows/site-map.yml`; push the commit to `main` (or merge a reviewed PR).
2. In [repository Settings → Pages](https://github.com/kyle-dorman/estuary/settings/pages),
   set **Build and deployment → Source → GitHub Actions**. Ensure Actions are enabled.
3. Run **Actions → Estuary site map → Run workflow → main**. If the push happened before
   Pages was enabled, rerun the workflow after enabling it. Approve the `github-pages`
   environment deployment if repository rules require it.
4. Verify the deployed URL above, its search and links. Only then add the Zenodo link below.

The workflow validates the data, runs the browser regression suite, and packages an explicit allowlist into
`_site/site-map/`. It publishes neither the whole repository nor the release archive.
Pull requests run validation and packaging without deployment; pushes affecting the map
on `main`, and manual runs on `main`, publish after successful checks.
See [GitHub's Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Zenodo metadata to add after publication

Append this paragraph to the existing description (preserve the rest):

> Explore the 66 study estuaries in the searchable site map:
> https://kyle-dorman.github.io/estuary/site-map/. The map provides site locations and
> selected metadata from dataset version 1.0.0 as a discovery interface; this Zenodo
> record remains the authoritative archived dataset.

Add a related identifier with these exact values:

| Field | Value |
| --- | --- |
| Identifier | `https://kyle-dorman.github.io/estuary/site-map/` |
| Scheme | URL (`url`) |
| Relation, from dataset to map | Is referenced by (`isReferencedBy`) |
| Resource type | Other (`other`) |

This expresses that the map references the dataset. Preserve the current preprint
relationship (`isSupplementTo`, DOI `10.31223/X5C228`, `publication-preprint`) and repository
link. Do not replace the dataset DOI or add the map as an alternate identifier for it.
These are manual instructions only; no Zenodo metadata or files have been changed.
Relation values follow the [Zenodo API vocabulary](https://developers.zenodo.org/).

## Licensing and external services

- Released site data: CC BY 4.0; credit Kyle Dorman, John L. Largier, and Kyle C. Cavanaugh
  and the dataset DOI. The map changes the metadata selection/serialization, not locations.
- Site code: repository MIT license (`../LICENSE.txt`).
- Leaflet 1.9.4 is vendored from `https://unpkg.com/leaflet@1.9.4/dist/`;
  its BSD-2-Clause license is retained at `vendor/leaflet/LICENSE`.
- Basemap tiles/data: © OpenStreetMap contributors, with visible map and footer links.
  Ordinary browser caching and referrers are preserved; there is no tile prefetch or
  offline tile download. Tiles need network access and are a best-effort third-party
  service. Follow the [OSM tile policy](https://operations.osmfoundation.org/policies/tiles/).
  If tiles fail, a notice appears and the site markers/list remain usable. If the map
  library fails, search/list still work; if data fails, the page offers the archive link.
