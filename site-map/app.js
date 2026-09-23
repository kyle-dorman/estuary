/* Leaflet is vendored locally; only the basemap requires a third-party service. */
"use strict";
const search = document.querySelector("#search");
const list = document.querySelector("#site-list");
const status = document.querySelector("#status");
const notice = document.querySelector("#map-notice");
let map;
let entries = [];
let selected = null;

function normalize(value) {
  return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase();
}

function icon(active) {
  const size = active ? 22 : 14;
  return L.divIcon({className: `estuary-marker${active ? " selected" : ""}`,
    iconSize: [size, size], iconAnchor: [size / 2, size / 2]});
}

function popup(feature) {
  const p = feature.properties;
  const box = document.createElement("div");
  const heading = document.createElement("h3");
  heading.textContent = p.site_name;
  box.append(heading);
  const fields = [["Site ID", p.site_id], ["Region", p.pmep_region],
    ["Classification", p.cmecs_class],
    ["Estuary area (ha)", p.estuary_area_ha == null ? null :
      p.estuary_area_ha.toLocaleString(undefined, {maximumFractionDigits: 2})]];
  for (const [label, value] of fields) {
    if (value !== null && value !== undefined && value !== "") {
      const line = document.createElement("p");
      line.textContent = `${label}: ${value}`;
      box.append(line);
    }
  }
  const [lon, lat] = feature.geometry.coordinates;
  const coords = document.createElement("p");
  coords.textContent = `Mouth: ${lat.toFixed(5)}° latitude, ${lon.toFixed(5)}° longitude (WGS 84)`;
  const link = document.createElement("a");
  link.href = "https://doi.org/10.5281/zenodo.20753031";
  link.textContent = "View archived dataset";
  box.append(coords, link);
  return box;
}

function select(entry, fromList = false) {
  if (selected) {
    selected.button.setAttribute("aria-pressed", "false");
    selected.marker?.setIcon(icon(false)).setZIndexOffset(0);
  }
  selected = entry;
  entry.button.setAttribute("aria-pressed", "true");
  if (map) {
    entry.marker.setIcon(icon(true)).setZIndexOffset(1000);
    map.setView(entry.marker.getLatLng(), 12);
    entry.marker.openPopup();
  }
  if (!fromList) entry.button.scrollIntoView({block: "nearest"});
  if (fromList && matchMedia("(max-width: 720px)").matches) {
    document.querySelector(".map-panel").scrollIntoView({block: "nearest"});
  }
}

function filter() {
  const query = normalize(search.value.trim());
  let count = 0;
  for (const entry of entries) {
    const visible = normalize(entry.feature.properties.site_name).includes(query);
    entry.item.hidden = !visible;
    count += Number(visible);
    if (map) {
      if (visible && !map.hasLayer(entry.marker)) entry.marker.addTo(map);
      if (!visible && map.hasLayer(entry.marker)) map.removeLayer(entry.marker);
    }
    if (!visible && selected === entry) {
      entry.button.setAttribute("aria-pressed", "false");
      entry.marker?.setIcon(icon(false)).setZIndexOffset(0);
      selected = null;
    }
  }
  status.textContent = `${count} of ${entries.length} estuaries`;
  document.querySelector("#empty").hidden = count !== 0;
}

async function start() {
  if (typeof L !== "undefined") {
    map = L.map("map", {scrollWheelZoom: false, zoomControl: false});
    L.control.zoom({position: "bottomleft"}).addTo(map);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
    }).on("tileerror", () => {
      notice.textContent = "Background tiles could not load. Site markers and search remain available.";
      notice.hidden = false;
    }).addTo(map);
  } else {
    notice.textContent = "The map could not load. You can still search the site list or download the GeoJSON below.";
    notice.hidden = false;
  }
  try {
    const response = await fetch("estuary-sites.geojson");
    if (!response.ok) throw new Error(`Site data: HTTP ${response.status}`);
    const data = await response.json();
    if (data.type !== "FeatureCollection" || !data.features?.length) throw new Error("Invalid site data");
    entries = data.features.sort((a, b) => a.properties.site_name.localeCompare(b.properties.site_name))
      .map(feature => {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "site-button";
        button.dataset.siteId = feature.properties.site_id;
        button.setAttribute("aria-pressed", "false");
        button.textContent = feature.properties.site_name;
        const detail = document.createElement("span");
        detail.textContent = `Site ${feature.properties.site_id}`;
        button.append(detail);
        item.append(button);
        list.append(item);
        const entry = {feature, button, item};
        if (map) {
          const [lon, lat] = feature.geometry.coordinates;
          entry.marker = L.marker([lat, lon], {icon: icon(false), title: feature.properties.site_name,
            alt: feature.properties.site_name}).bindPopup(popup(feature));
          entry.marker.on("click", () => select(entry));
          entry.marker.on("popupopen", () => {
            // Leaflet also opens popups when a focused marker is activated by keyboard.
            if (selected !== entry) select(entry);
          });
        }
        button.addEventListener("click", () => select(entry, true));
        return entry;
      });
    function showAll() {
      search.value = "";
      filter();
      if (selected) {
        selected.button.setAttribute("aria-pressed", "false");
        selected.marker?.setIcon(icon(false)).setZIndexOffset(0);
        selected = null;
      }
      if (map) {
        map.closePopup();
        map.fitBounds(L.latLngBounds(entries.map(e => e.marker.getLatLng())), {padding: [24, 24]});
      }
    }
    search.addEventListener("input", filter);
    document.querySelector("#clear").addEventListener("click", () => { showAll(); search.focus(); });
    document.querySelector("#show-all").addEventListener("click", showAll);
    showAll();
  } catch (error) {
    status.textContent = "Site data could not load. Reload this page or use the Zenodo dataset link.";
    search.disabled = true;
    document.querySelector("#clear").disabled = true;
    document.querySelector("#show-all").disabled = true;
    console.error(error);
  }
}
start();
