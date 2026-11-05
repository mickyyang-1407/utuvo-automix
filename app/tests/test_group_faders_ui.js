/*
 * R118: the "almost there?" faders, run from the page's own code.
 *
 * The fader block of web/index.html (between its marker comment and
 * `let mixedKey`) is taken out of the page and run against a small fake DOM,
 * so what is tested is the code the App ships:
 *   - a result from before the faders (no "groups") gets no fader panel;
 *   - only the groups the mix has get a fader, plus FX & space;
 *   - the faders on screen are what the render applied, clamped to the
 *     screen's range (a wider value is shown and sent as the nearest end);
 *   - moving one enables "Apply", resetting disables it again;
 *   - what is sent is what is shown, 0 dB faders left out.
 */
"use strict";
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const PAGE = path.join(__dirname, "..", "web", "index.html");
const html = fs.readFileSync(PAGE, "utf8");
const start = html.indexOf("/* ------------------------------------------------------- R118 group faders */");
const end = html.indexOf("let mixedKey = null;", start);
if (start < 0 || end < 0) throw new Error("the fader block is not in the page");
const block = html.slice(start, end);

function fakeElement(tag = "div") {
  const classes = new Set();
  const listeners = {};
  const el = {
    tagName: tag.toUpperCase(), children: [], dataset: {}, attrs: {}, value: "", textContent: "",
    disabled: false, type: "", min: "", max: "", step: "", className: "",
    classList: {
      add: (...c) => c.forEach(x => classes.add(x)), remove: (...c) => c.forEach(x => classes.delete(x)),
      contains: c => classes.has(c), toggle: (c, on) => { if (on === undefined) on = !classes.has(c); if (on) classes.add(c); else classes.delete(c); },
    },
    style: {},
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return this.attrs[k]; },
    addEventListener(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
    fire(ev) { (listeners[ev] || []).forEach(fn => fn({})); },
    append(...kids) { this.children.push(...kids); },
    prepend(...kids) { this.children.unshift(...kids); },
    appendChild(k) { this.children.push(k); return k; },
    replaceChildren(...kids) { this.children = kids; },
    querySelector(sel) { return find(this, sel)[0] || null; },
    querySelectorAll(sel) { return find(this, sel); },
  };
  return el;
}
function matches(el, sel) {
  if (sel.startsWith(".")) return (el.className || "").split(/\s+/).includes(sel.slice(1));
  return el.tagName === sel.toUpperCase();
}
function find(root, sel) {
  const out = [];
  (function walk(n) { for (const c of n.children || []) { if (matches(c, sel)) out.push(c); walk(c); } })(root);
  return out;
}

function harness(lang = "en") {
  const els = {};
  const $ = id => (els[id] = els[id] || fakeElement());
  const ctx = {
    $, L: (zh, en) => (lang === "zh" ? zh : en), stems: ["a.wav"],
    renderOverview: () => {}, Object, Number, Math, Array, Set, String,
    document: { createElement: tag => fakeElement(tag) },
  };
  vm.createContext(ctx);
  vm.runInContext(block + "\nthis.api = {renderFaders, faderValues, setFaders, get faders() { return faders; }, get applied() { return appliedFaders; }};", ctx);
  return { api: ctx.api, $ };
}
// objects made inside the page's context have that context's prototypes
const plain = o => JSON.parse(JSON.stringify(o));
const rows = h => h.$("faders").children;
const groupsShown = h => rows(h).map(r => r.dataset.group);

// a result from before R118: no groups -> no panel
{
  const h = harness();
  h.api.renderFaders({ tracks: [] });
  assert.ok(h.$("tune").classList.contains("hide"), "an older result must not get faders");
}

// only the groups the mix has, plus FX & space; the applied values on screen
{
  const h = harness();
  h.api.renderFaders({ groups: ["vocal", "music"], group_trims: { vocal: 2, fx: -1.5 } });
  assert.ok(!h.$("tune").classList.contains("hide"));
  assert.deepStrictEqual(groupsShown(h), ["vocal", "music", "fx"]);
  const vocal = rows(h)[0];
  assert.strictEqual(vocal.querySelector("input").value, "2");
  assert.strictEqual(vocal.querySelector(".fv").textContent, "+2.0 dB");
  assert.strictEqual(vocal.querySelector("input").getAttribute("aria-valuetext"), "+2.0 dB");
  assert.strictEqual(rows(h)[2].querySelector(".fv").textContent, "−1.5 dB");
  // nothing moved since the render: nothing to apply
  assert.strictEqual(h.$("tune-apply").disabled, true);
  assert.deepStrictEqual(plain(h.api.faderValues()), { vocal: 2, fx: -1.5 });
}

// a value wider than the screen (an API client may send 12 dB) is clamped
{
  const h = harness();
  h.api.renderFaders({ groups: ["vocal"], group_trims: { vocal: 9 } });
  assert.strictEqual(rows(h)[0].querySelector(".fv").textContent, "+6.0 dB");
  assert.deepStrictEqual(plain(h.api.faderValues()), { vocal: 6 });
}

// moving a fader enables Apply and says it is not applied yet; reset clears it
{
  const h = harness();
  h.api.renderFaders({ groups: ["drum", "bass"], group_trims: {} });
  const drum = rows(h)[0].querySelector("input");
  drum.value = "-2.5";
  drum.fire("input");
  assert.strictEqual(h.$("tune-apply").disabled, false);
  assert.strictEqual(h.$("tune-note").textContent, "Not applied yet");
  assert.deepStrictEqual(plain(h.api.faderValues()), { drum: -2.5 });
  h.api.setFaders({});
  assert.strictEqual(h.$("tune-apply").disabled, true);
  assert.deepStrictEqual(plain(h.api.faderValues()), {});
  // a double click puts one fader back at 0
  drum.value = "3"; drum.fire("input"); drum.fire("dblclick");
  assert.deepStrictEqual(plain(h.api.faderValues()), {});
}

// the Chinese names are the stem families' names
{
  const h = harness("zh");
  h.api.renderFaders({ groups: ["vocal", "drum", "bass", "music"], group_trims: {} });
  const names = rows(h).map(r => r.querySelector(".fn").textContent);
  assert.deepStrictEqual(names, ["人聲", "鼓", "貝斯", "樂器", "效果與空間"]);
}
console.log("group faders ui: ok");
