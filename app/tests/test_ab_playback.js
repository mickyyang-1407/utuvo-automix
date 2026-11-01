#!/usr/bin/env node
/*
 * The A/B switch must not restart the music.
 *
 * The code that does the switching lives in the page itself
 * (app/web/index.html), so this harness takes that code out of the page and
 * runs it against a media element that behaves like a browser's: replacing
 * the source makes the element forget where it was and stop, and the
 * metadata arrives later.  Nothing here re-implements the behaviour -- the
 * assertions are about what the page's own code does.
 *
 *   node tests/test_ab_playback.js
 */

"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const PAGE = path.join(__dirname, "..", "web", "index.html");
const START = "// ---------------------------------------------------------- the A/B switch";
const END = "// ---------------------------------------------------------- end of the A/B switch";

function extractBlock() {
  const html = fs.readFileSync(PAGE, "utf8");
  const from = html.indexOf(START);
  const to = html.indexOf(END);
  if (from < 0) throw new Error("the A/B block start marker is missing in " + PAGE);
  if (to < 0 || to < from) throw new Error("the A/B block end marker is missing in " + PAGE);
  const block = html.slice(from + START.length, to);
  if (!/function setAB\(/.test(block)) throw new Error("setAB is not in the A/B block");
  if (!/function switchSource\(/.test(block)) throw new Error("switchSource is not in the A/B block");
  return block;
}

function fakeElement(id) {
  const listeners = {};
  const classes = new Set();
  return {
    id,
    disabled: false,
    title: "",
    textContent: "",
    classList: {
      toggle(c, on) {
        const want = on === undefined ? !classes.has(c) : !!on;
        if (want) classes.add(c); else classes.delete(c);
      },
      add(c) { classes.add(c); },
      remove(c) { classes.delete(c); },
      contains(c) { return classes.has(c); },
    },
    addEventListener(type, fn) {
      (listeners[type] = listeners[type] || []).push(fn);
    },
    removeEventListener(type, fn) {
      const list = listeners[type] || [];
      const i = list.indexOf(fn);
      if (i >= 0) list.splice(i, 1);
    },
    dispatch(type, ev) {
      (listeners[type] || []).slice().forEach((fn) =>
        fn(Object.assign({ type, target: this }, ev || {})));
    },
  };
}

function fakePlayer() {
  const listeners = {};
  const el = {
    _src: "",
    currentTime: 0,
    paused: true,
    duration: NaN,
    readyState: 0,
    plays: 0,
    get src() { return el._src; },
    set src(v) {
      const wasPlaying = !el.paused;
      el._src = v;
      // a browser forgets the position and stops when the source changes
      el.currentTime = 0;
      el.paused = true;
      el.readyState = 0;
      if (wasPlaying) el.dispatch("pause");
    },
    load() {
      el.paused = true;
      el.readyState = 0;
    },
    play() {
      el.paused = false;
      el.readyState = 1;
      el.plays += 1;
      el.dispatch("play");
      return Promise.resolve();
    },
    pause() {
      el.paused = true;
      el.dispatch("pause");
    },
    addEventListener(type, fn) {
      (listeners[type] = listeners[type] || []).push(fn);
    },
    removeEventListener(type, fn) {
      const list = listeners[type] || [];
      const i = list.indexOf(fn);
      if (i >= 0) list.splice(i, 1);
    },
    dispatch(type, ev) {
      (listeners[type] || []).slice().forEach((fn) =>
        fn(Object.assign({ type, target: el }, ev || {})));
    },
  };
  return el;
}

const MATCHED = {
  status: "matched",
  level_lufs: -12.3,
  export_lufs: -9.1,
  level_offset_db: -3.2,
  margin_db: 0.2,
};

function harness(options) {
  const opts = options || {};
  // the page's language helper; the page defaults to English
  const lang = opts.lang || "en";
  const els = {};
  const $ = (id) => (els[id] = els[id] || fakeElement(id));
  const player = fakePlayer();
  els.player = player;
  const waves = [];
  const sandbox = {
    console,
    Math, JSON, Number, String, Object, Array, Boolean, Promise, Date,
    isFinite, parseFloat, parseInt, Error,
    $,
    sessionId: "sess1",
    current: "mix",
    monitoring: opts.monitoring === undefined ? MATCHED : opts.monitoring,
    document: { getElementById: $ },
    drawWave: (kind) => { waves.push(kind); },
    LANG: lang,
    L: (zh, en) => (lang === "zh" ? zh : en),
  };
  vm.createContext(sandbox);
  vm.runInContext(extractBlock(), sandbox);
  return {
    sandbox,
    player,
    $,
    waves,
    // what the browser does when the new source has been read
    finishLoad(duration) {
      player.duration = duration;
      player.readyState = 1;
      player.dispatch("loadedmetadata");
    },
  };
}

const tests = [];
function test(name, fn) { tests.push([name, fn]); }

test("a paused player keeps its position when the view changes", () => {
  const h = harness();
  h.player.currentTime = 101.87197;
  h.player.paused = true;
  h.sandbox.setAB("raw");
  assert.ok(h.player.src.indexOf("/audio/monitor/raw") >= 0, h.player.src);
  h.finishLoad(210);
  assert.ok(Math.abs(h.player.currentTime - 101.87197) < 1e-6,
    "position " + h.player.currentTime);
  assert.strictEqual(h.player.paused, true, "the music started on its own");
});

test("a playing player keeps playing where it was", () => {
  const h = harness();
  h.player.currentTime = 50;
  h.player.paused = false;
  h.sandbox.setAB("mix");
  h.finishLoad(210);
  assert.ok(Math.abs(h.player.currentTime - 50) < 1e-6,
    "position " + h.player.currentTime);
  assert.strictEqual(h.player.paused, false, "the music stopped");
});

test("a quick second click keeps the position that was saved", () => {
  const h = harness();
  h.player.currentTime = 30;
  h.player.paused = false;
  h.sandbox.setAB("raw");   // the first switch saves the position
  h.sandbox.setAB("mix");   // the second one must not lose it
  h.finishLoad(210);
  assert.ok(Math.abs(h.player.currentTime - 30) < 1e-6,
    "position " + h.player.currentTime);
  assert.strictEqual(h.player.paused, false, "the music stopped");
});

test("an answer that arrives late does not put the playhead back", () => {
  const h = harness();
  h.player.currentTime = 42;
  h.player.paused = false;
  h.sandbox.setAB("raw");
  h.finishLoad(210);
  assert.ok(Math.abs(h.player.currentTime - 42) < 1e-6,
    "position " + h.player.currentTime);
  h.player.currentTime = 90;
  h.sandbox.setAB("mix");
  h.finishLoad(210);
  assert.ok(Math.abs(h.player.currentTime - 90) < 1e-6,
    "position " + h.player.currentTime);
});

test("a new render starts at the beginning", () => {
  const h = harness();
  h.player.currentTime = 120;
  h.player.paused = false;
  h.sandbox.setAB("mix", { fresh: true });
  h.finishLoad(210);
  assert.strictEqual(h.player.currentTime, 0);
  assert.strictEqual(h.player.paused, true, "a new render must not start playing");
});

test("the comparison plays the level-matched pair", () => {
  const h = harness();
  h.sandbox.setAB("mix");
  assert.ok(h.player.src.indexOf("/audio/monitor/mix") >= 0, h.player.src);
  h.sandbox.setAB("raw");
  assert.ok(h.player.src.indexOf("/audio/monitor/raw") >= 0, h.player.src);
  // the peaks follow the view that is audible
  assert.deepStrictEqual(h.waves, ["monitor_mix", "monitor_raw"]);
});

for (const [lang, remix] of [["zh", /重新混音/], ["en", /mix again/i]]) {
  test(`without a matched pair the raw comparison is not offered (${lang})`, () => {
    const h = harness({ monitoring: null, lang });
    h.sandbox.refreshABState();
    assert.strictEqual(h.$("abraw").disabled, true, "the raw button is still live");
    assert.ok(remix.test(h.$("abraw").title), h.$("abraw").title);
    h.sandbox.setAB("raw");
    assert.ok(h.player.src.indexOf("monitor/raw") < 0,
      "the raw view was played anyway");
    assert.ok(remix.test(h.$("abnote").textContent),
      h.$("abnote").textContent);
  });
}

test("the matched note reports the levels in either language", () => {
  const zh = harness({ lang: "zh" });
  const en = harness({ lang: "en" });
  zh.sandbox.setAB("mix");
  en.sandbox.setAB("mix");
  assert.ok(/比較音量：-12\.3 LUFS（輸出 -9\.1 LUFS，比輸出低 3\.2 dB）/.test(zh.$("abnote").textContent),
    zh.$("abnote").textContent);
  assert.ok(/^Comparison level: -12\.3 LUFS \(output -9\.1 LUFS, 3\.2 dB below the output\)\./.test(en.$("abnote").textContent),
    en.$("abnote").textContent);
});

test("a pause the user made is not undone by a later switch", () => {
  const h = harness();
  h.player.currentTime = 30;
  h.player.paused = false;
  h.sandbox.setAB("raw");
  h.finishLoad(210);
  assert.strictEqual(h.player.paused, false);
  h.player.pause();          // the user stops the music
  h.sandbox.setAB("mix");
  h.finishLoad(210);
  assert.strictEqual(h.player.paused, true,
    "the music started against the user's pause");
  assert.ok(Math.abs(h.player.currentTime - 30) < 1e-6,
    "position " + h.player.currentTime);
});

test("a position beyond the end is clamped", () => {
  const h = harness();
  h.player.currentTime = 500;
  h.player.paused = true;
  h.sandbox.setAB("raw");
  h.finishLoad(210);
  assert.ok(h.player.currentTime <= 210, "position " + h.player.currentTime);
});

test("an explicit play request still plays", () => {
  const h = harness();
  h.player.currentTime = 12;
  h.player.paused = true;
  h.sandbox.switchSource("/api/session/sess1/audio/stem/kick.wav",
    { forcePlay: true });
  h.finishLoad(210);
  assert.strictEqual(h.player.paused, false, "the solo did not play");
  assert.ok(Math.abs(h.player.currentTime - 12) < 1e-6,
    "position " + h.player.currentTime);
});

test("a pause the user made while the source loads wins over the saved intent", () => {
  const h = harness();
  h.player.currentTime = 43;
  h.player.paused = false;
  h.sandbox.setAB("raw");
  // the user presses play and then pause while the new source loads
  h.player.play();
  h.player.pause();
  h.finishLoad(210);
  assert.strictEqual(h.player.paused, true,
    "the music was resumed against the user's pause");
  assert.ok(Math.abs(h.player.currentTime - 43) < 1e-6,
    "position " + h.player.currentTime);
});

test("a solo plays even while the previous switch is still waiting", () => {
  const h = harness();
  h.player.currentTime = 55;
  h.player.paused = true;
  h.sandbox.setAB("raw");
  h.sandbox.switchSource("/api/session/sess1/audio/stem/kick.wav",
    { forcePlay: true });
  h.finishLoad(210);
  assert.strictEqual(h.player.paused, false, "the solo did not play");
  assert.ok(Math.abs(h.player.currentTime - 55) < 1e-6,
    "position " + h.player.currentTime);
});

test("a new render starts at zero even while another switch waits", () => {
  const h = harness();
  h.player.currentTime = 55;
  h.player.paused = true;
  h.sandbox.setAB("raw");
  h.sandbox.setAB("mix", { fresh: true });
  h.finishLoad(210);
  assert.strictEqual(h.player.currentTime, 0);
  assert.strictEqual(h.player.paused, true);
});

test("an answer for an older switch does not move the playhead", () => {
  const h = harness();
  h.player.currentTime = 30;
  h.player.paused = false;
  h.sandbox.setAB("raw");   // the first switch
  h.sandbox.setAB("mix");   // a second one before the first answer
  h.finishLoad(210);        // the answers land here
  assert.ok(Math.abs(h.player.currentTime - 30) < 1e-6,
    "position " + h.player.currentTime);
  assert.strictEqual(h.player.paused, false, "the music stopped");
});

let failed = 0;
for (const [name, fn] of tests) {
  try {
    fn();
    console.log("ok   " + name);
  } catch (err) {
    failed += 1;
    console.log("FAIL " + name + "\n     " + (err && err.message));
  }
}
console.log((tests.length - failed) + " passed, " + failed + " failed");
process.exit(failed ? 1 : 0);
