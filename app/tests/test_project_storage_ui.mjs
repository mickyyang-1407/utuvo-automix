// Exercise the actual page's handlers with a small DOM/fetch harness, no browser.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const html = readFileSync(fileURLToPath(new URL('../web/index.html', import.meta.url)), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/\nboot\(\);\n/, '\n');
const old = {style:'rock', intensity:0.8, source_balance_profile:'balanced_stems',
  balance_preserve:0.35, blend:0.7, target_lufs:-20, export_stems:true, reverb_amount:0.5};
const draft = {...old, intensity:0, balance_preserve:null, blend:0,
  target_lufs:null, export_stems:false, reverb_amount:0};

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {resolve = yes; reject = no;});
  return {promise, resolve, reject};
}
const response = (body, ok = true) => ({ok, statusText:'failure',
  headers:{get:()=>'application/json'}, json:async()=>body});

// The page picks its language from ?lang=, then the saved choice, then the
// browser; every case runs once in each language.
function harness(lang = 'en', {navigatorLang, stored, search = ''} = {}) {
  function element() {
    const classes = new Set();
    return {value:'', textContent:'', innerHTML:'', checked:false, disabled:false,
      inert:false, children:[], get options(){return this.children;}, dataset:{}, style:{setProperty(){}},
      classList:{add:(...xs)=>xs.forEach(x=>classes.add(x)),
        remove:(...xs)=>xs.forEach(x=>classes.delete(x)), contains:x=>classes.has(x),
        toggle:(x,on)=>{if (on) classes.add(x); else classes.delete(x);}},
      addEventListener(){}, removeEventListener(){}, setAttribute(){}, removeAttribute(){},
      appendChild(child){this.children.push(child); return child;},
      replaceChildren(...children){this.children = children;},
      querySelectorAll(){return [];}, getContext(){return {};},
      pause(){}, load(){}, play(){return Promise.resolve();},
    };
  }
  const els = new Map();
  const get = id => {if (!els.has(id)) els.set(id, element()); return els.get(id);};
  const calls = [], navigation = [];
  const location = {hash:'', pathname:'/', search, reload(){navigation.push(['reload', this.hash]);}};
  let fetcher = async path => path === '/api/projects'
    ? response({projects:[{id:'abcdef123456', name:'saved', saved_at:1}], root:'/isolated/projects'})
    : path.endsWith('/project') ? response({id:'abcdef123456',name:'saved',path:'/isolated/projects/abcdef123456'})
    : path.endsWith('/open') ? response({id:'123456abcdef'}) : response({});
  const context = vm.createContext({console, setTimeout, clearTimeout, setInterval, clearInterval,
    performance, URL, Date, Map, Set, Promise, JSON, Number, Math,
    requestAnimationFrame(){},  // the overview redraw is browser-only; this harness checks handlers
    document:{getElementById:get, createElement:element, createTextNode:t=>({textContent:String(t)}), querySelectorAll:()=>[], body:element()},
    window:{addEventListener(){}}, location,
    navigator:{language: navigatorLang || (lang === 'zh' ? 'zh-TW' : 'en-US')},
    localStorage:{getItem(){ if (stored instanceof Error) throw stored; return stored ?? null; },
      setItem(){}},
    history:{replaceState(_state,_title,url){location.hash=url.includes('#') ? '#'+url.split('#')[1] : ''; navigation.push(['history',url]);}},
    fetch:async(path,opts)=>{calls.push({path,opts}); return fetcher(path,opts);},
  });
  vm.runInContext(script + `
    globalThis.test = {
      api, saveProject, openProject, refreshProjects, restore, runBody, lang: () => LANG,
      setLanguage, run, groupsSave(fn){saveGroups=fn;}, switching(){return languageSwitching;},
      statusError,
      setup(){projectLoading=false; sessionId='111111111111'; stems=['kick.wav']; styleLibrary=[{id:'rock'}];
        style='rock'; saveGroups=async()=>true; showStyleIntent=()=>{};},
      imports(n){importsInProgress=n;}, pending(){return pendingEdits.size;},
      restoreStubs(){renderFiles=()=>{}; refreshGo=()=>{}; refreshRefButton=()=>{};
        loadRoles=()=>{}; show=()=>{}; seedManualRoles=()=>{};},
    };`, context);
  context.test.setup();
  get('project-name').value = 'saved';
  get('sourceprofile').value = 'balanced_stems';
  get('intensity').value = '0.8'; get('balance').value = '0.35';
  get('blend').value = '0.7'; get('reverb-amount').value = '50';
  return {test:context.test, get, calls, context, navigation, location,
    setFetch(fn){fetcher=fn;}, defaultFetch:(...args)=>fetcher(...args)};
}

const TEXT = {
  zh: {saved:/已另存/, failed:/另存失敗.*disk full/, wait:/請先等/,
    draft:/既有結果仍屬上次成功混音/, restored:/已還原上次成功混音/},
  en: {saved:/Saved “/, failed:/Save failed: .*disk full/, wait:/Wait for the import or mix/,
    draft:/still from the last successful mix/, restored:/Restored the settings of the last successful mix/},
};

let cases = 0;
// the language the page starts in
for (const [opts, want] of [
  [{navigatorLang:'en-US'}, 'en'], [{navigatorLang:'zh-TW'}, 'zh'], [{navigatorLang:'zh'}, 'zh'],
  [{navigatorLang:'fr-FR'}, 'en'], [{navigatorLang:'en-US', stored:'zh'}, 'zh'],
  [{navigatorLang:'zh-TW', stored:'en'}, 'en'], [{navigatorLang:'zh-TW', stored:new Error('blocked')}, 'zh'],
  [{navigatorLang:'zh-TW', stored:'zh', search:'?lang=en'}, 'en'],
]) {
  assert.equal(harness('en', opts).test.lang(), want, JSON.stringify(opts));
  cases++;
}
for (const lang of ['zh', 'en']) {
const T = TEXT[lang];
assert.equal(harness(lang).test.lang(), lang);
{
  const h = harness(lang), edit = deferred();
  h.setFetch(async(path) => path.endsWith('/pan') ? edit.promise
    : path === '/api/projects' ? response({projects:[],root:'/isolated/projects'})
    : response({id:'abcdef123456',name:'saved',path:'/isolated/projects/abcdef123456'}));
  const request = h.test.api('/api/session/111111111111/pan', {method:'POST'});
  assert.equal(h.test.pending(), 1);
  const saving = h.test.saveProject();
  await Promise.resolve();
  assert.equal(h.get('workspace').inert, true);
  assert.equal(h.calls.filter(c=>c.path.endsWith('/project')).length, 0);
  assert.doesNotMatch(h.get('project-status').textContent, T.saved);
  edit.resolve(response({ok:true}));
  await request; await saving;
  assert.equal(h.test.pending(), 0);
  assert.equal(h.calls.filter(c=>c.path.endsWith('/project')).length, 1);
  assert.match(h.get('project-status').textContent, T.saved);
  assert.equal(h.get('workspace').inert, false);
  cases++;
}
{
  const h = harness(lang), save = deferred();
  h.setFetch(async path => path.endsWith('/project') ? save.promise
    : response({projects:[],root:'/isolated/projects'}));
  const saving = h.test.saveProject();
  await new Promise(resolve=>setImmediate(resolve));
  assert.doesNotMatch(h.get('project-status').textContent, T.saved);
  save.resolve(response({error:'disk full'},false));
  await saving;
  assert.match(h.get('project-status').textContent, T.failed);
  assert.equal(h.get('workspace').inert, false);
  assert.equal(h.get('save-project').disabled, false);
  cases++;
}
for (const busy of ['import','render']) {
  const h = harness(lang);
  if (busy === 'import') h.test.imports(1);
  else h.context.document.body.classList.add('busy');
  await h.test.saveProject();
  h.get('saved-projects').value = 'abcdef123456';
  await h.test.openProject();
  assert.equal(h.calls.length,0);
  assert.match(h.get('project-status').textContent,T.wait);
  assert.equal(h.navigation.length,0);
  cases++;
}
{
  const h = harness(lang);
  h.setFetch(async()=>response({projects:[{id:'abcdef123456',name:'<img src=x onerror=bad()>',saved_at:1}],root:'/isolated/projects'}));
  await h.test.refreshProjects();
  const option = h.get('saved-projects').children[1];
  assert.match(option.textContent,/<img src=x onerror=bad\(\)>/);
  assert.equal(option.innerHTML,'');
  cases++;
}
{
  const h = harness(lang), edit = deferred();
  h.setFetch(async path => path.endsWith('/trim') ? edit.promise : response({id:'123456abcdef'}));
  const request = h.test.api('/api/session/111111111111/trim', {method:'POST'});
  h.get('saved-projects').value = 'abcdef123456';
  const opening = h.test.openProject();
  await Promise.resolve();
  assert.equal(h.navigation.length,0);
  edit.resolve(response({ok:true}));
  await request; await opening;
  assert.deepEqual(h.navigation,[['history','/#s=123456abcdef'],['reload','#s=123456abcdef']]);
  assert.equal(h.calls.filter(c=>c.path.endsWith('/open')).length,1);
  cases++;
}
for (const savedDraft of [draft,null]) {
  const h = harness(lang);
  h.test.restoreStubs();
  h.location.hash='#s=111111111111';
  const state = {stems:['kick.wav'],track_ids:{'kick.wav':'kick.wav'},capture_groups:[],
    has_reference:false,has_mix:true,controls:old,draft_controls:savedDraft,
    project_name:'restored',reverb_amount:0.5};
  h.setFetch(async path=>response(path.endsWith('/state') ? state : {target_lufs:-20}));
  await h.test.restore();
  // R118: every mix request carries the group faders explicitly ({} = all at
  // 0 dB), so "Reset all" really resets them on the server
  assert.deepEqual(JSON.parse(h.test.runBody()), {...(savedDraft || old), group_trims:{}});
  assert.equal(h.get('project-name').value,'restored');
  if (savedDraft) assert.match(h.get('control-hint').textContent,T.draft);
  else assert.match(h.get('control-hint').textContent,T.restored);
  cases++;
}
// switching language reloads: a group draft that cannot be saved must stop it,
// an edit still in flight is waited for, and the session stays in the URL
{
  const h = harness(lang);
  h.test.groupsSave(async () => false);
  await h.test.setLanguage(lang === 'en' ? 'zh' : 'en');
  assert.deepEqual(h.navigation, []);
  assert.match(h.get('status').textContent, lang === 'en' ? /Group settings are not saved yet/ : /群組設定尚未保存/);
  cases++;
}
{
  const h = harness(lang), edit = deferred();
  h.setFetch(async path => path.endsWith('/trim') ? edit.promise : response({}));
  const request = h.test.api('/api/session/111111111111/trim', {method:'POST'});
  const switching = h.test.setLanguage(lang === 'en' ? 'zh' : 'en');
  await Promise.resolve();
  assert.deepEqual(h.navigation, []);
  edit.resolve(response({ok:true}));
  await request; await switching;
  assert.deepEqual(h.navigation.at(-1), ['reload', '#s=111111111111']);
  cases++;
}
// a mix that starts while the switch waits stops the reload, and no mix can
// start while a switch is under way
{
  const h = harness(lang), edit = deferred();
  h.setFetch(async path => path.endsWith('/trim') ? edit.promise : response({}));
  const request = h.test.api('/api/session/111111111111/trim', {method:'POST'});
  const switching = h.test.setLanguage(lang === 'en' ? 'zh' : 'en');
  await Promise.resolve();
  assert.equal(h.test.switching(), true);
  const before = h.calls.length;
  await h.test.run();
  assert.equal(h.calls.length, before);  // run() refused: nothing was sent
  h.context.document.body.classList.add('busy');
  edit.resolve(response({ok:true}));
  await request; await switching;
  assert.deepEqual(h.navigation, []);
  assert.equal(h.test.switching(), false);
  assert.match(h.get('status').textContent, lang === 'en' ? /before switching language/ : /再切換語言/);
  cases++;
}
// an error that is not about groups is reported as itself
{
  const h = harness(lang), edit = deferred();
  h.setFetch(async path => path.endsWith('/trim') ? edit.promise : response({}));
  const request = h.test.api('/api/session/111111111111/trim', {method:'POST'}).catch(() => {});
  const switching = h.test.setLanguage(lang === 'en' ? 'zh' : 'en');
  edit.resolve(response({detail:'server exploded'}, false));
  await request; await switching;
  assert.deepEqual(h.navigation, []);
  assert.doesNotMatch(h.get('status').textContent, /Group settings|群組設定/);
  assert.match(h.get('status').textContent, lang === 'en' ? /Could not switch language/ : /無法切換語言/);
  cases++;
}
// an error that carries a file name is shown as text, never as markup
{
  const h = harness(lang);
  h.test.statusError('mix<img src=x onerror="alert(1)">.mp3: ffmpeg failed', '12s');
  const status = h.get('status');
  assert.equal(status.innerHTML, '');
  assert.equal(status.children[0].textContent, 'mix<img src=x onerror="alert(1)">.mp3: ffmpeg failed');
  assert.equal(status.children[0].className, 'err');
  cases++;
}
// a blank project name is saved under the name the user can read
{
  const h = harness(lang);
  h.get('project-name').value = '   ';
  await h.test.saveProject();
  const body = JSON.parse(h.calls.find(c => c.path.endsWith('/project')).opts.body);
  assert.equal(body.name, lang === 'en' ? 'Untitled project' : '未命名專案');
  cases++;
}
}
console.log(`${cases} project UI cases passed`);
