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

function harness() {
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
  const location = {hash:'', pathname:'/', search:'', reload(){navigation.push(['reload', this.hash]);}};
  let fetcher = async path => path === '/api/projects'
    ? response({projects:[{id:'abcdef123456', name:'saved', saved_at:1}], root:'/isolated/projects'})
    : path.endsWith('/project') ? response({id:'abcdef123456',name:'saved',path:'/isolated/projects/abcdef123456'})
    : path.endsWith('/open') ? response({id:'123456abcdef'}) : response({});
  const context = vm.createContext({console, setTimeout, clearTimeout, setInterval, clearInterval,
    performance, URL, Date, Map, Set, Promise, JSON, Number, Math,
    document:{getElementById:get, createElement:element, querySelectorAll:()=>[], body:element()},
    window:{addEventListener(){}}, location,
    history:{replaceState(_state,_title,url){location.hash=url.includes('#') ? '#'+url.split('#')[1] : ''; navigation.push(['history',url]);}},
    fetch:async(path,opts)=>{calls.push({path,opts}); return fetcher(path,opts);},
  });
  vm.runInContext(script + `
    globalThis.test = {
      api, saveProject, openProject, refreshProjects, restore, runBody,
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

let cases = 0;
{
  const h = harness(), edit = deferred();
  h.setFetch(async(path) => path.endsWith('/pan') ? edit.promise
    : path === '/api/projects' ? response({projects:[],root:'/isolated/projects'})
    : response({id:'abcdef123456',name:'saved',path:'/isolated/projects/abcdef123456'}));
  const request = h.test.api('/api/session/111111111111/pan', {method:'POST'});
  assert.equal(h.test.pending(), 1);
  const saving = h.test.saveProject();
  await Promise.resolve();
  assert.equal(h.get('workspace').inert, true);
  assert.equal(h.calls.filter(c=>c.path.endsWith('/project')).length, 0);
  assert.doesNotMatch(h.get('project-status').textContent, /已另存/);
  edit.resolve(response({ok:true}));
  await request; await saving;
  assert.equal(h.test.pending(), 0);
  assert.equal(h.calls.filter(c=>c.path.endsWith('/project')).length, 1);
  assert.match(h.get('project-status').textContent, /已另存/);
  assert.equal(h.get('workspace').inert, false);
  cases++;
}
{
  const h = harness(), save = deferred();
  h.setFetch(async path => path.endsWith('/project') ? save.promise
    : response({projects:[],root:'/isolated/projects'}));
  const saving = h.test.saveProject();
  await new Promise(resolve=>setImmediate(resolve));
  assert.doesNotMatch(h.get('project-status').textContent, /已另存/);
  save.resolve(response({error:'disk full'},false));
  await saving;
  assert.match(h.get('project-status').textContent, /另存失敗.*disk full/);
  assert.equal(h.get('workspace').inert, false);
  assert.equal(h.get('save-project').disabled, false);
  cases++;
}
for (const busy of ['import','render']) {
  const h = harness();
  if (busy === 'import') h.test.imports(1);
  else h.context.document.body.classList.add('busy');
  await h.test.saveProject();
  h.get('saved-projects').value = 'abcdef123456';
  await h.test.openProject();
  assert.equal(h.calls.length,0);
  assert.match(h.get('project-status').textContent,/請先等/);
  assert.equal(h.navigation.length,0);
  cases++;
}
{
  const h = harness();
  h.setFetch(async()=>response({projects:[{id:'abcdef123456',name:'<img src=x onerror=bad()>',saved_at:1}],root:'/isolated/projects'}));
  await h.test.refreshProjects();
  const option = h.get('saved-projects').children[1];
  assert.match(option.textContent,/<img src=x onerror=bad\(\)>/);
  assert.equal(option.innerHTML,'');
  cases++;
}
{
  const h = harness(), edit = deferred();
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
  const h = harness();
  h.test.restoreStubs();
  h.location.hash='#s=111111111111';
  const state = {stems:['kick.wav'],track_ids:{'kick.wav':'kick.wav'},capture_groups:[],
    has_reference:false,has_mix:true,controls:old,draft_controls:savedDraft,
    project_name:'restored',reverb_amount:0.5};
  h.setFetch(async path=>response(path.endsWith('/state') ? state : {target_lufs:-20}));
  await h.test.restore();
  assert.deepEqual(JSON.parse(h.test.runBody()), savedDraft || old);
  assert.equal(h.get('project-name').value,'restored');
  if (savedDraft) assert.match(h.get('control-hint').textContent,/既有結果仍屬上次成功混音/);
  else assert.match(h.get('control-hint').textContent,/已還原上次成功混音/);
  cases++;
}
console.log(`${cases} project UI cases passed`);
