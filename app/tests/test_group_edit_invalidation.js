/* Execute actual web helpers: successful changed edits clear old media;
 * unchanged/rejected POSTs preserve it, including deferred playback intent. */
'use strict';
const assert=require('assert');
const fs=require('fs');
const path=require('path');
const vm=require('vm');
const source=fs.readFileSync(process.env.GROUP_WEB_SOURCE||path.join(__dirname,'../web/index.html'),'utf8');
function extract(name) {
  const marker=source.includes('async function '+name+'(')?'async function ':'function ';
  const lo=source.indexOf(marker+name+'(');
  assert(lo>=0,name+' missing');
  const hi=source.indexOf('\n}',lo);assert(hi>lo,name+' unterminated');
  return source.slice(lo,hi+2);
}
function fixture(response,decl=[{id:'room',members:['a','b'],pan:0}],lang='en') {
  const counts={pause:0,load:0,play:0};const hidden=new Set();
  const player={src:'old.wav',paused:false,currentTime:42,duration:100,
    pause(){this.paused=true;counts.pause++},
    removeAttribute(name){assert.equal(name,'src');this.src=''},
    getAttribute(){return this.src},load(){counts.load++},play(){counts.play++;return Promise.resolve()}};
  const result={classList:{add(c){hidden.add(c)},contains(c){return hidden.has(c)}}};
  const status={textContent:'old success',innerHTML:'',appendChild(c){this.child=c}};
  const elements={player,result,status,abraw:{disabled:false},abnote:{textContent:'matched'}};
  // the page's language helper, in the language this case runs in
  const ctx={console,sessionId:'fixture',groupSavedDecl:{},$:id=>elements[id],
    LANG:lang,L:(zh,en)=>lang==='zh'?zh:en,
    document:{createElement:()=>({className:'',textContent:''})},
    captureGroupsPayload:()=>decl,api:async()=>{if(response instanceof Error)throw response;return response;}};
  vm.createContext(ctx);
  vm.runInContext("let groupSaveQueue=Promise.resolve();let abGen=7,abIntent={gen:7,at:42,resume:true,url:'old.wav'},abRestored=false,abOwnPauses=1,abUserSpoke=false;let monitoring={status:'matched'};"+
    ['saveGroups','invalidateComparison','refreshABState','setABNote','restoreAfterSwitch'].map(extract).join('\n'),ctx);
  return {ctx,counts,player,result,status,elements,state:()=>vm.runInContext('({abGen,abIntent,monitoring})',ctx)};
}
(async()=>{
  let cases=0;
  for(const [lang,remix] of [['zh',/重新混音/],['en',/mix again/i]]) for(const declaration of [
    [{id:'new-name',members:['a','b'],pan:0}],
    [{id:'room',members:['a','c'],pan:0}],
    [{id:'room',members:['a','b'],pan:.2}],[],
    [{id:'room',members:['b','a'],mode:'stereo_pair',pan:0,trim_db:2}]
  ]) {
    const f=fixture({changed:true},declaration,lang);
    assert.equal(await f.ctx.saveGroups(),true);
    assert(f.result.classList.contains('hide'));
    assert.equal(f.player.paused,true);assert.equal(f.player.src,'');
    assert.equal(f.elements.abraw.disabled,true);
    assert.equal(f.state().monitoring,null);assert.equal(f.state().abIntent,null);
    assert(f.state().abGen>7);assert.match(f.status.textContent,remix);
    assert.match(f.elements.abnote.textContent,remix);
    assert.equal(f.counts.pause,1);assert.equal(f.counts.load,1);
    // Simulate the old metadata callback arriving after the group edit.
    f.ctx.restoreAfterSwitch(7,'old.wav');assert.equal(f.counts.play,0);
    assert.equal(f.player.currentTime,42);cases++;
  }
  for(const response of [{changed:false},new Error('rejected group')]) {
    const f=fixture(response);const ok=await f.ctx.saveGroups();
    assert.equal(ok,!(response instanceof Error));
    assert.equal(f.player.src,'old.wav');assert.equal(f.player.paused,false);
    assert.equal(f.counts.pause,0);assert.equal(f.counts.load,0);
    assert(!f.result.classList.contains('hide'));
    assert.equal(f.state().monitoring.status,'matched');assert.equal(f.state().abGen,7);
    f.ctx.restoreAfterSwitch(7,'old.wav');assert.equal(f.counts.play,1);cases++;
  }
  // A paused valid result also remains paused and intact on the Run no-op POST.
  const paused=fixture({changed:false});paused.player.paused=true;
  assert.equal(await paused.ctx.saveGroups(),true);assert(paused.player.paused);
  assert.equal(paused.counts.load,0);cases++;
  // A slow first save must finish before Run's subsequent no-op save resolves.
  const queued=fixture({changed:false});let release;const order=[];
  queued.ctx.api=async()=>{order.push('request');if(order.length===1)return await new Promise(r=>{release=r});return {changed:false}};
  const first=queued.ctx.saveGroups();await Promise.resolve();
  const runSave=queued.ctx.saveGroups().then(ok=>{assert(ok);order.push('publish new render')});
  await Promise.resolve();assert.deepEqual(order,['request']);
  release({changed:true});await first;await runSave;
  assert.deepEqual(order,['request','request','publish new render']);
  assert.equal(queued.counts.pause,1);cases++;
  // A response for a different session must not clear the newly selected one.
  const stale=fixture({changed:false});let late;
  stale.ctx.api=async()=>await new Promise(r=>{late=r});
  const pending=stale.ctx.saveGroups();await Promise.resolve();stale.ctx.sessionId='another';
  late({changed:true});assert.equal(await pending,false);
  assert.equal(stale.player.src,'old.wav');assert.equal(stale.counts.pause,0);
  assert.equal(stale.state().monitoring.status,'matched');cases++;
  console.log(cases+' actual group-save behavioral cases PASS');
})().catch(e=>{console.error(e);process.exitCode=1});
