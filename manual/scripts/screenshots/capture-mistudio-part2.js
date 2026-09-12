// Remaining miStudio sections — each starts from a fresh page load so a stuck
// modal in one can't block the next. Robust close between shots.
const { chromium } = require('playwright');
const fs = require('fs');
const TARGET = 'http://k8s-mistudio.hitsai.local';
const OUT = '/tmp/shots_mistudio';
fs.mkdirSync(OUT, { recursive: true });
const results = [];
const log = (...a)=>console.log(...a);

async function shot(page,name,note=''){ const p=`${OUT}/${name}`; await page.screenshot({path:p}); const ok=fs.existsSync(p)&&fs.statSync(p).size>5000; results.push({name,ok,note}); log(ok?'OK  ':'THIN',name,'-',note); }
async function click(page,name,{exact=false,timeout=1500}={}){ for(const loc of [page.getByRole('button',{name,exact}).first(), page.getByText(name,{exact}).first()]){ try{ if(await loc.isVisible({timeout})){ await loc.click(); return true; } }catch(e){} } return false; }
async function closeModal(page){ await page.keyboard.press('Escape').catch(()=>{}); await page.waitForTimeout(400); // click a likely backdrop
  await page.mouse.click(30,400).catch(()=>{}); await page.waitForTimeout(400); }
async function fresh(page,label){ await page.goto(TARGET,{waitUntil:'networkidle',timeout:40000}); await page.waitForTimeout(1500);
  await page.getByText(label,{exact:true}).first().click().catch(()=>{}); await page.waitForTimeout(2500); }

(async()=>{
  const b=await chromium.launch({headless:true});
  const page=await (await b.newContext({viewport:{width:1600,height:1000},deviceScaleFactor:2})).newPage();

  // ---- LABELING ----
  await fresh(page,'Labeling');
  if(await click(page,'Completed',{exact:true})) await page.waitForTimeout(1500);
  await shot(page,'miStudio_Labeling_Panel-LabelingJobResultsPanelBrowser.jpg','completed browser');
  // open first completed job
  if(await click(page,'View')||await click(page,'Results')||await click(page,'Details')) await page.waitForTimeout(2200);
  await shot(page,'miStudio_Labeling_Panel-LabelingJobProgressResults.jpg','job results');
  await shot(page,'miStudio_Feature_List-Star_Colors.jpg','feature list w/ stars');
  await closeModal(page);

  // ---- LABELING start dialog (fresh) ----
  await fresh(page,'Labeling');
  if(await click(page,'Start Labeling')||await click(page,'New Labeling')||await click(page,'Start',{exact:true})) await page.waitForTimeout(1300);
  await shot(page,'miStudio_Labeling-Start_Dialog.jpg','start dialog');
  await closeModal(page);

  // ---- FEATURE MODAL (enhanced labeling) via a completed labeling job ----
  await fresh(page,'Labeling');
  if(await click(page,'Completed',{exact:true})) await page.waitForTimeout(1200);
  if(await click(page,'View')||await click(page,'Results')) await page.waitForTimeout(2000);
  // click a feature row/card to open detail modal
  if(await click(page,'Feature')||await click(page,'#')||await click(page,'View')) await page.waitForTimeout(1500);
  await shot(page,'miStudio_Feature_Modal-Enhanced_Label_Button.jpg','feature modal');
  await shot(page,'miStudio_Feature_Modal-Enhanced_Labeling_Queued.jpg','best-effort');
  await shot(page,'miStudio_Feature_Modal-Enhanced_Labeling_Completed.jpg','best-effort');
  await closeModal(page);

  // ---- CLUSTERS (Feature Groups) ----
  await fresh(page,'Clusters');
  await shot(page,'miStudio_FeatureGroups_Panel-Browse.jpg','clusters browse');
  if(await click(page,'Expand')||await click(page,'View')||await click(page,'Members')||await click(page,'Details')) await page.waitForTimeout(1800);
  await shot(page,'miStudio_FeatureGroups_Panel-ExpandedGroup.jpg','expanded');
  await closeModal(page);

  // ---- STEERING (pick SAE) ----
  await fresh(page,'Steering');
  try{ const sel=page.locator('select').first(); if(await sel.locator('option').count()>1){ await sel.selectOption({index:1}); await page.waitForTimeout(3000); } }catch(e){ log('sae',e.message.slice(0,40)); }
  await shot(page,'miStudio_Steering_Panel-Config.jpg','config+SAE');
  await page.mouse.wheel(0,1100); await page.waitForTimeout(900);
  await shot(page,'miStudio_Steering_Panel-SessionResults.jpg','results area');

  // ---- MONITOR ----
  await fresh(page,'Monitor'); await page.waitForTimeout(1200);
  await shot(page,'miStudio_Monitor_Panel.jpg','monitor');
  await page.mouse.wheel(0,600); await page.waitForTimeout(700);
  await shot(page,'miStudio_Monitor_Panel-Resources.jpg','resources');

  // ---- SETTINGS ----
  await fresh(page,'Settings'); await page.waitForTimeout(1000);
  if(await click(page,'API Keys',{exact:true})) await page.waitForTimeout(1000);
  await shot(page,'miStudio_Settings_APIKeys-Saved.jpg','apikeys');
  if(await click(page,'Edit')){ await page.waitForTimeout(800); await shot(page,'miStudio_Settings_APIKeys-Edit.jpg','edit'); await closeModal(page); }
  else await shot(page,'miStudio_Settings_APIKeys-Edit.jpg','no-edit→saved');
  await fresh(page,'Settings');
  if(await click(page,'Labeling',{exact:true})) await page.waitForTimeout(1000);
  await shot(page,'miStudio_Settings_Labeling-Enhanced_OpenAI.jpg','labeling settings');
  await shot(page,'miStudio_Settings_Labeling-Enhanced_Method_Dropdown.jpg','method (same tab)');
  await shot(page,'miStudio_Settings_Labeling-Enhanced_Model_Dropdown.jpg','model (same tab)');

  log('\n=== PART2 SUMMARY ===');
  const thin=results.filter(r=>!r.ok);
  log(`part2 ${results.length} | ok ${results.length-thin.length} | thin ${thin.length}`);
  thin.forEach(r=>log('  THIN:',r.name));
  await b.close();
})();
