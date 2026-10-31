const { chromium } = require('playwright');
const fs = require('fs');
const TARGET = 'http://k8s-millm.hitsai.local/';
const OUT = '/tmp/shots_millm';
fs.mkdirSync(OUT, { recursive: true });
const results = [];
async function nav(page, label){ await page.getByText(label,{exact:true}).first().click().catch(()=>{}); await page.waitForTimeout(2500); }
async function shot(page, name){ const p=`${OUT}/${name}`; await page.screenshot({path:p,fullPage:false}); const ok=fs.existsSync(p)&&fs.statSync(p).size>5000; results.push({name,ok}); console.log(ok?'OK  ':'THIN', name); }
(async()=>{
  const b=await chromium.launch({headless:true});
  const page=await (await b.newContext({viewport:{width:1600,height:1000},deviceScaleFactor:2})).newPage();
  await page.goto(TARGET,{waitUntil:'networkidle',timeout:40000}); await page.waitForTimeout(2000);
  await shot(page, 'miLLM_Dashboard_01.jpg');           // landing = dashboard
  await nav(page,'Models');   await shot(page, 'miLLM_Models_01.jpg');
  await nav(page,'SAEs');     await shot(page, 'miLLM_SAEs_01.jpg');
  await nav(page,'Profiles'); await shot(page, 'miLLM_Profiles_01.jpg');
  await nav(page,'Settings'); await shot(page, 'miLLM_Settings_01.jpg');
  console.log('total', results.length, 'ok', results.filter(r=>r.ok).length);
  fs.writeFileSync(`${OUT}/_manifest.json`, JSON.stringify(results,null,1));
  await b.close();
})();
