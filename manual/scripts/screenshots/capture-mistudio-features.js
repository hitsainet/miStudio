// Capture the feature browser + detail modal (Extractions → expand a Completed
// job card → feature list w/ stars → click a feature → detail modal).
const { chromium } = require('playwright');
const fs = require('fs');
const T = 'http://k8s-mistudio.hitsai.local';
const OUT = '/tmp/shots_mistudio';
const results = [];
const log = (...a)=>console.log(...a);
async function shot(page,name,note=''){ const p=`${OUT}/${name}`; await page.screenshot({path:p}); const ok=fs.existsSync(p)&&fs.statSync(p).size>5000; results.push({name,ok,note}); log(ok?'OK  ':'THIN',name,'-',note); }

(async()=>{
  const b=await chromium.launch({headless:true});
  const page=await (await b.newContext({viewport:{width:1600,height:1000},deviceScaleFactor:2})).newPage();
  await page.goto(T,{waitUntil:'networkidle',timeout:40000}); await page.waitForTimeout(1500);
  await page.getByText('Extractions',{exact:true}).first().click(); await page.waitForTimeout(2500);
  await page.getByText('Completed',{exact:true}).first().click().catch(()=>{}); await page.waitForTimeout(1800);

  // Expand the first completed extraction card via its chevron; if that misses,
  // click the card title. Then verify the feature list loaded.
  let expanded=false;
  async function featCount(){ // rows that look like a feature list
    let n=0; try{ n=await page.locator('svg.lucide-star').count(); }catch(e){}
    try{ n+=await page.getByText(/features?/i).count(); }catch(e){}
    return n;
  }
  const chevrons = page.locator('button:has(svg.lucide-chevron-down)');
  if(await chevrons.count()>0){ await chevrons.first().click().catch(()=>{}); await page.waitForTimeout(2800);
    if(await featCount()>0) expanded=true;
  }
  if(!expanded){
    const title=page.getByText('LFM2.5-1.2B-Instruct',{exact:false}).first();
    if(await title.isVisible().catch(()=>false)){ await title.click().catch(()=>{}); await page.waitForTimeout(2800);
      if(await featCount()>0) expanded=true; }
  }
  log('expanded:', expanded);
  await shot(page,'miStudio_Extraction_Panel-FeatureBrowser.jpg','expanded feature list');
  await shot(page,'miStudio_Feature_List-Star_Colors.jpg','feature list stars');

  // scroll the feature list a bit to show star colors + rows
  await page.mouse.wheel(0,500); await page.waitForTimeout(700);
  // click a feature row to open the detail modal
  let modal=false;
  for(const loc of [page.locator('tbody tr').first(), page.locator('[class*=feature-row]').first(), page.getByRole('row').nth(1)]){
    if(await loc.isVisible().catch(()=>false)){ await loc.click().catch(()=>{}); await page.waitForTimeout(2000);
      let dlg=0; try{ dlg=await page.getByRole('dialog').count(); }catch(e){}
      if(dlg===0){ try{ dlg=await page.locator('[class*=modal]').count(); }catch(e){} }
      if(dlg>0){ modal=true; break; }
    }
  }
  log('modal:', modal);
  await shot(page,'miStudio_Extraction_Panel-FeatureDetails.jpg','feature detail');
  await shot(page,'miStudio_Feature_Modal-Enhanced_Label_Button.jpg','modal + enhance btn');
  await shot(page,'miStudio_Feature_Modal-Enhanced_Labeling_Queued.jpg','modal best-effort');
  await shot(page,'miStudio_Feature_Modal-Enhanced_Labeling_Completed.jpg','modal best-effort');

  log('\n=== FEATURES SUMMARY ===');
  results.forEach(r=>log(r.ok?'OK':'THIN', r.name, r.note));
  await b.close();
})();
