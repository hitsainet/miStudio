const { chromium } = require('playwright');
const fs=require('fs'); const T='http://k8s-mistudio.hitsai.local'; const OUT='/tmp/shots_mistudio';
async function shot(page,n){ const p=`${OUT}/${n}`; await page.screenshot({path:p}); console.log(fs.statSync(p).size>5000?'OK':'THIN', n); }
(async()=>{
  const b=await chromium.launch({headless:true});
  const page=await (await b.newContext({viewport:{width:1600,height:1000},deviceScaleFactor:2})).newPage();
  await page.goto(T,{waitUntil:'networkidle',timeout:40000}); await page.waitForTimeout(1500);
  await page.getByText('Extractions',{exact:true}).first().click(); await page.waitForTimeout(2500);
  await page.getByText('Completed',{exact:true}).first().click().catch(()=>{}); await page.waitForTimeout(1500);
  await page.locator('button:has(svg.lucide-chevron-down)').first().click().catch(()=>{}); await page.waitForTimeout(2800);
  await page.mouse.wheel(0,500); await page.waitForTimeout(600);
  // the feature rows are <tr> with cursor-pointer; click the first data row's CELL area (not a button)
  const rows = page.locator('table tbody tr');
  const rc = await rows.count();
  console.log('rows:', rc);
  let opened=false;
  for(let i=0;i<Math.min(rc,3) && !opened;i++){
    // click the LABEL cell (2nd td) to avoid the star button in the last cell
    const cell = rows.nth(i).locator('td').nth(1);
    await cell.click().catch(()=>{}); await page.waitForTimeout(1800);
    opened=(await page.getByRole('dialog').count())>0;
    if(!opened){ // maybe not role=dialog; check for a large overlay w/ the feature label
      opened = await page.getByText('Feature Detail',{exact:false}).count()>0 || await page.getByText('Enhanced Label',{exact:false}).count()>0;
    }
  }
  console.log('MODAL_OPENED:', opened);
  await shot(page,'miStudio_Extraction_Panel-FeatureDetails.jpg');
  await shot(page,'miStudio_Feature_Modal-Enhanced_Label_Button.jpg');
  await shot(page,'miStudio_Feature_Modal-Enhanced_Labeling_Completed.jpg');
  await shot(page,'miStudio_Feature_Modal-Enhanced_Labeling_Queued.jpg');
  await b.close();
})();
