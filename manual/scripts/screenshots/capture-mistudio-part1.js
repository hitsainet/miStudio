// miStudio manual screenshot recapture (verified selectors). Saves to OUT.
const { chromium } = require('playwright');
const fs = require('fs');
const TARGET = process.env.TARGET || 'http://k8s-mistudio.hitsai.local';
const OUT = process.env.OUT || '/tmp/shots_mistudio';
fs.mkdirSync(OUT, { recursive: true });
const log = (...a) => console.log(...a);
const results = [];

async function nav(page, label) {
  await page.getByRole('button', { name: label, exact: true }).first().click().catch(async () => {
    await page.getByText(label, { exact: true }).first().click().catch(()=>{});
  });
  await page.waitForTimeout(2200);
}
async function shot(page, name, note='') {
  const p = `${OUT}/${name}`;
  await page.screenshot({ path: p, fullPage: false });
  const ok = fs.existsSync(p) && fs.statSync(p).size > 5000;
  results.push({ name, ok, note });
  log(ok ? 'OK  ' : 'THIN', name, '-', note);
}
async function click(page, name, {exact=false, timeout=1800}={}) {
  const el = page.getByRole('button', { name, exact }).first();
  try { if (await el.isVisible({ timeout })) { await el.click(); return true; } } catch(e){}
  const t = page.getByText(name, { exact }).first();
  try { if (await t.isVisible({ timeout })) { await t.click(); return true; } } catch(e){}
  return false;
}
const esc = (page) => page.keyboard.press('Escape').catch(()=>{});

(async () => {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  await page.goto(TARGET, { waitUntil: 'networkidle', timeout: 40000 });
  await page.waitForTimeout(2000);

  // MODELS
  await nav(page, 'Models');
  await shot(page, 'miStudio_Model_Panel-Browse.jpg');
  if (await click(page, 'Preview')) { await page.waitForTimeout(1800);
    await shot(page, 'miStudio_Model_Panel-PreviewModal.jpg', 'modal'); await esc(page); await page.waitForTimeout(600);
  } else await shot(page, 'miStudio_Model_Panel-PreviewModal.jpg', 'no-preview→browse');

  // SAEs
  await nav(page, 'SAEs');
  await shot(page, 'miStudio_SAE_Panel-Browse.jpg');
  if (await click(page, 'Download')) { await page.waitForTimeout(1500);
    await shot(page, 'miStudio_SAE_Panel-DownloadPretrainedSAE.jpg', 'download'); await esc(page);
  } else await shot(page, 'miStudio_SAE_Panel-DownloadPretrainedSAE.jpg', 'fallback→browse');

  // TEMPLATES (verified tab names)
  await nav(page, 'Templates');
  if (await click(page, 'Training Templates', {exact:true})) await page.waitForTimeout(1200);
  await shot(page, 'miStudio_Template_Panel-Browse-Training_Templates.jpg');
  if (await click(page, 'Extraction Templates', {exact:true})) await page.waitForTimeout(1200);
  await shot(page, 'miStudio_Template_Panel-Browse-Extraction_Templates.jpg');
  if (await click(page, 'Labeling Templates', {exact:true})) await page.waitForTimeout(1200);
  await shot(page, 'miStudio_Template_Panel-Browse-Labeling_Templates.jpg');
  await shot(page, 'miStudio_Templates_Labeling-Context_Aware.jpg', 'labeling tab');
  await shot(page, 'miStudio_Templates_Labeling-Context_Aware_Preview.jpg', 'labeling tab dup');
  if (await click(page, 'Steering Prompts', {exact:true})) await page.waitForTimeout(1000);
  if (await click(page, 'Create New')) await page.waitForTimeout(1400);
  await shot(page, 'miStudio_Template_Panel-CreateSteeringPromptTemplate.jpg', 'create form'); await esc(page);

  // TRAINING
  await nav(page, 'Training');
  await shot(page, 'miStudio_Training_Panel-Browse.jpg');
  if (await click(page, 'New Training') || await click(page, 'Start Training') || await click(page, 'Configure')) await page.waitForTimeout(1500);
  await shot(page, 'miStudio_Training_Panel-Config-ModelChoice.jpg', 'cfg model');
  await page.mouse.wheel(0, 650); await page.waitForTimeout(600);
  await shot(page, 'miStudio_Training_Panel-Config-SAEChoice.jpg', 'cfg sae');
  await page.mouse.wheel(0, 650); await page.waitForTimeout(600);
  await shot(page, 'miStudio_Training_Panel-Config-HyperParameters.jpg', 'cfg hyperparams'); await esc(page);

  // EXTRACTIONS
  await nav(page, 'Extractions');
  await shot(page, 'miStudio_Extraction_Panel-JobBrowser.jpg', 'job browser');
  await shot(page, 'miStudio_Extraction_Panel-Browse.jpg', 'browse dup');
  if (await click(page, 'Start Extraction')) await page.waitForTimeout(1500);
  await shot(page, 'miStudio_Extraction_Panel-FeatureExtractionJobConfig_01.jpg', 'feat cfg 1');
  await page.mouse.wheel(0, 650); await page.waitForTimeout(600);
  await shot(page, 'miStudio_Extraction_Panel-FeatureExtractionJobConfig_02.jpg', 'feat cfg 2');
  await shot(page, 'miStudio_Extraction_Panel-ActivationExtractionJobConfig_01.jpg', 'activation cfg'); await esc(page);
  // Label Features modal = the label-config panel
  await nav(page, 'Extractions');
  if (await click(page, 'Label Features')) { await page.waitForTimeout(2000);
    await shot(page, 'miStudio_Extraction_Panel-FeatureLabelConfigPanel.jpg', 'label cfg panel'); await esc(page);
  } else await shot(page, 'miStudio_Extraction_Panel-FeatureLabelConfigPanel.jpg', 'no-btn fallback');

  // LABELING (results live in Completed tab)
  await nav(page, 'Labeling');
  if (await click(page, 'Completed', {exact:true})) await page.waitForTimeout(1500);
  await shot(page, 'miStudio_Labeling_Panel-LabelingJobResultsPanelBrowser.jpg', 'completed results');
  // open a completed job to see its results / feature list
  if (await click(page, 'View') || await click(page, 'Results') || await click(page, 'Details')) await page.waitForTimeout(2000);
  await shot(page, 'miStudio_Labeling_Panel-LabelingJobProgressResults.jpg', 'job results');
  await shot(page, 'miStudio_Feature_List-Star_Colors.jpg', 'feature list (results view)');
  // a feature detail modal
  if (await click(page, 'Feature') || await click(page, '#0') || await click(page, 'View')) await page.waitForTimeout(1500);
  await shot(page, 'miStudio_Feature_Modal-Enhanced_Label_Button.jpg', 'feature modal');
  await shot(page, 'miStudio_Feature_Modal-Enhanced_Labeling_Queued.jpg', 'modal best-effort');
  await shot(page, 'miStudio_Feature_Modal-Enhanced_Labeling_Completed.jpg', 'modal best-effort'); await esc(page);
  // start dialog
  await nav(page, 'Labeling');
  if (await click(page, 'Start Labeling') || await click(page, 'New Labeling') || await click(page, 'Start')) await page.waitForTimeout(1200);
  await shot(page, 'miStudio_Labeling-Start_Dialog.jpg', 'start dialog'); await esc(page);

  // CLUSTERS (= Feature Groups)
  await nav(page, 'Clusters');
  await shot(page, 'miStudio_FeatureGroups_Panel-Browse.jpg', 'clusters browse');
  if (await click(page, 'Expand') || await click(page, 'View') || await click(page, 'Members')) await page.waitForTimeout(1500);
  await shot(page, 'miStudio_FeatureGroups_Panel-ExpandedGroup.jpg', 'expanded');

  // STEERING (select an SAE to populate config)
  await nav(page, 'Steering');
  await page.waitForTimeout(1000);
  try {
    const sel = page.locator('select').first();
    const n = await sel.locator('option').count();
    if (n > 1) { await sel.selectOption({ index: 1 }); await page.waitForTimeout(2800); }
  } catch(e){ log('sae-sel', e.message.slice(0,50)); }
  await shot(page, 'miStudio_Steering_Panel-Config.jpg', 'config+SAE');
  await page.mouse.wheel(0, 1100); await page.waitForTimeout(800);
  await shot(page, 'miStudio_Steering_Panel-SessionResults.jpg', 'results scrolled');

  // MONITOR
  await nav(page, 'Monitor');
  await page.waitForTimeout(1500);
  await shot(page, 'miStudio_Monitor_Panel.jpg');
  await page.mouse.wheel(0, 600); await page.waitForTimeout(600);
  await shot(page, 'miStudio_Monitor_Panel-Resources.jpg', 'resources');

  // SETTINGS (verified tabs: Endpoints/API Keys/Labeling/Display)
  await nav(page, 'Settings');
  await page.waitForTimeout(1000);
  if (await click(page, 'API Keys', {exact:true})) await page.waitForTimeout(1000);
  await shot(page, 'miStudio_Settings_APIKeys-Saved.jpg', 'apikeys');
  if (await click(page, 'Edit')) { await page.waitForTimeout(800);
    await shot(page, 'miStudio_Settings_APIKeys-Edit.jpg', 'apikeys edit'); await esc(page);
  } else await shot(page, 'miStudio_Settings_APIKeys-Edit.jpg', 'no-edit fallback');
  if (await click(page, 'Labeling', {exact:true})) await page.waitForTimeout(1000);
  await shot(page, 'miStudio_Settings_Labeling-Enhanced_OpenAI.jpg', 'labeling settings');
  await shot(page, 'miStudio_Settings_Labeling-Enhanced_Method_Dropdown.jpg', 'method (same tab)');
  await shot(page, 'miStudio_Settings_Labeling-Enhanced_Model_Dropdown.jpg', 'model (same tab)');

  log('\n=== SUMMARY ===');
  const thin = results.filter(r => !r.ok);
  log(`total ${results.length} | ok ${results.length - thin.length} | thin ${thin.length}`);
  thin.forEach(r => log('  THIN:', r.name));
  fs.writeFileSync(`${OUT}/_manifest.json`, JSON.stringify(results, null, 1));
  await browser.close();
})();
