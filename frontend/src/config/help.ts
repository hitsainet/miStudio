// In-app Help (ENH-004, operator decisions 2026-10-20). One topic per screen, plus three that
// cut across screens: the researcher's loop, how strong a finding is, and the words used here.
// This is the short summary a person needs on screen; the public manual (config/manual.ts) is the
// full reference, and each topic names the manual page (and section) that explains it in depth.
//
// Rules this file is held to by tests:
//   * manualLinks.test.tsx: every `manual` page exists in manual/docs and every "#section" is a
//     real heading on that page.
//   * helpPlainLanguage.test.ts: no internal ids or planning language, and every acronym is
//     spelled out in parentheses at its first use in a topic and defined on the manual's
//     glossary page.
//   * App.help.test.tsx: every panel maps to a topic (HELP_FOR_PANEL), and the header's Help
//     link opens the topic for the screen it is on.
//
// Origin: miDataworks frontend/src/config/help.ts @ 32cf406 (ENH-004).
import type { ActivePanel } from './panels';

export interface HelpTopic {
  id: string;
  title: string;
  /** Paragraphs; each is plain text. */
  body: string[];
  /** Short "how to" steps, in order. */
  steps?: string[];
  /** The manual page that explains this in depth, relative to the manual's base URL, with an
   *  optional "#section" (a heading slug on that page). */
  manual: string;
}

export const HELP_TOPICS: HelpTopic[] = [
  {
    id: 'getting-started',
    manual: 'getting-started/quickstart-tutorial',
    title: "Getting started: the researcher's loop",
    body: [
      'miStudio helps you look inside a language model. You train a sparse autoencoder (SAE), a small network that splits what one layer of the model is doing into many separate features, each of which tends to mean one thing. Then you find out what each feature means and test it by turning it up or down while the model writes.',
      'The screens on the left follow that loop from top to bottom. Each one has a Help link at the top of the page, beside the light and dark switch, that opens the matching topic here.',
    ],
    steps: [
      'Datasets: download some text and tokenize it (split it into the pieces the model reads).',
      'Models: download the model you want to study.',
      'Training: train a sparse autoencoder on one layer of that model, or download a ready-made one on the SAEs screen.',
      'Extractions: run the model over text and record which features fire on which words.',
      'Labeling: give each feature a short name that says what it responds to.',
      'Steering: turn a feature up or down while the model writes, to check that it does what its name says.',
    ],
  },
  {
    id: 'models',
    manual: 'core-workflow/data-model-management',
    title: 'Models',
    body: [
      'Models lists the language models you have downloaded from Hugging Face. Type a repository name such as google/gemma-2-2b, press Preview to see its size and layers, then Download.',
      'You can download a model in a smaller number format (quantization) so that it fits on your graphics card. Once downloaded, miStudio maps every layer of the model so the other screens can offer them by name.',
    ],
  },
  {
    id: 'datasets',
    manual: 'core-workflow/dataset-management',
    title: 'Datasets',
    body: [
      'Datasets is where text comes from. Download a dataset from Hugging Face, or upload your own files, and look inside it before you spend hours of graphics card time on it.',
      'A model does not read text directly; it reads tokens, and every model family splits text differently. So you tokenize a dataset once for each model you want to use it with. One download can carry several tokenizations.',
    ],
  },
  {
    id: 'saes',
    manual: 'advanced/external-saes',
    title: 'Sparse autoencoders (SAEs)',
    body: [
      'The SAEs screen lists every sparse autoencoder (SAE) you have: the ones you trained here, ones downloaded from Hugging Face, and ones imported from files.',
      'Each card shows the model and layer the sparse autoencoder reads, how many features it has, and where it came from. From a card you can start an extraction, open steering, or delete it.',
    ],
  },
  {
    id: 'templates',
    manual: 'advanced/templates',
    title: 'Templates',
    body: [
      'A template saves a set of choices so you can repeat an experiment exactly or share it. There are templates for extraction settings, for training settings, for the instructions given to the labeling model, and for the prompts used in steering.',
      'Every template can be created, edited, copied, exported to a file and imported again.',
    ],
  },
  {
    id: 'training',
    manual: 'core-workflow/sae-training#configuring-a-training-job',
    title: 'Training',
    body: [
      'Training builds a sparse autoencoder (SAE) for one or more layers of a model, using a dataset you have tokenized for that model. Choose the model and dataset, then the kind of sparse autoencoder, then its settings. Press the information icon beside a setting to read what it does.',
      'Progress, the reconstruction error and the number of features that never fire are shown live on the job card. Stop and Finalize ends a run early and still keeps a usable sparse autoencoder from it.',
    ],
  },
  {
    id: 'extractions',
    manual: 'core-workflow/feature-extraction',
    title: 'Extractions',
    body: [
      'An extraction runs the model over a mix of datasets and records, for every feature of a sparse autoencoder (SAE), the passages where it fires most strongly. Those passages are the evidence you will read when you name a feature.',
      'Features that almost never fire on the text you chose are dropped, so pick a mix of text that contains what you are looking for.',
    ],
  },
  {
    id: 'labeling',
    manual: 'core-workflow/auto-labeling',
    title: 'Labeling',
    body: [
      'Labeling gives each feature a short name and category by showing its strongest passages to a language model and asking what they have in common. You can label every feature of an extraction in one job, or study one feature at a time in more depth.',
      'Labeling can use a model on your own machines or a hosted service; set where it runs on the Settings screen. If a labeling job stops part way, it can resume from where it stopped.',
    ],
  },
  {
    id: 'clusters',
    manual: 'core-workflow/feature-groups',
    title: 'Clusters',
    body: [
      'A large extraction has tens of thousands of features, and related ones are scattered among them. Clusters groups features that respond most strongly to the same word, then splits each group by the context the word appears in.',
      'Pick a finished extraction and press Compute Index. Save a group you care about as a cluster profile, check it with steering, and export it to share.',
    ],
  },
  {
    id: 'circuits',
    manual: 'core-workflow/circuits#the-loop',
    title: 'Circuits',
    body: [
      'A circuit is a set of features at different layers that work together: an earlier feature that drives a later one. The Circuits screen finds candidate circuits and then tests them.',
      'Each circuit shows how strong its evidence is, from "found by counting" up to "tested by switching parts off". A candidate is never presented as proven.',
    ],
    steps: [
      'Capture: record which features fire at each chosen layer over a body of text. You see an estimate of the time and disk space first.',
      'Discovery: search the capture for pairs of features that fire together more often than chance.',
      'Validation: switch the earlier feature off and measure whether the later one changes.',
      'Promote a circuit you trust, then steer with it.',
    ],
  },
  {
    id: 'probes',
    manual: 'core-workflow/probe-monitors',
    title: 'Probes',
    body: [
      'A probe is a small detector that reads one layer of the model and answers one question, for example "is this text sarcastic?". It is cheap to run next to a model, so most of this screen is about measuring honestly how well it works.',
      'You give it labeled examples, it trains and is tested on text it has not seen, and each run shows its scores side by side so you can choose a threshold knowing what it costs. A probe you trust can be exported for miLLM to run.',
    ],
  },
  {
    id: 'judge-prompts',
    manual: 'core-workflow/probe-monitors#judge-prompts',
    title: 'Judge prompts',
    body: [
      'A judge prompt is the set of instructions a separate judge model follows when it labels the same rows a probe was tested on. Comparing the probe with the judge is the top step of the probe\'s evidence ladder, and it only counts when the judge and the probe are about the same behaviour.',
      'Each prompt names the behaviour it judges by its key, and says how the judge\'s reply is read: a rating on a scale, a yes or no, or one label from a list. A reply that cannot be read is never guessed; too many of them stop the run.',
      'Once a judge run has used a version of a prompt, that version is locked so the run always keeps the exact text it was measured with. To change it, start a new version.',
      'Prompts are data, not code. The two starter prompts are files loaded when the server starts, and you can add your own without any programming: here with New prompt or Import, through an agent, or by putting exported prompt files in the folder named by JUDGE_PROMPT_SEED_DIR and restarting the server.',
    ],
    steps: [
      'Open Probes, then the Judge prompts tab, and press New prompt, or copy an existing one.',
      'Write the instructions, choose what the judge is shown for each row, and choose how the answer is read.',
      'Pick a dataset under "Check it on real rows" to see exactly what the judge will receive, then press Try it to send a few rows to a judge model.',
      'Give your probe the same behaviour key, then start a judge run from the Judge tab with this prompt.',
    ],
  },
  {
    id: 'behaviours',
    manual: 'core-workflow/probe-monitors#behaviours',
    title: 'Behaviours',
    body: [
      'A behaviour is what a probe detects and what a judge prompt judges, named by a short key. Prompts, runs and probes already carry a behaviour key; the catalogue records what each key means.',
      'Cataloguing a key changes nothing on the records that use it: they are listed under it by their key. A key with no record is shown as uncatalogued, which is allowed.',
      'Each behaviour has a checklist from definition to armed in miLLM. Every step is read from real records and is never ticked by hand. "Unknown" means miLLM could not be asked, not that the step is missing.',
      'One probe can be primary. It must carry the same key, and its training data\'s positive labels must have been confirmed as this behaviour, by you on the Data list or by a miDataworks send that named it.',
    ],
    steps: [
      'Open Probes, then the Behaviours tab, and press New behaviour, or Catalogue this beside a key already in use.',
      'Write what counts as positive and negative, and the false-positive rate (FPR) its probes should be calibrated to.',
      'Follow the checklist: each step links to the screen where it is done.',
      'Once the primary probe is armed in miLLM, start a spot-check on the behaviour page: a judge model re-reads recorded conversation turns and the report shows where it and the probes disagree.',
    ],
  },
  {
    id: 'probe-families',
    manual: 'core-workflow/probe-monitors#probe-families',
    title: 'Probe families',
    body: [
      'A behaviour can have several probes on one model: probes at different layers, or specialists that each answer one sub-question of the behaviour. Together they are its family.',
      'The family section on a behaviour page lists every sub-question the current definition declares, with its probes or the word missing and the training run that would fill it. Each probe shows its facts, and a note where it differs from most of the family, for example in precision or in the data it was trained on.',
      'The checklist reports the whole family: a step reads partly done when some probes are exported or armed and others are not. The behaviour counts as monitoring only when its primary probe is armed.',
      'A blend report measures combinations of the family\'s probes the way an application would use them: fitted on data held back for that, set to the behaviour\'s false-positive rate (FPR) on its calibration data, and tested on unseen data. A blend has no rung of its own. The suggested blend is advice for an application and changes nothing that is served.',
    ],
    steps: [
      'Open the behaviour page and find Probe family. If the behaviour has probes on several models, choose one.',
      'Read the missing sub-questions at the top, and train a probe for each with the run it names.',
      'Read the notes beside each probe before you compare or combine them.',
    ],
  },
  {
    id: 'jlens',
    manual: 'core-workflow/jlens#what-a-readout-is',
    title: 'J-Lens',
    body: [
      'J-Lens shows, for each word of a prompt and each layer of the model, what the model is getting ready to say at that point. It needs no training to start: the basic view works on any model you have downloaded.',
      'A more accurate view needs a lens fitted for that exact model. Until one exists, that view is shown as unavailable with the reason. You can share a readout setup as a link.',
    ],
  },
  {
    id: 'steering',
    manual: 'core-workflow/steering#steering-modes',
    title: 'Steering',
    body: [
      'Steering turns chosen features up or down while the model writes, and shows the result next to the normal output. It is how you check that a feature does what its name says.',
      'Try one feature at several strengths, compare several features at the same strength, or apply them all together. A strength that is too high makes the text fall apart, so start low.',
    ],
  },
  {
    id: 'monitor',
    manual: 'getting-started/dashboard#real-time-system-monitoring',
    title: 'Monitor',
    body: [
      'Monitor shows how busy the machine is: memory and load on each graphics card, the processor and the disk, updated live. Use it to see whether there is room for another job.',
      'The graphics card summary at the top of every page opens this screen.',
    ],
  },
  {
    id: 'settings',
    manual: 'advanced/settings-reference',
    title: 'Settings',
    body: [
      'Settings holds the choices the rest of the app depends on: where the labeling model runs, the keys for hosted services and Hugging Face, labeling defaults, and display options.',
      'Settings are saved on the server, not in your browser, so they apply everywhere you use miStudio. Keys are stored encrypted.',
      'The Environment tab shows where each server setting comes from and lets you change a few of them while miStudio runs, such as the labeling batch size, the steering time limit and the log level. A saved value overrides the environment until you choose Use the environment value. Each change is kept in a history, and each setting says when a change takes effect. Settings that need a restart are listed with the part of miStudio to restart.',
    ],
    steps: [
      'Open Settings, then Environment (enter the Settings personal identification number (PIN) if one is set).',
      'Type the new value, add a reason if you like, and press Save.',
      'To go back, press Use the environment value.',
    ],
  },
  {
    id: 'evidence',
    manual: 'concepts/evidence-ladder',
    title: 'Evidence: how strong is a finding?',
    body: [
      'Every circuit and every link in one carries a grade of how strong its evidence is. There are four steps, and a circuit is only as strong as its weakest link.',
      'The word "causal" is only used once a link has been tested by switching something off. A failed test is kept on record and does not erase what was earned another way.',
    ],
    steps: [
      'Found: two features fire together more often than chance would explain.',
      'Supported: a second, independent measurement agrees.',
      'Causally tested: switching the earlier feature off changes the later one.',
      'Faithful: the whole circuit is shown to be needed, and enough, for the behaviour.',
    ],
  },
  {
    id: 'glossary',
    manual: 'concepts/glossary',
    title: 'Words used here',
    body: [
      'Feature: one direction a sparse autoencoder (SAE) learned, which tends to stand for one idea, such as "French text" or "a closing bracket".',
      "Layer: one step of the model's processing. Earlier layers tend to handle spelling and grammar, later ones meaning.",
      'Token: a piece of text the model reads, often part of a word.',
      'Activation: how strongly a feature responds to a token.',
      "Steering: changing a feature's strength while the model writes, to see what it does.",
      "The manual's glossary lists every term and abbreviation miStudio uses.",
    ],
  },
];

export const HELP_TOPIC_IDS = HELP_TOPICS.map(t => t.id);

/** The topic with this id, if there is one. */
export function helpTopic(id: string | null | undefined): HelpTopic | undefined {
  return id ? HELP_TOPICS.find(t => t.id === id) : undefined;
}

/**
 * The topic the header's Help link opens on each screen. A `Record` over the panel registry, so
 * adding a panel without a Help topic is a type error, not a silent gap.
 */
export const HELP_FOR_PANEL: Record<ActivePanel, string> = {
  models: 'models',
  datasets: 'datasets',
  training: 'training',
  extractions: 'extractions',
  labeling: 'labeling',
  'feature-groups': 'clusters',
  circuits: 'circuits',
  'probe-monitors': 'probes',
  jlens: 'jlens',
  templates: 'templates',
  saes: 'saes',
  steering: 'steering',
  system: 'monitor',
  settings: 'settings',
  help: 'getting-started',
};
