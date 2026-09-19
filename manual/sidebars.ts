import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  manualSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/introduction',
        'getting-started/quickstart-tutorial',
        'getting-started/installation',
        'getting-started/dashboard',
        {
          type: 'category',
          label: 'Automated Install (AI-assisted)',
          items: [
            'getting-started/install-guide-compose',
            'getting-started/install-guide-k8s',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Concepts',
      items: [
        'concepts/architecture',
        'concepts/interpretability-primer',
        'concepts/extraction-pipeline',
        'concepts/evidence-ladder',
        'concepts/tier-2.5',
      ],
    },
    {
      type: 'category',
      label: 'Core Workflow',
      items: [
        'core-workflow/researcher-journey',
        'core-workflow/dataset-management',
        'core-workflow/data-model-management',
        'core-workflow/sae-training',
        'core-workflow/training-lifecycle',
        'core-workflow/feature-extraction',
        'core-workflow/feature-groups',
        'core-workflow/auto-labeling',
        'core-workflow/enhanced-labeling',
        'core-workflow/labeling-trials',
        'core-workflow/steering',
        'core-workflow/circuits',
        'core-workflow/jlens',
      ],
    },
    {
      type: 'category',
      label: 'Advanced Usage',
      items: [
        'advanced/templates',
        'advanced/external-saes',
        'advanced/multi-gpu',
        'advanced/exporting',
        'advanced/multi-dataset',
        'advanced/mcp-server',
        'advanced/mcp-agent-instructions',
        'advanced/settings-reference',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        {
          type: 'category',
          label: 'REST API',
          items: [
            'reference/api/overview',
            'reference/api/datasets',
            'reference/api/models',
            'reference/api/trainings',
            'reference/api/circuits',
            'reference/api/saes',
            'reference/api/features-labeling',
            'reference/api/steering',
            'reference/api/neuronpedia',
            'reference/api/system',
            'reference/api/templates-settings',
          ],
        },
        'reference/websocket-channels',
        'reference/data-model',
      ],
    },
    'troubleshooting',
    'faq',
  ],
};

export default sidebars;
