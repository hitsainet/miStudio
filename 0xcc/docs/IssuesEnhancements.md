Please analyze the following requirements and create a detailed todo list that when executed will accomplish them:

## Completed Enhancements ✅

✅ **Model Tile - Extraction Status Indicator (Swirl Clock)**
@0xcc/img/Model_Tile_GreySwirlClock.jpg >Tab>Models>ModelTile: The small swirl clock to the right of the "Extract Activations" button...when there are no completed "Extracted Activations" jobs, the swirl clock should be greyed, but when there is one or more completed "Extracted Activation" jobs for a model, the small swirl clock should be bright green.

✅ **Training Jobs - Filter Category Counts Bug**
@0xcc/img/TrainingJobsFilterCategoryCountsBroken.jpg Tab>Training>TrainingJobs>Job filter buttons: The totals for each category change based on what is on the screen rather than the total completed training jobs that fit each category despite what is visible

✅ **Features Table Pagination - "Go to" Functionality**
Tab>Extractions>Features Table Pagination: Add "Go to" functionality to navigate directly to a specific page/record in the features table. Currently showing "Showing 50 of 23308 features" with Previous/Next buttons, but no way to jump to a specific page or record number. Should add input field allowing user to enter either:
  - Page number (e.g., "Go to page: [___]")
  - OR Feature number/record number (e.g., "Go to feature: [___]")
This would be especially useful for large datasets with thousands of features where sequential pagination is slow.
**Status:** Implemented in FeaturesPanel.tsx with smart detection (page number if ≤ total pages, otherwise feature number)#

# Outstanding Enhancements

0xcc/img/DiscoveredFeaturesActionsStarFavorites.jpg I want to be able to toggle the stars under the "Actions" column to bright green when on, and have the toggle persist. The column should be changed to "Of Interest".

Please normalize the interface to that when zoomed to 100%, all text is comfortable to read, but is not too large either. Where it makes sense compress the width. Do this to most page elements by taking advantage of blank spaces. Focus especially on the various tiles.

@0xcc/img/Extraction_Layer(s).jpg shows "Layers: 20", but I think it should be "Layer(s): 20" to disambiguate that its the layer number rather than number of layers.

Tab>Labeling>Labeling Job Progress Tile: Add a scrolling results window to the labeling job progress tile that displays real-time results from the feature labeling process. This window should show recently labeled features as they are being processed, allowing users to monitor the quality and progress of the labeling in real-time. The window should auto-scroll to show the most recent results and display key information for each labeled feature:
  - Feature ID (e.g., #23319)
  - Label (the assigned feature name)
  - Category (e.g., semantic, syntactic, positional)
  - Description (if available)
  - Example tokens (top activating tokens that represent the feature)
The results window should display the last 10-20 labeled features in a compact, readable format, updating in real-time as new features are labeled.

**Multi-Layer Training: Per-Layer Metrics Tracking**
Tab>Training>Training Job Detail/Metrics: For multi-layer training jobs (e.g., training layers 7, 14, 18 simultaneously), track and display layer-wise differences explicitly. While using the same hyperparameters across all layers is appropriate, we need visibility into how each layer performs individually to determine if any layer needs different tuning later. Implement per-layer tracking and visualization for:
  - Average activation magnitude (per layer)
  - L0 sparsity (per layer)
  - Dead feature counts (per layer)
  - Loss (per layer)
This will reveal whether specific layers (e.g., layer 7 vs 14 vs 18) have different characteristics and may benefit from layer-specific hyperparameter tuning in future training runs. Display this as separate metrics lines/charts in the training progress UI, with a layer selector or overlaid charts showing all layers simultaneously for comparison.

