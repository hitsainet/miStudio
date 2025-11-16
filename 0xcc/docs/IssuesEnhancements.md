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
**Status:** Implemented in FeaturesPanel.tsx with smart detection (page number if ≤ total pages, otherwise feature number)

✅ **Extraction Display - Layer(s) Label Clarification**
@0xcc/img/Extraction_Layer(s).jpg Tab>Models>Extraction History: Changed "Layers" label to "Layer(s)" to disambiguate that the display shows layer numbers (e.g., "7, 14, 18") rather than the count of layers.
**Status:** Updated in ActivationExtractionHistory.tsx line 302

✅ **Features Table - "Of Interest" Star Toggle**
@0xcc/img/DiscoveredFeaturesActionsStarFavorites.jpg Tab>Features>Features Table: Implemented star toggle feature under "Of Interest" column with bright green (emerald-500) color when toggled on. Toggle state persists via API calls to backend. Filter button also updated to use emerald colors.
**Status:** Updated in FeaturesPanel.tsx - star colors changed from yellow to emerald, tooltips updated, column already labeled "Of Interest"

✅ **Labeling Job Progress - Real-time Results Window**
Tab>Labeling>Labeling Job Progress Tile: Implemented scrolling results window that displays real-time results from the feature labeling process via WebSocket. Window shows recently labeled features with:
  - Feature ID (e.g., #23319)
  - Label (the assigned feature name)
  - Category (e.g., semantic, syntactic, positional) with color coding
  - Description (if available)
  - Example tokens (top activating tokens that represent the feature)
The results window displays the last 20 labeled features in a compact, readable format, with auto-scroll to show most recent results at the top.
**Status:** Fully implemented in LabelingResultsWindow.tsx component with WebSocket subscription via useLabelingResultsWebSocket hook

✅ **CPU Utilization Display - Per-Core Calculation**
Tab>Monitor>CPU Utilization: Updated CPU utilization display to work on a per-core basis where 100% = 1 full core utilized. On a 16-core system, maximum is 1600% (all cores at 100%). Backend now sums per-core percentages from psutil, and frontend progress bar scales based on core count.
**Status:** Implemented in [system_monitor_service.py:176-180](backend/src/services/system_monitor_service.py#L176-L180) and [SystemMonitor.tsx:181](frontend/src/components/SystemMonitor/SystemMonitor.tsx#L181)

# Outstanding Enhancements

Please normalize the interface to that when zoomed to 100%, all text is comfortable to read, but is not too large either. Where it makes sense compress the width. Do this to most page elements by taking advantage of blank spaces. Focus especially on the various tiles.

**Multi-Layer Training: Per-Layer Metrics Tracking**
Tab>Training>Training Job Detail/Metrics: For multi-layer training jobs (e.g., training layers 7, 14, 18 simultaneously), track and display layer-wise differences explicitly. While using the same hyperparameters across all layers is appropriate, we need visibility into how each layer performs individually to determine if any layer needs different tuning later. Implement per-layer tracking and visualization for:
  - Average activation magnitude (per layer)
  - L0 sparsity (per layer)
  - Dead feature counts (per layer)
  - Loss (per layer)
This will reveal whether specific layers (e.g., layer 7 vs 14 vs 18) have different characteristics and may benefit from layer-specific hyperparameter tuning in future training runs. Display this as separate metrics lines/charts in the training progress UI, with a layer selector or overlaid charts showing all layers simultaneously for comparison.

