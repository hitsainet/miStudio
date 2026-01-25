"""
Neuronpedia Local Push Service.

This service handles direct pushing of SAE features to a local Neuronpedia instance
via PostgreSQL connection. Unlike the export service which creates ZIP files for
upload, this service writes directly to the Neuronpedia database.

The local Neuronpedia instance should be deployed separately (e.g., in its own K8s namespace)
with its own PostgreSQL database.

Data mapping:
- miStudio Training/ExternalSAE → Neuronpedia Model + SourceSet + Source
- miStudio Feature → Neuronpedia Neuron
- miStudio FeatureActivation → Neuronpedia Activation
- miStudio Feature.name/description → Neuronpedia Explanation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import asyncpg
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..models.external_sae import ExternalSAE
from ..models.feature import Feature
from ..models.feature_activation import FeatureActivation
from ..models.feature_dashboard import FeatureDashboardData
from ..models.model import Model
from ..models.training import Training
from .logit_lens_service import get_logit_lens_service, LogitLensResult
from .histogram_service import get_histogram_service, HistogramData

logger = logging.getLogger(__name__)


@dataclass
class LocalPushConfig:
    """Configuration for local Neuronpedia push."""
    # What to include
    include_activations: bool = True
    include_explanations: bool = True
    include_statistics: bool = True

    # Limits
    max_activations_per_feature: int = 20

    # Feature selection
    feature_indices: Optional[List[int]] = None

    # Visibility: 'PUBLIC' (discoverable) or 'UNLISTED' (direct link only)
    visibility: str = "PUBLIC"

    # Dashboard data computation
    compute_dashboard_data: bool = True  # Compute logit lens + histograms if not cached
    logit_lens_k: int = 20  # Top-k tokens for logit lens


@dataclass
class LocalPushResult:
    """Result of a local push operation."""
    success: bool
    model_id: str
    source_id: str
    neurons_created: int
    activations_created: int
    explanations_created: int
    neuronpedia_url: Optional[str] = None
    error_message: Optional[str] = None


class NeuronpediaLocalClient:
    """
    Client for direct PostgreSQL connection to local Neuronpedia instance.

    This client bypasses the Neuronpedia API and writes directly to the database,
    which is faster and doesn't require API authentication for local deployments.
    """

    def __init__(self, db_url: Optional[str] = None):
        """Initialize with database URL."""
        self.db_url = db_url or settings.neuronpedia_local_db_url
        if not self.db_url:
            raise ValueError("Neuronpedia local database URL not configured")
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish database connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.db_url,
                min_size=1,
                max_size=5,
            )
            logger.info("Connected to Neuronpedia database")

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from Neuronpedia database")

    async def ensure_admin_user(self, user_id: str = None) -> str:
        """Ensure the admin user exists, create if needed.

        The user_id must be in Neuronpedia's PUBLIC_ACTIVATIONS_USER_IDS list
        for activations to be visible in the UI. The default value is the
        SAELens creator ID which is pre-approved in Neuronpedia.

        Args:
            user_id: The user ID to use (must be a pre-approved Neuronpedia ID).
                    Defaults to settings.neuronpedia_local_admin_user_id.

        Returns:
            The user ID (same as input, for use as creator_id in records).
        """
        user_id = user_id or settings.neuronpedia_local_admin_user_id

        async with self._pool.acquire() as conn:
            # Check if user exists by ID (not by name)
            user = await conn.fetchrow(
                'SELECT id FROM "User" WHERE id = $1',
                user_id
            )

            if not user:
                # Create admin user with the approved ID
                # Important: Use the approved ID directly, not a random UUID
                email_unsubscribe_code = str(uuid4())  # Required NOT NULL field
                await conn.execute(
                    '''
                    INSERT INTO "User" (id, name, "emailUnsubscribeCode", admin, bot, "createdAt")
                    VALUES ($1, $2, $3, true, true, $4)
                    ''',
                    user_id, "neuronpedia-saelens", email_unsubscribe_code, datetime.utcnow()
                )
                logger.info(f"Created Neuronpedia admin user with approved ID: {user_id}")

            return user_id

    async def create_model(
        self,
        model_id: str,
        display_name: str,
        layers: int,
        creator_id: str,
        visibility: str = "PUBLIC",
    ) -> bool:
        """Create a Model record if it doesn't exist."""
        async with self._pool.acquire() as conn:
            # Check if model exists
            existing = await conn.fetchrow(
                'SELECT id FROM "Model" WHERE id = $1',
                model_id
            )

            if existing:
                # Update visibility if model exists
                await conn.execute(
                    'UPDATE "Model" SET visibility = $1, "updatedAt" = $2 WHERE id = $3',
                    visibility, datetime.utcnow(), model_id
                )
                logger.info(f"Model {model_id} already exists, updated visibility to {visibility}")
                return False

            # Create model
            await conn.execute(
                '''
                INSERT INTO "Model" (
                    id, "displayName", "displayNameShort", "creatorId",
                    layers, "neuronsPerLayer", owner, visibility,
                    "inferenceEnabled", instruct, "createdAt", "updatedAt"
                )
                VALUES ($1, $2, $2, $3, $4, 0, 'miStudio', $5, false, false, $6, $6)
                ''',
                model_id, display_name, creator_id, layers, visibility, datetime.utcnow()
            )
            logger.info(f"Created Neuronpedia model: {model_id} with visibility={visibility}")
            return True

    async def create_source_set(
        self,
        model_id: str,
        name: str,
        description: str,
        creator_id: str,
        visibility: str = "PUBLIC",
        neuron_count: int = 0,
    ) -> bool:
        """Create a SourceSet record if it doesn't exist."""
        async with self._pool.acquire() as conn:
            # Check if source set exists
            existing = await conn.fetchrow(
                'SELECT name FROM "SourceSet" WHERE "modelId" = $1 AND name = $2',
                model_id, name
            )

            if existing:
                # Update visibility if exists (neuronCount may not exist in schema)
                await conn.execute(
                    'UPDATE "SourceSet" SET visibility = $1 WHERE "modelId" = $2 AND name = $3',
                    visibility, model_id, name
                )
                logger.info(f"SourceSet {name} already exists, updated visibility to {visibility}")
                return False

            # Create source set (neuronCount not in Neuronpedia schema)
            await conn.execute(
                '''
                INSERT INTO "SourceSet" (
                    "modelId", name, description, type, "creatorName", "creatorId",
                    visibility, "hasDashboards", "allowInferenceSearch", urls, "createdAt"
                )
                VALUES ($1, $2, $3, 'sae', 'miStudio', $4, $5, true, false, ARRAY[]::text[], $6)
                ''',
                model_id, name, description, creator_id, visibility, datetime.utcnow()
            )
            logger.info(f"Created Neuronpedia source set: {name} with visibility={visibility}")
            return True

    async def create_source(
        self,
        source_id: str,
        model_id: str,
        set_name: str,
        creator_id: str,
        visibility: str = "PUBLIC",
    ) -> bool:
        """Create a Source record if it doesn't exist."""
        async with self._pool.acquire() as conn:
            # Check if source exists
            existing = await conn.fetchrow(
                'SELECT id FROM "Source" WHERE id = $1 AND "modelId" = $2',
                source_id, model_id
            )

            if existing:
                # Update visibility if exists
                await conn.execute(
                    'UPDATE "Source" SET visibility = $1 WHERE id = $2 AND "modelId" = $3',
                    visibility, source_id, model_id
                )
                logger.info(f"Source {source_id} already exists, updated visibility to {visibility}")
                return False

            # Create source
            await conn.execute(
                '''
                INSERT INTO "Source" (
                    id, "modelId", "setName", "creatorId",
                    visibility, "hasDashboards", "inferenceEnabled", "createdAt"
                )
                VALUES ($1, $2, $3, $4, $5, true, false, $6)
                ''',
                source_id, model_id, set_name, creator_id, visibility, datetime.utcnow()
            )
            logger.info(f"Created Neuronpedia source: {source_id} with visibility={visibility}")
            return True

    async def upsert_neuron(
        self,
        model_id: str,
        layer: str,
        index: str,
        creator_id: str,
        source_set_name: str,
        pos_str: List[str],
        pos_values: List[float],
        neg_str: List[str],
        neg_values: List[float],
        frac_nonzero: float,
        freq_hist_heights: List[float],
        freq_hist_values: List[float],
        max_act_approx: float = 0.0,
    ) -> bool:
        """Create or update a Neuron record."""
        async with self._pool.acquire() as conn:
            # Upsert neuron
            # Note: layer is the Source.id (e.g., "14-res-8k")
            # sourceSetName is the SourceSet.name (e.g., "res-8k")
            await conn.execute(
                '''
                INSERT INTO "Neuron" (
                    "modelId", layer, index, "creatorId", "sourceSetName",
                    "pos_str", "pos_values", "neg_str", "neg_values",
                    "frac_nonzero", "freq_hist_data_bar_heights", "freq_hist_data_bar_values",
                    "maxActApprox", "createdAt"
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT ("modelId", layer, index) DO UPDATE SET
                    "pos_str" = EXCLUDED."pos_str",
                    "pos_values" = EXCLUDED."pos_values",
                    "neg_str" = EXCLUDED."neg_str",
                    "neg_values" = EXCLUDED."neg_values",
                    "frac_nonzero" = EXCLUDED."frac_nonzero",
                    "freq_hist_data_bar_heights" = EXCLUDED."freq_hist_data_bar_heights",
                    "freq_hist_data_bar_values" = EXCLUDED."freq_hist_data_bar_values",
                    "maxActApprox" = EXCLUDED."maxActApprox"
                ''',
                model_id, layer, index, creator_id, source_set_name,
                pos_str, pos_values, neg_str, neg_values,
                frac_nonzero, freq_hist_heights, freq_hist_values,
                max_act_approx, datetime.utcnow()
            )
            return True

    async def create_activation(
        self,
        model_id: str,
        layer: str,
        index: str,
        creator_id: str,
        tokens: List[str],
        values: List[float],
        max_value: float,
        max_value_token_index: int,
    ) -> str:
        """Create an Activation record."""
        activation_id = str(uuid4())[:25]
        min_value = min(values) if values else 0.0

        async with self._pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO "Activation" (
                    id, "modelId", layer, index, "creatorId",
                    tokens, "values", "maxValue", "maxValueTokenIndex", "minValue",
                    "dfaValues", "lossValues", "createdAt"
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, ARRAY[]::double precision[], ARRAY[]::double precision[], $11)
                ''',
                activation_id, model_id, layer, index, creator_id,
                tokens, values, max_value, max_value_token_index, min_value,
                datetime.utcnow()
            )

        return activation_id

    async def create_explanation(
        self,
        model_id: str,
        layer: str,
        index: str,
        description: str,
        author_id: str,
        score: float = 0.0,
    ) -> str:
        """Create an Explanation record."""
        explanation_id = str(uuid4())[:25]

        async with self._pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO "Explanation" (
                    id, "modelId", layer, index, description, "authorId",
                    "scoreV1", "createdAt"
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT DO NOTHING
                ''',
                explanation_id, model_id, layer, index, description, author_id,
                score, datetime.utcnow()
            )

        return explanation_id

    async def delete_source_data(
        self,
        model_id: str,
        layer: str,
    ) -> int:
        """Delete all data for a source (neurons, activations, explanations)."""
        async with self._pool.acquire() as conn:
            # Delete in order due to foreign keys
            # Explanations reference Neurons
            await conn.execute(
                'DELETE FROM "Explanation" WHERE "modelId" = $1 AND layer = $2',
                model_id, layer
            )

            # Activations reference Neurons
            await conn.execute(
                'DELETE FROM "Activation" WHERE "modelId" = $1 AND layer = $2',
                model_id, layer
            )

            # Neurons
            result = await conn.execute(
                'DELETE FROM "Neuron" WHERE "modelId" = $1 AND layer = $2',
                model_id, layer
            )

            # Parse count from result
            count = int(result.split()[-1]) if result else 0
            logger.info(f"Deleted {count} neurons for {model_id}/{layer}")
            return count


class NeuronpediaLocalPushService:
    """
    Service for pushing SAE features to a local Neuronpedia instance.

    Coordinates the full push process:
    1. Creates Model, SourceSet, Source records
    2. Converts Features to Neurons
    3. Converts FeatureActivations to Activations
    4. Creates Explanations from Feature names
    """

    def __init__(self):
        """Initialize the push service."""
        self._client: Optional[NeuronpediaLocalClient] = None

    async def _get_client(self) -> NeuronpediaLocalClient:
        """Get or create the database client."""
        if self._client is None:
            self._client = NeuronpediaLocalClient()
            await self._client.connect()
        return self._client

    async def close(self) -> None:
        """Close the database connection."""
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def _compute_dashboard_data_if_needed(
        self,
        db: AsyncSession,
        sae: ExternalSAE,
        config: LocalPushConfig,
        progress_callback: Optional[callable] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute dashboard data (logit lens, histograms) if not already cached.

        Returns:
            Dictionary mapping feature index to dashboard data dict with keys:
            - logit_lens: LogitLensResult or None
            - histogram: HistogramData or None
        """
        dashboard_data: Dict[int, Dict[str, Any]] = {}

        if not config.compute_dashboard_data:
            return dashboard_data

        # Get feature indices to compute
        feature_indices = config.feature_indices
        if not feature_indices:
            if sae.n_features:
                feature_indices = list(range(sae.n_features))
            else:
                # Get from database
                stmt = select(Feature.neuron_index).where(Feature.external_sae_id == sae.id)
                result = await db.execute(stmt)
                feature_indices = [row[0] for row in result.fetchall()]

        if not feature_indices:
            logger.warning("No feature indices to compute dashboard data for")
            return dashboard_data

        total = len(feature_indices)
        logger.info(f"Computing dashboard data for {total} features...")

        # 1. Compute logit lens data
        try:
            if progress_callback:
                progress_callback(5, "Computing logit lens data...")

            logit_lens_service = get_logit_lens_service()

            def logit_progress(completed: int, total: int, msg: str):
                if progress_callback:
                    # Scale logit lens progress from 5-40%
                    pct = 5 + int((completed / max(total, 1)) * 35)
                    progress_callback(pct, msg)

            logit_results = await logit_lens_service.compute_logit_lens_for_sae(
                db=db,
                sae_id=sae.id,
                feature_indices=feature_indices,
                k=config.logit_lens_k,
                progress_callback=logit_progress,
                force_recompute=False,
            )

            # Save logit lens results to database
            await logit_lens_service.save_logit_lens_results(db, sae.id, logit_results)

            # Add to dashboard data
            for idx, result in logit_results.items():
                if idx not in dashboard_data:
                    dashboard_data[idx] = {}
                dashboard_data[idx]["logit_lens"] = result

            logger.info(f"Computed logit lens data for {len(logit_results)} features")

        except Exception as e:
            logger.warning(f"Failed to compute logit lens data: {e}")
            # Continue without logit lens data

        # 2. Compute histogram data
        try:
            if progress_callback:
                progress_callback(40, "Computing histogram data...")

            histogram_service = get_histogram_service()

            # Create progress callback that maps histogram progress (0-100%) to overall (40-50%)
            def histogram_progress_callback(completed: int, total: int, message: str):
                if progress_callback and total > 0:
                    # Map to 40-50% range
                    pct = 40 + int((completed / total) * 10)
                    progress_callback(pct, f"Computing histograms... ({completed}/{total})")

            histogram_results = await histogram_service.compute_histograms_for_sae(
                db=db,
                sae_id=sae.id,
                n_bins=50,
                log_scale=True,
                progress_callback=histogram_progress_callback,
                force_recompute=False,
            )

            # Save histogram results to database
            await histogram_service.save_histogram_results(db, sae.id, histogram_results)

            # Add to dashboard data
            for idx, result in histogram_results.items():
                if idx not in dashboard_data:
                    dashboard_data[idx] = {}
                dashboard_data[idx]["histogram"] = result

            logger.info(f"Computed histogram data for {len(histogram_results)} features")

        except Exception as e:
            logger.warning(f"Failed to compute histogram data: {e}")
            # Continue without histogram data

        if progress_callback:
            progress_callback(50, "Dashboard data computation complete")

        return dashboard_data

    def _generate_model_id(self, model_name: str) -> str:
        """Generate a Neuronpedia-compatible model ID."""
        # Convert to lowercase, replace spaces with hyphens
        model_id = model_name.lower().replace(" ", "-").replace("/", "-")
        # Remove any characters that aren't alphanumeric, hyphen, or period
        model_id = "".join(c for c in model_id if c.isalnum() or c in "-.")
        return model_id

    def _generate_source_id(self, layer: int, source_set_name: str) -> str:
        """Generate a source ID like '0-res-16k' from layer and source set name."""
        return f"{layer}-{source_set_name}"

    def _generate_source_set_name(self, n_features: int) -> str:
        """Generate a source set name like 'res-16k' from feature count."""
        k_features = n_features // 1000
        return f"res-{k_features}k"

    async def push_sae_to_local(
        self,
        db: AsyncSession,
        sae_id: str,
        config: Optional[LocalPushConfig] = None,
        progress_callback: Optional[callable] = None,
    ) -> LocalPushResult:
        """
        Push an SAE's features to local Neuronpedia.

        Args:
            db: Database session
            sae_id: ID of the ExternalSAE to push
            config: Push configuration
            progress_callback: Optional callback(progress, message) for updates

        Returns:
            LocalPushResult with push details
        """
        if config is None:
            config = LocalPushConfig()

        try:
            client = await self._get_client()

            # Load SAE
            sae = await db.get(ExternalSAE, sae_id)
            if not sae:
                return LocalPushResult(
                    success=False,
                    model_id="",
                    source_id="",
                    neurons_created=0,
                    activations_created=0,
                    explanations_created=0,
                    error_message=f"SAE not found: {sae_id}"
                )

            # Determine model name by looking up the Model record
            model_name = sae.model_name  # May be None
            if not model_name and sae.model_id:
                # Look up model name from Model table
                model_record = await db.get(Model, sae.model_id)
                if model_record:
                    model_name = model_record.name

            # Fallback to parsing from SAE name if still not found
            if not model_name and sae.name:
                # SAE names are like "SAE from Phi-4-mini-instruct (L10-residual)"
                if sae.name.startswith("SAE from "):
                    # Extract model name between "SAE from " and " ("
                    name_part = sae.name[9:]  # Remove "SAE from "
                    if " (" in name_part:
                        model_name = name_part.split(" (")[0]
                    else:
                        model_name = name_part

            # Final fallback
            if not model_name:
                model_name = "unknown-model"

            layer = sae.layer or 0
            n_features = sae.n_features or 16384

            # Generate IDs
            # Source set name is like "res-8k" (without layer number)
            # Source ID is like "14-res-8k" (layer-sourceset_name)
            np_model_id = self._generate_model_id(model_name)
            np_source_set_name = self._generate_source_set_name(n_features)
            np_source_id = self._generate_source_id(layer, np_source_set_name)

            if progress_callback:
                progress_callback(5, f"Creating model: {np_model_id}")

            # Ensure admin user exists
            creator_id = await client.ensure_admin_user()

            # Create Model
            await client.create_model(
                model_id=np_model_id,
                display_name=model_name,
                layers=layer + 1,  # At least this many layers
                creator_id=creator_id,
                visibility=config.visibility,
            )

            if progress_callback:
                progress_callback(10, f"Creating source set: {np_source_set_name}")

            # Create SourceSet
            await client.create_source_set(
                model_id=np_model_id,
                name=np_source_set_name,
                description=f"SAE from miStudio - {sae.name}",
                creator_id=creator_id,
                visibility=config.visibility,
                neuron_count=n_features,
            )

            # Create Source
            await client.create_source(
                source_id=np_source_id,
                model_id=np_model_id,
                set_name=np_source_set_name,
                creator_id=creator_id,
                visibility=config.visibility,
            )

            # Compute dashboard data (logit lens, histograms) if needed
            computed_dashboard_data: Dict[int, Dict[str, Any]] = {}
            if config.compute_dashboard_data:
                if progress_callback:
                    progress_callback(15, "Computing dashboard data (logit lens, histograms)...")

                try:
                    computed_dashboard_data = await self._compute_dashboard_data_if_needed(
                        db=db,
                        sae=sae,
                        config=config,
                        progress_callback=progress_callback,
                    )
                except Exception as e:
                    logger.warning(f"Dashboard data computation failed, continuing without it: {e}")

            if progress_callback:
                progress_callback(50, "Loading features...")

            # Load features
            features = await self._load_features(db, sae, config)
            total_features = len(features)

            if total_features == 0:
                return LocalPushResult(
                    success=False,
                    model_id=np_model_id,
                    source_id=np_source_id,
                    neurons_created=0,
                    activations_created=0,
                    explanations_created=0,
                    error_message="No features found for SAE"
                )

            if progress_callback:
                progress_callback(20, f"Pushing {total_features} features...")

            neurons_created = 0
            activations_created = 0
            explanations_created = 0

            # Process features in batches
            for i, feature in enumerate(features):
                # Update progress every 10 features
                if progress_callback and i % 10 == 0:
                    pct = 50 + int((i / total_features) * 45)
                    progress_callback(pct, f"Processing feature {i+1}/{total_features}")

                # Extract statistics from computed dashboard data or database
                pos_str, pos_values = [], []
                neg_str, neg_values = [], []
                freq_hist_heights, freq_hist_values = [], []
                frac_nonzero = feature.activation_frequency or 0.0

                # First check computed dashboard data (from memory)
                feature_idx = feature.neuron_index
                if feature_idx in computed_dashboard_data:
                    computed = computed_dashboard_data[feature_idx]

                    # Extract logit lens data
                    if "logit_lens" in computed and computed["logit_lens"]:
                        logit_result = computed["logit_lens"]
                        # LogitLensResult has top_positive and top_negative lists
                        if hasattr(logit_result, 'top_positive'):
                            pos_str = [t["token"] for t in logit_result.top_positive[:10]]
                            pos_values = [t["logit"] for t in logit_result.top_positive[:10]]
                        if hasattr(logit_result, 'top_negative'):
                            neg_str = [t["token"] for t in logit_result.top_negative[:10]]
                            neg_values = [t["logit"] for t in logit_result.top_negative[:10]]

                    # Extract histogram data
                    if "histogram" in computed and computed["histogram"]:
                        hist_result = computed["histogram"]
                        if hasattr(hist_result, 'counts'):
                            freq_hist_heights = hist_result.counts
                            freq_hist_values = hist_result.bin_edges
                        if hasattr(hist_result, 'nonzero_count') and hasattr(hist_result, 'total_count'):
                            if hist_result.total_count > 0:
                                frac_nonzero = hist_result.nonzero_count / hist_result.total_count

                # Fall back to database lookup if no computed data
                if not pos_str:
                    dashboard_data = await self._load_dashboard_data(db, feature)
                    if dashboard_data and dashboard_data.logit_lens_data:
                        logit_data = dashboard_data.logit_lens_data
                        if isinstance(logit_data, dict):
                            # Handle both old and new field names
                            top_pos = logit_data.get("top_positive", [])
                            if top_pos:
                                pos_str = [t["token"] for t in top_pos[:10]]
                                pos_values = [t.get("logit", t.get("value", 0)) for t in top_pos[:10]]
                            top_neg = logit_data.get("top_negative", [])
                            if top_neg:
                                neg_str = [t["token"] for t in top_neg[:10]]
                                neg_values = [t.get("logit", t.get("value", 0)) for t in top_neg[:10]]

                    if dashboard_data and dashboard_data.histogram_data:
                        hist_data = dashboard_data.histogram_data
                        if isinstance(hist_data, dict):
                            freq_hist_heights = hist_data.get("counts", [])
                            freq_hist_values = hist_data.get("bin_edges", [])

                # Create neuron
                await client.upsert_neuron(
                    model_id=np_model_id,
                    layer=np_source_id,  # Source.id like "14-res-8k"
                    index=str(feature.neuron_index),
                    creator_id=creator_id,
                    source_set_name=np_source_set_name,  # SourceSet.name like "res-8k"
                    pos_str=pos_str,
                    pos_values=pos_values,
                    neg_str=neg_str,
                    neg_values=neg_values,
                    frac_nonzero=frac_nonzero,
                    freq_hist_heights=freq_hist_heights,
                    freq_hist_values=freq_hist_values,
                    max_act_approx=feature.max_activation or 0.0,
                )
                neurons_created += 1

                # Create activations
                if config.include_activations:
                    activations = await self._load_activations(
                        db, feature, config.max_activations_per_feature
                    )
                    for act in activations:
                        tokens = act.tokens or []
                        values = act.activations or []
                        if tokens and values:
                            max_val = max(values) if values else 0.0
                            max_idx = values.index(max_val) if values and max_val in values else 0

                            await client.create_activation(
                                model_id=np_model_id,
                                layer=np_source_id,
                                index=str(feature.neuron_index),
                                creator_id=creator_id,
                                tokens=tokens,
                                values=values,
                                max_value=max_val,
                                max_value_token_index=max_idx,
                            )
                            activations_created += 1

                # Create explanation
                if config.include_explanations and feature.name:
                    # Skip generic names like "feature_123"
                    if not feature.name.startswith("feature_"):
                        await client.create_explanation(
                            model_id=np_model_id,
                            layer=np_source_id,
                            index=str(feature.neuron_index),
                            description=feature.name,
                            author_id=creator_id,
                            score=feature.interpretability_score or 0.0,
                        )
                        explanations_created += 1

            if progress_callback:
                progress_callback(100, "Push complete")

            # Build Neuronpedia URL
            np_url = None
            if settings.neuronpedia_local_url:
                np_url = f"{settings.neuronpedia_local_url}/{np_model_id}/{np_source_id}"

            return LocalPushResult(
                success=True,
                model_id=np_model_id,
                source_id=np_source_id,
                neurons_created=neurons_created,
                activations_created=activations_created,
                explanations_created=explanations_created,
                neuronpedia_url=np_url,
            )

        except Exception as e:
            logger.exception(f"Failed to push SAE to local Neuronpedia: {e}")
            return LocalPushResult(
                success=False,
                model_id="",
                source_id="",
                neurons_created=0,
                activations_created=0,
                explanations_created=0,
                error_message=str(e)
            )

    async def _load_features(
        self,
        db: AsyncSession,
        sae: ExternalSAE,
        config: LocalPushConfig,
    ) -> List[Feature]:
        """Load features for the SAE."""
        from sqlalchemy import or_

        # Build query that checks both external_sae_id and training_id
        if sae.training_id:
            stmt = select(Feature).where(
                or_(
                    Feature.external_sae_id == sae.id,
                    Feature.training_id == sae.training_id
                )
            )
        else:
            stmt = select(Feature).where(Feature.external_sae_id == sae.id)

        if config.feature_indices:
            stmt = stmt.where(Feature.neuron_index.in_(config.feature_indices))

        stmt = stmt.order_by(Feature.neuron_index)

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def _load_dashboard_data(
        self,
        db: AsyncSession,
        feature: Feature,
    ) -> Optional[FeatureDashboardData]:
        """Load dashboard data for a feature."""
        stmt = select(FeatureDashboardData).where(
            FeatureDashboardData.feature_id == feature.id
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def _load_activations(
        self,
        db: AsyncSession,
        feature: Feature,
        limit: int,
    ) -> List[FeatureActivation]:
        """Load activations for a feature."""
        stmt = select(FeatureActivation).where(
            FeatureActivation.feature_id == feature.id
        ).order_by(FeatureActivation.max_activation.desc()).limit(limit)

        result = await db.execute(stmt)
        return list(result.scalars().all())


# Global service instance
_push_service: Optional[NeuronpediaLocalPushService] = None


def get_neuronpedia_local_push_service() -> NeuronpediaLocalPushService:
    """Get the global Neuronpedia local push service instance."""
    global _push_service
    if _push_service is None:
        _push_service = NeuronpediaLocalPushService()
    return _push_service
