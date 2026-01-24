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
from ..models.training import Training

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
        """Ensure the admin user exists, create if needed."""
        user_id = user_id or settings.neuronpedia_local_admin_user_id

        async with self._pool.acquire() as conn:
            # Check if user exists
            user = await conn.fetchrow(
                'SELECT id FROM "User" WHERE name = $1',
                user_id
            )

            if not user:
                # Create admin user
                new_id = str(uuid4())[:25]  # cuid-like
                email_unsubscribe_code = str(uuid4())  # Required NOT NULL field
                await conn.execute(
                    '''
                    INSERT INTO "User" (id, name, "emailUnsubscribeCode", admin, bot, "createdAt")
                    VALUES ($1, $2, $3, true, true, $4)
                    ''',
                    new_id, user_id, email_unsubscribe_code, datetime.utcnow()
                )
                logger.info(f"Created Neuronpedia admin user: {user_id}")
                return new_id

            return user['id']

    async def create_model(
        self,
        model_id: str,
        display_name: str,
        layers: int,
        creator_id: str,
    ) -> bool:
        """Create a Model record if it doesn't exist."""
        async with self._pool.acquire() as conn:
            # Check if model exists
            existing = await conn.fetchrow(
                'SELECT id FROM "Model" WHERE id = $1',
                model_id
            )

            if existing:
                logger.info(f"Model {model_id} already exists")
                return False

            # Create model
            await conn.execute(
                '''
                INSERT INTO "Model" (
                    id, "displayName", "displayNameShort", "creatorId",
                    layers, "neuronsPerLayer", owner, visibility,
                    "inferenceEnabled", instruct, "createdAt", "updatedAt"
                )
                VALUES ($1, $2, $2, $3, $4, 0, 'miStudio', 'UNLISTED', false, false, $5, $5)
                ''',
                model_id, display_name, creator_id, layers, datetime.utcnow()
            )
            logger.info(f"Created Neuronpedia model: {model_id}")
            return True

    async def create_source_set(
        self,
        model_id: str,
        name: str,
        description: str,
        creator_id: str,
    ) -> bool:
        """Create a SourceSet record if it doesn't exist."""
        async with self._pool.acquire() as conn:
            # Check if source set exists
            existing = await conn.fetchrow(
                'SELECT name FROM "SourceSet" WHERE "modelId" = $1 AND name = $2',
                model_id, name
            )

            if existing:
                logger.info(f"SourceSet {name} already exists for model {model_id}")
                return False

            # Create source set
            await conn.execute(
                '''
                INSERT INTO "SourceSet" (
                    "modelId", name, description, type, "creatorName", "creatorId",
                    visibility, "hasDashboards", "allowInferenceSearch", urls, "createdAt"
                )
                VALUES ($1, $2, $3, 'sae', 'miStudio', $4, 'UNLISTED', true, false, ARRAY[]::text[], $5)
                ''',
                model_id, name, description, creator_id, datetime.utcnow()
            )
            logger.info(f"Created Neuronpedia source set: {name}")
            return True

    async def create_source(
        self,
        source_id: str,
        model_id: str,
        set_name: str,
        creator_id: str,
    ) -> bool:
        """Create a Source record if it doesn't exist."""
        async with self._pool.acquire() as conn:
            # Check if source exists
            existing = await conn.fetchrow(
                'SELECT id FROM "Source" WHERE id = $1 AND "modelId" = $2',
                source_id, model_id
            )

            if existing:
                logger.info(f"Source {source_id} already exists")
                return False

            # Create source
            await conn.execute(
                '''
                INSERT INTO "Source" (
                    id, "modelId", "setName", "creatorId",
                    visibility, "hasDashboards", "inferenceEnabled", "createdAt"
                )
                VALUES ($1, $2, $3, $4, 'UNLISTED', true, false, $5)
                ''',
                source_id, model_id, set_name, creator_id, datetime.utcnow()
            )
            logger.info(f"Created Neuronpedia source: {source_id}")
            return True

    async def upsert_neuron(
        self,
        model_id: str,
        layer: str,
        index: str,
        creator_id: str,
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
                model_id, layer, index, creator_id, layer,  # sourceSetName = layer for SAEs
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

    def _generate_model_id(self, model_name: str) -> str:
        """Generate a Neuronpedia-compatible model ID."""
        # Convert to lowercase, replace spaces with hyphens
        model_id = model_name.lower().replace(" ", "-").replace("/", "-")
        # Remove any characters that aren't alphanumeric, hyphen, or period
        model_id = "".join(c for c in model_id if c.isalnum() or c in "-.")
        return model_id

    def _generate_source_id(self, layer: int, n_features: int) -> str:
        """Generate a source ID like '0-res-16k'."""
        k_features = n_features // 1000
        return f"{layer}-res-{k_features}k"

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

            # Determine model name
            model_name = sae.model_name or "unknown-model"
            layer = sae.layer or 0
            n_features = sae.n_features or 16384

            # Generate IDs
            np_model_id = self._generate_model_id(model_name)
            np_source_id = self._generate_source_id(layer, n_features)
            np_source_set_name = np_source_id

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
            )

            if progress_callback:
                progress_callback(10, f"Creating source set: {np_source_set_name}")

            # Create SourceSet
            await client.create_source_set(
                model_id=np_model_id,
                name=np_source_set_name,
                description=f"SAE from miStudio - {sae.name}",
                creator_id=creator_id,
            )

            # Create Source
            await client.create_source(
                source_id=np_source_id,
                model_id=np_model_id,
                set_name=np_source_set_name,
                creator_id=creator_id,
            )

            if progress_callback:
                progress_callback(15, "Loading features...")

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
                    pct = 20 + int((i / total_features) * 70)
                    progress_callback(pct, f"Processing feature {i+1}/{total_features}")

                # Load dashboard data for this feature
                dashboard_data = await self._load_dashboard_data(db, feature)

                # Extract statistics
                pos_str, pos_values = [], []
                neg_str, neg_values = [], []
                freq_hist_heights, freq_hist_values = [], []

                if dashboard_data and dashboard_data.logit_lens_data:
                    logit_data = dashboard_data.logit_lens_data
                    if isinstance(logit_data, dict):
                        pos_str = logit_data.get("top_promoted_tokens", [])[:10]
                        pos_values = logit_data.get("top_promoted_values", [])[:10]
                        neg_str = logit_data.get("top_suppressed_tokens", [])[:10]
                        neg_values = logit_data.get("top_suppressed_values", [])[:10]

                if dashboard_data and dashboard_data.histogram_data:
                    hist_data = dashboard_data.histogram_data
                    if isinstance(hist_data, dict):
                        freq_hist_heights = hist_data.get("counts", [])
                        freq_hist_values = hist_data.get("bin_edges", [])

                # Create neuron
                await client.upsert_neuron(
                    model_id=np_model_id,
                    layer=np_source_id,
                    index=str(feature.neuron_index),
                    creator_id=creator_id,
                    pos_str=pos_str,
                    pos_values=pos_values,
                    neg_str=neg_str,
                    neg_values=neg_values,
                    frac_nonzero=feature.activation_frequency or 0.0,
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
