"""
Neuronpedia Export Service.

This service orchestrates the export of SAEs and feature dashboard data to
Neuronpedia-compatible format. It coordinates:
1. Computing missing dashboard data (logit lens, histograms, top tokens)
2. Generating Neuronpedia JSON files
3. Creating SAELens-compatible cfg.json and weights
4. Packaging everything into a ZIP archive

The export process runs as a background Celery task with progress tracking.
"""

import asyncio
import json
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..core.config import settings
from ..models.external_sae import ExternalSAE, SAEStatus
from ..models.feature import Feature
from ..models.feature_activation import FeatureActivation
from ..models.feature_dashboard import FeatureDashboardData
from ..models.model import Model
from ..models.neuronpedia_export import NeuronpediaExportJob, ExportStatus
from ..ml.community_format import (
    CommunityStandardConfig,
    load_sae_auto_detect,
    save_sae_community_format,
)
from ..utils.transformerlens_mapping import (
    get_transformerlens_model_id,
    get_transformerlens_hook_name,
    build_neuronpedia_config,
)
from .logit_lens_service import get_logit_lens_service
from .histogram_service import get_histogram_service
from .token_aggregator_service import get_token_aggregator_service

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for Neuronpedia export."""
    # Feature selection
    feature_selection: str = "all"  # "all" or "extracted" or list of indices
    feature_indices: Optional[List[int]] = None

    # Dashboard data options
    include_logit_lens: bool = True
    include_histograms: bool = True
    include_top_tokens: bool = True

    # Dashboard data parameters
    logit_lens_k: int = 20
    histogram_bins: int = 50
    top_tokens_k: int = 50

    # SAE format options
    include_saelens_format: bool = True

    # Metadata
    include_explanations: bool = True


class NeuronpediaExportService:
    """
    Service for exporting SAEs to Neuronpedia-compatible format.

    Orchestrates the entire export process including:
    - Loading SAE and feature data
    - Computing dashboard data (logit lens, histograms, top tokens)
    - Generating JSON files
    - Creating ZIP archive
    """

    def __init__(self):
        """Initialize the export service."""
        self._exports_dir = Path(settings.data_dir) / "exports" / "neuronpedia"
        self._exports_dir.mkdir(parents=True, exist_ok=True)

    async def start_export(
        self,
        db: AsyncSession,
        sae_id: str,
        config: Optional[ExportConfig] = None,
    ) -> str:
        """
        Start a new export job.

        Args:
            db: Database session
            sae_id: ID of the SAE to export
            config: Export configuration

        Returns:
            Job ID for tracking progress
        """
        if config is None:
            config = ExportConfig()

        # Validate SAE exists and is ready
        sae = await db.get(ExternalSAE, sae_id)
        if not sae:
            raise ValueError(f"SAE not found: {sae_id}")
        if sae.status != SAEStatus.READY.value:
            raise ValueError(f"SAE is not ready: {sae.status}")

        # Check for existing features - try both external_sae_id and training_id
        feature_found = False

        # First try external_sae_id
        stmt = select(Feature).where(Feature.external_sae_id == sae_id).limit(1)
        result = await db.execute(stmt)
        if result.scalar_one_or_none():
            feature_found = True

        # If not found, try training_id (for trained SAEs)
        if not feature_found and sae.training_id:
            stmt = select(Feature).where(Feature.training_id == sae.training_id).limit(1)
            result = await db.execute(stmt)
            if result.scalar_one_or_none():
                feature_found = True

        # If still no features, allow export without features (just SAE weights)
        # This allows exporting SAEs that haven't had feature extraction run yet
        if not feature_found:
            logger.warning(f"No features found for SAE {sae_id}. Export will only include SAE weights.")

        # Create export job record
        job = NeuronpediaExportJob(
            sae_id=sae_id,
            source_type="external_sae",
            config={
                "feature_selection": config.feature_selection,
                "feature_indices": config.feature_indices,
                "include_logit_lens": config.include_logit_lens,
                "include_histograms": config.include_histograms,
                "include_top_tokens": config.include_top_tokens,
                "logit_lens_k": config.logit_lens_k,
                "histogram_bins": config.histogram_bins,
                "top_tokens_k": config.top_tokens_k,
                "include_saelens_format": config.include_saelens_format,
                "include_explanations": config.include_explanations,
            },
            status=ExportStatus.PENDING.value,
            progress=0.0,
            current_stage="Queued",
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        logger.info(f"Created export job {job.id} for SAE {sae_id}")
        return str(job.id)

    async def execute_export(
        self,
        db: AsyncSession,
        job_id: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Execute the export job.

        Args:
            db: Database session
            job_id: Export job ID
            progress_callback: Optional callback(progress, message) for updates

        Returns:
            Path to the generated ZIP archive
        """
        # Load job
        job = await db.get(NeuronpediaExportJob, job_id)
        if not job:
            raise ValueError(f"Export job not found: {job_id}")

        # Update status to computing
        job.status = ExportStatus.COMPUTING.value
        job.started_at = datetime.utcnow()
        await db.commit()

        try:
            # Load SAE
            sae = await db.get(ExternalSAE, job.sae_id)
            config = self._parse_config(job.config)

            # Resolve model name - look up from Model table if needed
            model_name = sae.model_name
            if not model_name and sae.model_id:
                model = await db.get(Model, sae.model_id)
                if model:
                    model_name = model.repo_id or model.name
            if not model_name:
                model_name = "unknown"

            # Create output directory
            output_dir = self._exports_dir / str(job.id)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load features (pass SAE for training_id lookup)
            features = await self._load_features(db, job.sae_id, config, sae)
            job.feature_count = len(features)
            await db.commit()

            logger.info(f"Loaded {len(features)} features for export job {job_id}")

            # Stage 1: Compute logit lens (if enabled)
            if config.include_logit_lens:
                await self._update_stage(db, job, "Computing logit lens data", 0)
                try:
                    await self._compute_logit_lens(db, sae, features, config, progress_callback)
                except Exception as e:
                    logger.warning(f"Logit lens computation failed, skipping: {e}")

            # Stage 2: Compute histograms (if enabled)
            if config.include_histograms:
                await self._update_stage(db, job, "Generating histograms", 25)
                try:
                    await self._compute_histograms(db, sae, features, config, progress_callback)
                except Exception as e:
                    logger.warning(f"Histogram computation failed, skipping: {e}")

            # Stage 3: Aggregate tokens (if enabled)
            if config.include_top_tokens:
                await self._update_stage(db, job, "Aggregating tokens", 50)
                try:
                    await self._aggregate_tokens(db, sae, features, config, progress_callback)
                except Exception as e:
                    logger.warning(f"Token aggregation failed, skipping: {e}")

            # Stage 4: Generate JSON files
            await self._update_stage(db, job, "Generating JSON files", 70)
            await self._generate_json_files(db, sae, features, output_dir, config, model_name)

            # Stage 5: Generate SAELens format (if enabled)
            if config.include_saelens_format:
                await self._update_stage(db, job, "Generating SAELens format", 85)
                await self._generate_saelens_format(db, sae, output_dir, model_name)

            # Stage 6: Generate README
            await self._generate_readme(sae, output_dir, config, model_name)

            # Stage 7: Create archive
            job.status = ExportStatus.PACKAGING.value
            await self._update_stage(db, job, "Creating archive", 95)
            archive_path = await self._create_archive(output_dir, job_id)

            # Complete job
            job.status = ExportStatus.COMPLETED.value
            job.output_path = str(archive_path)
            job.file_size_bytes = archive_path.stat().st_size
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            job.current_stage = "Complete"
            await db.commit()

            logger.info(f"Export job {job_id} completed: {archive_path}")
            return str(archive_path)

        except Exception as e:
            logger.exception(f"Export job {job_id} failed: {e}")
            job.status = ExportStatus.FAILED.value
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()
            raise

    def _parse_config(self, config_dict: Dict[str, Any]) -> ExportConfig:
        """Parse config dictionary into ExportConfig."""
        return ExportConfig(
            feature_selection=config_dict.get("feature_selection", "all"),
            feature_indices=config_dict.get("feature_indices"),
            include_logit_lens=config_dict.get("include_logit_lens", True),
            include_histograms=config_dict.get("include_histograms", True),
            include_top_tokens=config_dict.get("include_top_tokens", True),
            logit_lens_k=config_dict.get("logit_lens_k", 20),
            histogram_bins=config_dict.get("histogram_bins", 50),
            top_tokens_k=config_dict.get("top_tokens_k", 50),
            include_saelens_format=config_dict.get("include_saelens_format", True),
            include_explanations=config_dict.get("include_explanations", True),
        )

    async def _update_stage(
        self,
        db: AsyncSession,
        job: NeuronpediaExportJob,
        stage: str,
        progress: float,
    ):
        """Update job stage and progress."""
        job.current_stage = stage
        job.progress = progress
        await db.commit()

    async def _load_features(
        self,
        db: AsyncSession,
        sae_id: str,
        config: ExportConfig,
        sae: Optional[ExternalSAE] = None,
    ) -> List[Feature]:
        """Load features based on config.

        Tries to find features by:
        1. external_sae_id (for HuggingFace SAEs)
        2. training_id (for trained SAEs)
        """
        from sqlalchemy import or_

        # Load SAE if not provided
        if sae is None:
            sae = await db.get(ExternalSAE, sae_id)

        # Build query that checks both external_sae_id and training_id
        if sae and sae.training_id:
            stmt = select(Feature).where(
                or_(
                    Feature.external_sae_id == sae_id,
                    Feature.training_id == sae.training_id
                )
            )
        else:
            stmt = select(Feature).where(Feature.external_sae_id == sae_id)

        if config.feature_indices:
            stmt = stmt.where(Feature.neuron_index.in_(config.feature_indices))

        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def _compute_logit_lens(
        self,
        db: AsyncSession,
        sae: ExternalSAE,
        features: List[Feature],
        config: ExportConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """Compute logit lens for all features."""
        service = get_logit_lens_service()
        feature_indices = [f.neuron_index for f in features]

        results = await service.compute_logit_lens_for_sae(
            db=db,
            sae_id=sae.id,
            feature_indices=feature_indices,
            k=config.logit_lens_k,
            force_recompute=False,
        )

        # Save results
        await service.save_logit_lens_results(db, sae.id, results)

    async def _compute_histograms(
        self,
        db: AsyncSession,
        sae: ExternalSAE,
        features: List[Feature],
        config: ExportConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """Compute histograms for all features."""
        service = get_histogram_service()

        results = await service.compute_histograms_for_sae(
            db=db,
            sae_id=sae.id,
            n_bins=config.histogram_bins,
            force_recompute=False,
        )

        await service.save_histogram_results(db, sae.id, results)

    async def _aggregate_tokens(
        self,
        db: AsyncSession,
        sae: ExternalSAE,
        features: List[Feature],
        config: ExportConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """Aggregate top tokens for all features."""
        service = get_token_aggregator_service()

        results = await service.aggregate_tokens_for_sae(
            db=db,
            sae_id=sae.id,
            k=config.top_tokens_k,
            force_recompute=False,
        )

        await service.save_token_aggregation_results(db, sae.id, results)

    async def _generate_json_files(
        self,
        db: AsyncSession,
        sae: ExternalSAE,
        features: List[Feature],
        output_dir: Path,
        config: ExportConfig,
        model_name: str,
    ):
        """Generate Neuronpedia JSON files."""
        # Create directories
        features_dir = output_dir / "features"
        features_dir.mkdir(exist_ok=True)

        if config.include_explanations:
            explanations_dir = output_dir / "explanations"
            explanations_dir.mkdir(exist_ok=True)

        # Generate metadata.json
        metadata = self._generate_metadata_json(sae, model_name)
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Generate feature JSON files
        for feature in features:
            # Load dashboard data
            stmt = select(FeatureDashboardData).where(
                FeatureDashboardData.feature_id == feature.id
            )
            result = await db.execute(stmt)
            dashboard_data = result.scalar_one_or_none()

            # Load activations
            stmt = select(FeatureActivation).where(
                FeatureActivation.feature_id == feature.id
            ).limit(100)  # Limit to top 100 examples
            result = await db.execute(stmt)
            activations = list(result.scalars().all())

            feature_json = self._generate_feature_json(feature, dashboard_data, activations)

            with open(features_dir / f"{feature.neuron_index}.json", "w") as f:
                json.dump(feature_json, f, indent=2)

        # Generate explanations.json
        if config.include_explanations:
            explanations = self._generate_explanations_json(features)
            with open(explanations_dir / "explanations.json", "w") as f:
                json.dump(explanations, f, indent=2)

    def _generate_metadata_json(self, sae: ExternalSAE, model_name: str) -> dict:
        """Generate metadata.json content."""
        model_id = get_transformerlens_model_id(model_name)
        hook_name = get_transformerlens_hook_name(sae.layer or 0, "resid_post")

        return {
            "model_id": model_id,
            "sae_id": f"layer_{sae.layer}_res_{(sae.n_features or 0) // 1000}k",
            "neuronpedia_id": f"{model_id}/{sae.layer}-mistudio-res-{(sae.n_features or 0) // 1000}k",
            "hook_point": hook_name,
            "hook_name": hook_name,
            "d_sae": sae.n_features,
            "d_model": sae.d_model,
            "architecture": sae.architecture or "standard",
            "source": {
                "tool": "miStudio",
                "version": "1.0.0",
                "sae_id": sae.id,
            },
            "created_at": datetime.utcnow().isoformat() + "Z",
            "export_version": "1.0",
        }

    def _generate_feature_json(
        self,
        feature: Feature,
        dashboard_data: Optional[FeatureDashboardData],
        activations: List[FeatureActivation],
    ) -> dict:
        """Generate individual feature JSON content."""
        result = {
            "feature_index": feature.neuron_index,
            "statistics": {
                "activation_frequency": feature.activation_frequency,
                "max_activation": feature.max_activation,
                "mean_activation": feature.mean_activation,
                "interpretability_score": feature.interpretability_score,
            },
        }

        # Add dashboard data if available
        if dashboard_data:
            if dashboard_data.logit_lens_data:
                result["logits"] = dashboard_data.logit_lens_data

            if dashboard_data.histogram_data:
                result["histogram"] = dashboard_data.histogram_data

            if dashboard_data.top_tokens:
                result["top_tokens"] = dashboard_data.top_tokens

        # Add activation examples
        result["activations"] = [
            {
                "text": act.text if hasattr(act, 'text') else None,
                "tokens": act.tokens,
                "values": act.activations,
            }
            for act in activations
        ]

        return result

    def _generate_explanations_json(self, features: List[Feature]) -> dict:
        """Generate explanations.json content."""
        explanations = []

        for feature in features:
            # Skip features without meaningful labels
            if not feature.name or feature.name == f"feature_{feature.neuron_index}":
                continue

            explanations.append({
                "feature_index": feature.neuron_index,
                "explanations": [{
                    "description": feature.name,
                    "method": f"miStudio_{feature.label_source}",
                    "score": feature.interpretability_score,
                    "created_at": feature.labeled_at.isoformat() + "Z" if feature.labeled_at else None,
                }],
            })

        return {"explanations": explanations}

    async def _generate_saelens_format(
        self,
        db: AsyncSession,
        sae: ExternalSAE,
        output_dir: Path,
        model_name: str,
    ):
        """Generate SAELens-compatible format files."""
        saelens_dir = output_dir / "saelens"
        saelens_dir.mkdir(exist_ok=True)

        # Copy SAE weights
        sae_path = settings.resolve_data_path(sae.local_path)
        weights_src = sae_path / "sae_weights.safetensors"
        if weights_src.exists():
            shutil.copy(weights_src, saelens_dir / "sae_weights.safetensors")
        elif (sae_path / "checkpoint.safetensors").exists():
            # Convert from miStudio format
            shutil.copy(sae_path / "checkpoint.safetensors", saelens_dir / "sae_weights.safetensors")

        # Generate cfg.json
        model_id = get_transformerlens_model_id(model_name)
        hook_name = get_transformerlens_hook_name(sae.layer or 0, "resid_post")

        cfg = build_neuronpedia_config(
            model_name=model_name,
            layer=sae.layer or 0,
            d_in=sae.d_model or 768,
            d_sae=sae.n_features or 16384,
            hook_type="resid_post",
            architecture=sae.architecture or "standard",
        )

        with open(saelens_dir / "cfg.json", "w") as f:
            json.dump(cfg, f, indent=2)

    async def _generate_readme(
        self,
        sae: ExternalSAE,
        output_dir: Path,
        config: ExportConfig,
        model_name: str,
    ):
        """Generate README.md file."""
        model_id = get_transformerlens_model_id(model_name)

        readme = f"""# SAE Export - {sae.name}

## Overview
- **Model**: {model_id}
- **Layer**: {sae.layer}
- **Features**: {sae.n_features}
- **Architecture**: {sae.architecture or "standard"}

## Generated by miStudio

This export was generated by [miStudio](https://github.com/your-repo/mistudio), an open-source
sparse autoencoder training and analysis tool.

## Contents

- `metadata.json` - SAE metadata and configuration
- `features/` - Individual feature JSON files with:
  - Activation examples
  - Logit lens data (top promoted/suppressed tokens)
  - Activation histograms
  - Top activating tokens
- `explanations/` - Feature explanations/labels
- `saelens/` - SAELens-compatible format (cfg.json + weights)

## Usage with Neuronpedia

1. Visit https://www.neuronpedia.org/upload
2. Upload this ZIP archive
3. Your SAE will be available for exploration!

## Usage with SAELens

```python
from sae_lens import SAE

sae = SAE.load_from_pretrained(
    path="./saelens/",
    device="cuda"
)
```

## Export Date
{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
"""

        with open(output_dir / "README.md", "w") as f:
            f.write(readme)

    async def _create_archive(self, output_dir: Path, job_id: str) -> Path:
        """Create ZIP archive of export."""
        archive_name = f"neuronpedia_export_{job_id}.zip"
        archive_path = self._exports_dir / archive_name

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir)
                    zipf.write(file_path, arcname)

        logger.info(f"Created archive: {archive_path} ({archive_path.stat().st_size} bytes)")

        # Cleanup output directory
        shutil.rmtree(output_dir)

        return archive_path

    async def get_job_status(
        self,
        db: AsyncSession,
        job_id: str,
    ) -> Dict[str, Any]:
        """Get export job status."""
        job = await db.get(NeuronpediaExportJob, job_id)
        if not job:
            raise ValueError(f"Export job not found: {job_id}")

        return {
            "id": str(job.id),
            "sae_id": job.sae_id,
            "status": job.status,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "feature_count": job.feature_count,
            "output_path": job.output_path,
            "file_size_bytes": job.file_size_bytes,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message,
        }

    async def cancel_export(
        self,
        db: AsyncSession,
        job_id: str,
    ) -> bool:
        """Cancel an export job."""
        job = await db.get(NeuronpediaExportJob, job_id)
        if not job:
            raise ValueError(f"Export job not found: {job_id}")

        if job.status in (ExportStatus.COMPLETED.value, ExportStatus.FAILED.value):
            return False

        job.status = ExportStatus.CANCELLED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        return True


# Global service instance
_export_service: Optional[NeuronpediaExportService] = None


def get_neuronpedia_export_service() -> NeuronpediaExportService:
    """Get the global Neuronpedia export service instance."""
    global _export_service
    if _export_service is None:
        _export_service = NeuronpediaExportService()
    return _export_service
