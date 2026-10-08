"""
SteeringExperiment service layer for business logic.

This module contains the SteeringExperimentsService class which handles all
steering experiment-related business logic and database operations.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.steering_experiment import SteeringExperiment


class SteeringExperimentsService:
    """Service class for steering experiment operations."""

    @staticmethod
    async def create_experiment(
        db: AsyncSession,
        name: str,
        comparison_id: str,
        results: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> SteeringExperiment:
        """
        Create a new steering experiment.

        Args:
            db: Database session
            name: Experiment name
            comparison_id: The comparison ID from the steering result
            results: Full SteeringComparisonResponse as dict
            description: Optional description
            tags: Optional list of tags

        Returns:
            Created steering experiment object
        """
        # Extract data from results
        sae_id = results.get("sae_id", "")
        model_id = results.get("model_id", "")
        prompt = results.get("prompt", "")

        # Extract selected_features from steered outputs
        selected_features = []
        for steered_output in results.get("steered", []):
            if "feature_config" in steered_output:
                selected_features.append(steered_output["feature_config"])

        # Extract generation_params from the first steered output's metrics or use defaults
        generation_params = {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "num_samples": 1,
        }

        db_experiment = SteeringExperiment(
            name=name,
            description=description,
            sae_id=sae_id,
            model_id=model_id,
            comparison_id=comparison_id,
            prompt=prompt,
            selected_features=selected_features,
            generation_params=generation_params,
            results=results,
            tags=tags or [],
        )

        db.add(db_experiment)
        await db.commit()
        await db.refresh(db_experiment)

        return db_experiment

    @staticmethod
    async def get_experiment(
        db: AsyncSession,
        experiment_id: UUID
    ) -> Optional[SteeringExperiment]:
        """
        Get a steering experiment by ID.

        Args:
            db: Database session
            experiment_id: Experiment UUID

        Returns:
            SteeringExperiment object if found, None otherwise
        """
        result = await db.execute(
            select(SteeringExperiment).where(SteeringExperiment.id == experiment_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_experiment_by_comparison_id(
        db: AsyncSession,
        comparison_id: str
    ) -> Optional[SteeringExperiment]:
        """
        Get a steering experiment by comparison ID.

        Args:
            db: Database session
            comparison_id: Comparison ID

        Returns:
            SteeringExperiment object if found, None otherwise
        """
        result = await db.execute(
            select(SteeringExperiment).where(
                SteeringExperiment.comparison_id == comparison_id
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_experiments(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        sae_id: Optional[str] = None,
        model_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> tuple[List[SteeringExperiment], int]:
        """
        List steering experiments with filtering and pagination.

        Args:
            db: Database session
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return
            search: Search query for name or description
            sae_id: Filter by SAE ID
            model_id: Filter by model ID
            tag: Filter by tag

        Returns:
            Tuple of (list of experiments, total count)
        """
        # Build base query
        query = select(SteeringExperiment)
        count_query = select(func.count()).select_from(SteeringExperiment)

        # Apply filters
        filters = []

        if search:
            search_filter = or_(
                SteeringExperiment.name.ilike(f"%{search}%"),
                SteeringExperiment.description.ilike(f"%{search}%"),
                SteeringExperiment.prompt.ilike(f"%{search}%"),
            )
            filters.append(search_filter)

        if sae_id:
            filters.append(SteeringExperiment.sae_id == sae_id)

        if model_id:
            filters.append(SteeringExperiment.model_id == model_id)

        if tag:
            # JSONB contains for tags array
            filters.append(SteeringExperiment.tags.contains([tag]))

        if filters:
            query = query.where(*filters)
            count_query = count_query.where(*filters)

        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply sorting (newest first)
        query = query.order_by(SteeringExperiment.created_at.desc())

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await db.execute(query)
        experiments = result.scalars().all()

        return list(experiments), total

    @staticmethod
    async def update_experiment(
        db: AsyncSession,
        experiment_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[SteeringExperiment]:
        """
        Update a steering experiment.

        Args:
            db: Database session
            experiment_id: Experiment UUID
            name: Optional new name
            description: Optional new description
            tags: Optional new tags

        Returns:
            Updated steering experiment object if found, None otherwise
        """
        result = await db.execute(
            select(SteeringExperiment).where(SteeringExperiment.id == experiment_id)
        )
        db_experiment = result.scalar_one_or_none()

        if not db_experiment:
            return None

        # Apply updates
        if name is not None:
            db_experiment.name = name
        if description is not None:
            db_experiment.description = description
        if tags is not None:
            db_experiment.tags = tags

        db_experiment.updated_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(db_experiment)

        return db_experiment

    @staticmethod
    async def delete_experiment(
        db: AsyncSession,
        experiment_id: UUID
    ) -> bool:
        """
        Delete a steering experiment.

        Args:
            db: Database session
            experiment_id: Experiment UUID

        Returns:
            True if deleted, False if not found
        """
        result = await db.execute(
            select(SteeringExperiment).where(SteeringExperiment.id == experiment_id)
        )
        db_experiment = result.scalar_one_or_none()

        if not db_experiment:
            return False

        await db.delete(db_experiment)
        await db.commit()

        return True

    @staticmethod
    async def delete_experiments_batch(
        db: AsyncSession,
        experiment_ids: List[UUID]
    ) -> int:
        """
        Delete multiple steering experiments.

        Args:
            db: Database session
            experiment_ids: List of experiment UUIDs to delete

        Returns:
            Number of experiments deleted
        """
        deleted_count = 0

        for experiment_id in experiment_ids:
            result = await db.execute(
                select(SteeringExperiment).where(SteeringExperiment.id == experiment_id)
            )
            db_experiment = result.scalar_one_or_none()

            if db_experiment:
                await db.delete(db_experiment)
                deleted_count += 1

        await db.commit()

        return deleted_count
