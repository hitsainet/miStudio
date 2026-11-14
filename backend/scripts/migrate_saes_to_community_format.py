#!/usr/bin/env python3
"""
Migration script to convert existing miStudio SAE checkpoints to Community Standard format.

This script:
1. Scans the SAE storage directory for existing SAEs
2. Detects which ones are in miStudio format (not Community Standard)
3. Converts them to Community Standard format in place
4. Updates the database format field

Usage:
    python scripts/migrate_saes_to_community_format.py [--dry-run] [--sae-id SAE_ID]

Options:
    --dry-run     Show what would be migrated without making changes
    --sae-id      Migrate a specific SAE by ID
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from core.config import settings
from models.external_sae import ExternalSAE, SAEFormat
from ml.community_format import (
    is_community_format,
    is_mistudio_format,
    migrate_mistudio_to_community,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def get_saes_to_migrate(
    session: AsyncSession,
    sae_id: str = None
) -> list[ExternalSAE]:
    """Get SAEs that need migration."""
    query = select(ExternalSAE).where(
        ExternalSAE.format == SAEFormat.MISTUDIO.value
    )
    if sae_id:
        query = query.where(ExternalSAE.id == sae_id)

    result = await session.execute(query)
    return list(result.scalars().all())


async def migrate_sae(
    session: AsyncSession,
    sae: ExternalSAE,
    dry_run: bool = False
) -> bool:
    """
    Migrate a single SAE from miStudio to Community Standard format.

    Returns:
        True if migration was successful, False otherwise
    """
    local_path = Path(sae.local_path)

    if not local_path.exists():
        logger.warning(f"SAE {sae.id}: Local path does not exist: {local_path}")
        return False

    # Check if already Community Standard format (might have been manually converted)
    if is_community_format(local_path):
        logger.info(f"SAE {sae.id}: Already in Community Standard format, updating database")
        if not dry_run:
            sae.format = SAEFormat.COMMUNITY_STANDARD.value
            await session.commit()
        return True

    # Check if it's miStudio format
    if not is_mistudio_format(local_path):
        logger.warning(f"SAE {sae.id}: Unknown format at {local_path}")
        return False

    logger.info(f"SAE {sae.id}: Migrating from miStudio to Community Standard format")

    if dry_run:
        logger.info(f"  [DRY RUN] Would convert {local_path}")
        return True

    try:
        # Determine model name and layer
        model_name = sae.model_name or "unknown"
        layer = sae.layer or 0

        # Get hyperparameters if available
        hyperparams = None
        if sae.sae_metadata and "training_hyperparameters" in sae.sae_metadata:
            hyperparams = sae.sae_metadata["training_hyperparameters"]

        # Create backup directory name
        backup_path = local_path.parent / f"{local_path.name}_backup_mistudio"

        # Copy existing files to backup
        import shutil
        if local_path.is_dir():
            shutil.copytree(local_path, backup_path)
        else:
            backup_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, backup_path / local_path.name)

        logger.info(f"  Created backup at {backup_path}")

        # Find the checkpoint file
        source_checkpoint = local_path
        if local_path.is_dir():
            # Look for checkpoint.safetensors or any .safetensors
            checkpoint_file = local_path / "checkpoint.safetensors"
            if not checkpoint_file.exists():
                safetensors_files = list(local_path.glob("*.safetensors"))
                if safetensors_files:
                    checkpoint_file = safetensors_files[0]
                else:
                    logger.error(f"  No safetensors file found in {local_path}")
                    return False
            source_checkpoint = checkpoint_file

        # Perform migration
        migrate_mistudio_to_community(
            source_path=source_checkpoint.parent if source_checkpoint.is_file() else source_checkpoint,
            output_dir=local_path,
            model_name=model_name,
            layer=layer,
            hyperparams=hyperparams,
        )

        # Update database
        sae.format = SAEFormat.COMMUNITY_STANDARD.value
        if sae.sae_metadata is None:
            sae.sae_metadata = {}
        sae.sae_metadata["migrated_from_mistudio"] = True
        sae.sae_metadata["backup_path"] = str(backup_path)
        await session.commit()

        logger.info(f"  Successfully migrated {sae.id}")
        return True

    except Exception as e:
        logger.error(f"  Error migrating {sae.id}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main(dry_run: bool = False, sae_id: str = None):
    """Main migration function."""
    # Create database engine
    engine = create_async_engine(settings.database_url.replace("postgresql://", "postgresql+asyncpg://"))
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Get SAEs to migrate
        saes = await get_saes_to_migrate(session, sae_id)

        if not saes:
            logger.info("No SAEs found that need migration")
            return

        logger.info(f"Found {len(saes)} SAE(s) to migrate")

        success_count = 0
        fail_count = 0

        for sae in saes:
            logger.info(f"\nProcessing SAE: {sae.id} ({sae.name})")
            if await migrate_sae(session, sae, dry_run):
                success_count += 1
            else:
                fail_count += 1

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Migration complete:")
        logger.info(f"  Successful: {success_count}")
        logger.info(f"  Failed: {fail_count}")

        if dry_run:
            logger.info("\n[DRY RUN] No changes were made")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate miStudio SAE checkpoints to Community Standard format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--sae-id",
        type=str,
        default=None,
        help="Migrate a specific SAE by ID"
    )

    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, sae_id=args.sae_id))
