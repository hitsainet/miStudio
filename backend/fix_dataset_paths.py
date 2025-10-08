"""Fix dataset paths in database to match actual HuggingFace cache locations."""
import asyncio
from pathlib import Path
from uuid import UUID

from src.core.database import AsyncSessionLocal
from src.services.dataset_service import DatasetService
from src.schemas.dataset import DatasetUpdate


async def fix_paths():
    """Update dataset paths to match actual filesystem."""
    async with AsyncSessionLocal() as session:
        # TinyStories dataset
        tinystories_id = UUID("8b240600-0ee2-4fb6-95e4-fb210a1836c6")
        tinystories_path = "data/datasets/roneneldan___tiny_stories"

        if Path(tinystories_path).exists():
            print(f"Updating TinyStories path to: {tinystories_path}")
            await DatasetService.update_dataset(
                session,
                tinystories_id,
                DatasetUpdate(raw_path=tinystories_path)
            )
        else:
            print(f"TinyStories path not found: {tinystories_path}")

        # hh-rlhf dataset
        hhrlhf_id = UUID("c363d616-204b-44e0-b274-c68dc93ed33c")
        hhrlhf_path = "data/datasets/Anthropic___hh-rlhf"

        if Path(hhrlhf_path).exists():
            print(f"Updating hh-rlhf path to: {hhrlhf_path}")
            await DatasetService.update_dataset(
                session,
                hhrlhf_id,
                DatasetUpdate(raw_path=hhrlhf_path)
            )
        else:
            print(f"hh-rlhf path not found: {hhrlhf_path}")

        await session.commit()
        print("Database updated successfully!")


if __name__ == "__main__":
    asyncio.run(fix_paths())
