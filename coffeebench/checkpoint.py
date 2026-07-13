"""Checkpoint system for resuming experiments after API failures."""

import json
from pathlib import Path


class CheckpointManager:
    """Manages saving and loading experiment checkpoints.
    
    Checkpoints save minimal state (last completed day) to enable resume.
    Full state reconstruction happens by replaying the trajectory file.
    """

    def __init__(self, output_dir: str, enabled: bool = True):
        """Initialize checkpoint manager.
        
        Args:
            output_dir: Directory where checkpoints will be saved
            enabled: Whether checkpointing is enabled
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
    def save_checkpoint(self, day: int) -> None:
        """Save checkpoint after completing a day.
        
        Args:
            day: Last completed day number
        """
        if not self.enabled:
            return
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "last_completed_day": day,
            "trajectory_file": str(self.output_dir / "run.json"),
        }
        
        # Write to temp file first, then rename (atomic operation)
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        temp_file.rename(self.checkpoint_file)
        
        if day % 10 == 0:  # Only print every 10 days to reduce noise
            print(f"[checkpoint] Saved at day {day}")
        
    def load_checkpoint(self) -> int | None:
        """Load checkpoint if it exists.
        
        Returns:
            Last completed day if checkpoint exists, None otherwise
        """
        if not self.enabled or not self.checkpoint_file.exists():
            return None
            
        try:
            with open(self.checkpoint_file) as f:
                checkpoint = json.load(f)
            day = checkpoint["last_completed_day"]
            print(f"[checkpoint] Found checkpoint at day {day}")
            return day
        except Exception as e:
            print(f"[checkpoint] Failed to load checkpoint: {e}")
            return None
            
    def clear_checkpoint(self) -> None:
        """Remove checkpoint file after successful completion."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("[checkpoint] Cleared checkpoint")
            
    def checkpoint_exists(self) -> bool:
        """Check if a checkpoint file exists."""
        return self.checkpoint_file.exists()
