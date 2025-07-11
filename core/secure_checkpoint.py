#!/usr/bin/env python3
"""
Secure Checkpoint System for AGI
================================

Replaces pickle with secure serialization using safetensors and JSON.
Prevents code execution vulnerabilities while maintaining full state.
"""

import json
import torch
import os
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import logging
from safetensors.torch import save_file, load_file
from safetensors import safe_open
import gzip
import base64

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    version: str
    timestamp: str
    iteration: int
    generation: int
    checksum: str
    modules: List[str]
    stats: Dict[str, Any]
    
    
class SecureCheckpointError(Exception):
    """Custom exception for checkpoint errors"""
    pass


class SecureCheckpointManager:
    """
    Secure checkpoint system using safetensors for PyTorch tensors
    and JSON for metadata. No pickle = no code execution risk.
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, checkpoint_dir: str = "checkpoints", 
                 compression: bool = True,
                 validate_on_load: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.compression = compression
        self.validate_on_load = validate_on_load
        
        # Supported types for JSON serialization
        self.json_encoders = {
            np.ndarray: lambda x: {"_type": "ndarray", "data": x.tolist(), "dtype": str(x.dtype)},
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            torch.Tensor: lambda x: {"_type": "tensor", "data": x.cpu().numpy().tolist(), 
                                    "dtype": str(x.dtype), "shape": list(x.shape)},
            type: lambda x: {"_type": "type", "module": x.__module__, "name": x.__name__}
        }
        
    def save(self, state: Dict[str, Any], name: str, 
             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save checkpoint securely without pickle.
        
        Args:
            state: State dictionary to save
            name: Name for the checkpoint
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.ckpt"
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Separate tensors and metadata
            tensors = {}
            metadata_dict = {
                "version": self.VERSION,
                "timestamp": datetime.now().isoformat(),
                "user_metadata": metadata or {}
            }
            
            # Process state recursively
            processed_state = self._process_state_for_saving(state, tensors, "")
            metadata_dict["state"] = processed_state
            
            # Save tensors with safetensors
            if tensors:
                tensor_path = Path(temp_dir) / "tensors.safetensors"
                save_file(tensors, str(tensor_path))
                
            # Save metadata as JSON
            metadata_path = Path(temp_dir) / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=self._json_encoder)
                
            # Calculate checksum
            checksum = self._calculate_checksum(temp_dir)
            metadata_dict["checksum"] = checksum
            
            # Re-save metadata with checksum
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=self._json_encoder)
                
            # Create final checkpoint (compressed or not)
            if self.compression:
                self._create_compressed_checkpoint(temp_dir, checkpoint_path)
            else:
                self._create_uncompressed_checkpoint(temp_dir, checkpoint_path)
                
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise SecureCheckpointError(f"Failed to save checkpoint: {e}")
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def load(self, name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load checkpoint securely.
        
        Args:
            name: Name of checkpoint to load
            
        Returns:
            Tuple of (state_dict, metadata)
        """
        # Try different possible paths
        possible_paths = [
            self.checkpoint_dir / f"{name}.ckpt",
            self.checkpoint_dir / f"{name}.ckpt.gz", 
            self.checkpoint_dir / name,
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                break
                
        if checkpoint_path is None:
            raise SecureCheckpointError(f"Checkpoint not found: {name}")
                
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Extract checkpoint
            if checkpoint_path.suffix == '.gz' or self._is_compressed(checkpoint_path):
                self._extract_compressed_checkpoint(checkpoint_path, temp_dir)
            else:
                self._extract_uncompressed_checkpoint(checkpoint_path, temp_dir)
                
            # Load metadata
            metadata_path = Path(temp_dir) / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                
            # Validate if required
            if self.validate_on_load:
                self._validate_checkpoint(temp_dir, metadata_dict)
                
            # Load tensors if they exist
            tensors = {}
            tensor_path = Path(temp_dir) / "tensors.safetensors"
            if tensor_path.exists():
                tensors = load_file(str(tensor_path))
                
            # Reconstruct state
            state = self._reconstruct_state(metadata_dict["state"], tensors, "")
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return state, metadata_dict.get("user_metadata", {})
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise SecureCheckpointError(f"Failed to load checkpoint: {e}")
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata"""
        checkpoints = []
        
        for ckpt_file in self.checkpoint_dir.glob("*.ckpt*"):
            try:
                # Quick load just metadata
                _, metadata = self.load(ckpt_file.stem)
                checkpoints.append({
                    "name": ckpt_file.stem,
                    "path": str(ckpt_file),
                    "size_mb": ckpt_file.stat().st_size / 1024**2,
                    "metadata": metadata
                })
            except Exception as e:
                logger.warning(f"Could not load metadata for {ckpt_file}: {e}")
                
        return sorted(checkpoints, key=lambda x: x["name"], reverse=True)
        
    def delete_checkpoint(self, name: str) -> bool:
        """Delete a checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.ckpt"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {name}")
            return True
        return False
        
    def _process_state_for_saving(self, obj: Any, tensors: Dict[str, torch.Tensor], 
                                  prefix: str) -> Any:
        """Recursively process state, extracting tensors"""
        if isinstance(obj, torch.Tensor):
            # Save tensor separately
            key = f"{prefix}_tensor_{len(tensors)}"
            tensors[key] = obj.cpu()  # Ensure on CPU
            return {"_tensor_ref": key}
            
        elif isinstance(obj, torch.nn.Module):
            # Save module state dict
            return {
                "_module_type": type(obj).__name__,
                "state_dict": self._process_state_for_saving(
                    obj.state_dict(), tensors, f"{prefix}_module"
                )
            }
            
        elif isinstance(obj, dict):
            return {
                k: self._process_state_for_saving(v, tensors, f"{prefix}_{k}")
                for k, v in obj.items()
            }
            
        elif isinstance(obj, (list, tuple)):
            processed = [
                self._process_state_for_saving(item, tensors, f"{prefix}_{i}")
                for i, item in enumerate(obj)
            ]
            return processed if isinstance(obj, list) else tuple(processed)
            
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            # Custom object - save its __dict__
            return {
                "_custom_type": type(obj).__name__,
                "data": self._process_state_for_saving(
                    obj.__dict__, tensors, f"{prefix}_obj"
                )
            }
            
        else:
            # Primitive type or unsupported - save as is
            return obj
            
    def _reconstruct_state(self, data: Any, tensors: Dict[str, torch.Tensor], 
                          prefix: str) -> Any:
        """Recursively reconstruct state from saved data"""
        if isinstance(data, dict):
            if "_tensor_ref" in data:
                # Restore tensor
                return tensors.get(data["_tensor_ref"])
                
            elif "_module_type" in data:
                # Note: We don't reconstruct modules, just return state dict
                # The caller is responsible for creating the module
                return self._reconstruct_state(
                    data["state_dict"], tensors, f"{prefix}_module"
                )
                
            elif "_custom_type" in data:
                # Return the data dict - caller handles reconstruction
                return self._reconstruct_state(
                    data["data"], tensors, f"{prefix}_obj"
                )
                
            elif "_type" in data:
                # Handle special JSON-encoded types
                if data["_type"] == "ndarray":
                    return np.array(data["data"], dtype=data["dtype"])
                elif data["_type"] == "tensor":
                    return torch.tensor(data["data"], dtype=getattr(torch, data["dtype"]))
                    
            else:
                # Regular dict
                return {
                    k: self._reconstruct_state(v, tensors, f"{prefix}_{k}")
                    for k, v in data.items()
                }
                
        elif isinstance(data, list):
            return [
                self._reconstruct_state(item, tensors, f"{prefix}_{i}")
                for i, item in enumerate(data)
            ]
            
        elif isinstance(data, tuple):
            return tuple(
                self._reconstruct_state(item, tensors, f"{prefix}_{i}")
                for i, item in enumerate(data)
            )
            
        else:
            return data
            
    def _json_encoder(self, obj):
        """Custom JSON encoder for special types"""
        obj_type = type(obj)
        if obj_type in self.json_encoders:
            return self.json_encoders[obj_type](obj)
        elif hasattr(obj, '__dict__'):
            return {"_custom_type": obj_type.__name__, "data": obj.__dict__}
        else:
            raise TypeError(f"Object of type {obj_type} is not JSON serializable")
            
    def _create_compressed_checkpoint(self, temp_dir: str, output_path: Path) -> None:
        """Create compressed checkpoint file"""
        import tarfile
        
        # Ensure .gz extension
        if not str(output_path).endswith('.gz'):
            output_path = Path(str(output_path) + '.gz')
            
        with tarfile.open(output_path, 'w:gz') as tar:
            for item in Path(temp_dir).iterdir():
                tar.add(item, arcname=item.name)
                
    def _create_uncompressed_checkpoint(self, temp_dir: str, output_path: Path) -> None:
        """Create uncompressed checkpoint directory"""
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(temp_dir, output_path)
        
    def _extract_compressed_checkpoint(self, checkpoint_path: Path, temp_dir: str) -> None:
        """Extract compressed checkpoint"""
        import tarfile
        
        with tarfile.open(checkpoint_path, 'r:gz') as tar:
            tar.extractall(temp_dir)
            
    def _extract_uncompressed_checkpoint(self, checkpoint_path: Path, temp_dir: str) -> None:
        """Extract uncompressed checkpoint"""
        if checkpoint_path.is_dir():
            shutil.copytree(checkpoint_path, temp_dir, dirs_exist_ok=True)
        else:
            # Try as tarfile
            self._extract_compressed_checkpoint(checkpoint_path, temp_dir)
            
    def _is_compressed(self, path: Path) -> bool:
        """Check if file is compressed"""
        if path.is_dir():
            return False
        try:
            with open(path, 'rb') as f:
                # Check for gzip magic number
                return f.read(2) == b'\x1f\x8b'
        except:
            return False
            
    def _calculate_checksum(self, directory: str) -> str:
        """Calculate checksum for all files in directory"""
        hasher = hashlib.sha256()
        
        for file_path in sorted(Path(directory).rglob('*')):
            if file_path.is_file() and file_path.name != 'metadata.json':
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
                    
        return hasher.hexdigest()
        
    def _validate_checkpoint(self, temp_dir: str, metadata: Dict[str, Any]) -> None:
        """Validate checkpoint integrity"""
        if "checksum" not in metadata:
            logger.warning("No checksum in checkpoint metadata")
            return
            
        # Temporarily remove checksum from metadata for calculation
        saved_checksum = metadata.pop("checksum", None)
        
        # Recalculate checksum
        calculated_checksum = self._calculate_checksum(temp_dir)
        
        # Restore checksum
        if saved_checksum:
            metadata["checksum"] = saved_checksum
            
        if saved_checksum != calculated_checksum:
            raise SecureCheckpointError(
                f"Checkpoint validation failed! "
                f"Expected: {saved_checksum}, Got: {calculated_checksum}"
            )
            
    def save_genome(self, genome: Any, generation: int, fitness: float) -> str:
        """Specialized method for saving genomes"""
        name = f"genome_gen_{generation}_fit_{fitness:.3f}"
        
        # Extract genome data safely
        genome_data = {
            "genes": genome.genes if hasattr(genome, 'genes') else {},
            "hyperparameters": genome.hyperparameters if hasattr(genome, 'hyperparameters') else {},
            "connection_genes": genome.connection_genes if hasattr(genome, 'connection_genes') else {},
            "module_order": genome.module_order if hasattr(genome, 'module_order') else []
        }
        
        metadata = {
            "generation": generation,
            "fitness": fitness,
            "type": "genome"
        }
        
        return self.save(genome_data, name, metadata)
        
    def load_genome(self, name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load a saved genome"""
        state, metadata = self.load(name)
        
        if metadata.get("type") != "genome":
            logger.warning(f"Checkpoint {name} is not a genome type")
            
        return state, metadata


# Global instance
_checkpoint_manager: Optional[SecureCheckpointManager] = None


def get_checkpoint_manager(checkpoint_dir: str = "checkpoints") -> SecureCheckpointManager:
    """Get or create global checkpoint manager"""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = SecureCheckpointManager(checkpoint_dir)
    return _checkpoint_manager