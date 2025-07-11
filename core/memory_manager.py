#!/usr/bin/env python3
"""
Central Memory Manager for AGI System
=====================================

Critical component for preventing memory leaks and ensuring system stability.
Tracks memory usage per module and enforces hard limits.
"""

import torch
import psutil
import gc
import weakref
from typing import Dict, Optional, Any, List, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import time
import logging
import os
import traceback
from abc import ABC, abstractmethod
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics for a module"""
    allocated_bytes: int = 0
    peak_bytes: int = 0
    n_allocations: int = 0
    n_cleanups: int = 0
    last_cleanup: float = 0.0
    tensors_tracked: int = 0


@dataclass  
class MemoryAllocation:
    """Track individual memory allocations"""
    size_bytes: int
    timestamp: float
    tensor_shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    device: Optional[str] = None


class CircularBuffer:
    """Fixed-size circular buffer for preventing infinite growth"""
    
    def __init__(self, max_size: int, dtype: torch.dtype = torch.float32):
        self.max_size = max_size
        self.dtype = dtype
        self.buffer = []
        self.index = 0
        
    def append(self, item: torch.Tensor) -> None:
        """Add item to buffer, overwriting oldest if full"""
        # For scalar tensors, store as Python float to avoid memory overhead
        if item.numel() == 1:
            item_to_store = item.detach().item()  # Convert to Python scalar
        else:
            # For larger tensors, keep as tensor but ensure it's detached and on CPU
            item_to_store = item.detach().cpu()
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(item_to_store)
        else:
            # Overwrite oldest
            old_item = self.buffer[self.index]
            if isinstance(old_item, torch.Tensor):
                # Explicitly free old tensor memory
                del old_item
            self.buffer[self.index] = item_to_store
            self.index = (self.index + 1) % self.max_size
            
    def get_all(self) -> List[Union[torch.Tensor, float]]:
        """Get all items in order (oldest to newest)"""
        if len(self.buffer) < self.max_size:
            return self.buffer.copy()
        else:
            # Reorder to have oldest first
            return self.buffer[self.index:] + self.buffer[:self.index]
            
    def clear(self) -> None:
        """Clear buffer and free memory"""
        for item in self.buffer:
            if isinstance(item, torch.Tensor):
                del item
        self.buffer.clear()
        self.index = 0
        gc.collect()
        
    def __len__(self) -> int:
        return len(self.buffer)


class ModuleMemoryTracker:
    """Track memory usage for a specific module"""
    
    def __init__(self, module_name: str, budget_bytes: int):
        self.module_name = module_name
        self.budget_bytes = budget_bytes
        self.stats = MemoryStats()
        self.allocations = OrderedDict()  # track_id -> MemoryAllocation
        self.tensor_registry = weakref.WeakValueDictionary()  # track tensors without preventing GC
        self._lock = threading.Lock()
        
    def allocate(self, size_bytes: int, tensor: Optional[torch.Tensor] = None) -> bool:
        """Try to allocate memory, return True if successful"""
        with self._lock:
            if self.stats.allocated_bytes + size_bytes > self.budget_bytes:
                logger.warning(f"{self.module_name}: Memory allocation of {size_bytes/1024**2:.2f}MB "
                             f"would exceed budget ({self.stats.allocated_bytes/1024**2:.2f}/"
                             f"{self.budget_bytes/1024**2:.2f}MB used)")
                return False
                
            track_id = id(tensor) if tensor is not None else len(self.allocations)
            self.allocations[track_id] = MemoryAllocation(
                size_bytes=size_bytes,
                timestamp=time.time(),
                tensor_shape=tuple(tensor.shape) if tensor is not None else None,
                dtype=tensor.dtype if tensor is not None else None,
                device=str(tensor.device) if tensor is not None else None
            )
            
            if tensor is not None:
                self.tensor_registry[track_id] = tensor
                
            self.stats.allocated_bytes += size_bytes
            self.stats.peak_bytes = max(self.stats.peak_bytes, self.stats.allocated_bytes)
            self.stats.n_allocations += 1
            self.stats.tensors_tracked = len(self.tensor_registry)
            
            return True
            
    def deallocate(self, size_bytes: int, tensor: Optional[torch.Tensor] = None) -> None:
        """Deallocate memory"""
        with self._lock:
            track_id = id(tensor) if tensor is not None else None
            
            if track_id and track_id in self.allocations:
                del self.allocations[track_id]
                
            self.stats.allocated_bytes = max(0, self.stats.allocated_bytes - size_bytes)
            self.stats.tensors_tracked = len(self.tensor_registry)
            
    def cleanup_old_allocations(self, max_age_seconds: float = 300) -> int:
        """Remove allocations older than max_age_seconds"""
        with self._lock:
            current_time = time.time()
            cleaned = 0
            
            # Find old allocations
            to_remove = []
            for track_id, alloc in self.allocations.items():
                if current_time - alloc.timestamp > max_age_seconds:
                    to_remove.append((track_id, alloc.size_bytes))
                    
            # Remove old allocations
            for track_id, size_bytes in to_remove:
                del self.allocations[track_id]
                self.stats.allocated_bytes -= size_bytes
                cleaned += 1
                
            if cleaned > 0:
                self.stats.n_cleanups += 1
                self.stats.last_cleanup = current_time
                logger.info(f"{self.module_name}: Cleaned {cleaned} old allocations, "
                          f"freed {sum(s for _, s in to_remove)/1024**2:.2f}MB")
                
            return cleaned


class CentralMemoryManager:
    """
    Central memory management system with hard limits per module.
    Prevents memory leaks and enforces resource constraints.
    """
    
    def __init__(self, total_budget_gb: float = 16.0, 
                 cleanup_interval_seconds: float = 60.0,
                 emergency_threshold: float = 0.9):
        self.total_budget_bytes = int(total_budget_gb * 1024**3)
        self.cleanup_interval = cleanup_interval_seconds
        self.emergency_threshold = emergency_threshold
        
        self.module_trackers: Dict[str, ModuleMemoryTracker] = {}
        self.module_budgets: Dict[str, float] = {}  # fraction of total budget
        
        self._last_cleanup = time.time()
        self._lock = threading.Lock()
        
        # System memory monitoring
        self.process = psutil.Process(os.getpid())
        self.system_memory = psutil.virtual_memory().total
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"CentralMemoryManager initialized with {total_budget_gb}GB budget")
        
    def register_module(self, module_name: str, budget_fraction: float) -> None:
        """Register a module with its memory budget as fraction of total"""
        with self._lock:
            if module_name in self.module_budgets:
                logger.warning(f"Module {module_name} already registered, updating budget")
                
            self.module_budgets[module_name] = budget_fraction
            budget_bytes = int(self.total_budget_bytes * budget_fraction)
            self.module_trackers[module_name] = ModuleMemoryTracker(module_name, budget_bytes)
            
            logger.info(f"Registered {module_name} with {budget_bytes/1024**2:.1f}MB budget "
                       f"({budget_fraction*100:.1f}% of total)")
            
    def allocate(self, module_name: str, size_bytes: int, 
                 tensor: Optional[torch.Tensor] = None) -> bool:
        """Try to allocate memory for a module"""
        if module_name not in self.module_trackers:
            logger.error(f"Module {module_name} not registered!")
            return False
            
        tracker = self.module_trackers[module_name]
        
        # Try allocation
        if tracker.allocate(size_bytes, tensor):
            return True
            
        # Allocation failed, try cleanup first
        logger.info(f"{module_name}: Allocation failed, triggering cleanup")
        self.cleanup_module(module_name)
        
        # Retry allocation
        return tracker.allocate(size_bytes, tensor)
        
    def deallocate(self, module_name: str, size_bytes: int,
                   tensor: Optional[torch.Tensor] = None) -> None:
        """Deallocate memory for a module"""
        if module_name in self.module_trackers:
            self.module_trackers[module_name].deallocate(size_bytes, tensor)
            
    @contextmanager
    def track_allocation(self, module_name: str, tensor: torch.Tensor):
        """Context manager to track tensor allocation"""
        size_bytes = tensor.element_size() * tensor.nelement()
        allocated = self.allocate(module_name, size_bytes, tensor)
        try:
            yield allocated
        finally:
            if allocated:
                self.deallocate(module_name, size_bytes, tensor)
                
    def cleanup_module(self, module_name: str, force: bool = False) -> None:
        """Trigger cleanup for a specific module"""
        if module_name not in self.module_trackers:
            return
            
        tracker = self.module_trackers[module_name]
        
        # Cleanup old allocations
        tracker.cleanup_old_allocations()
        
        # Force garbage collection if needed
        if force or self._is_memory_critical():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        logger.info(f"{module_name}: Memory after cleanup: "
                   f"{tracker.stats.allocated_bytes/1024**2:.1f}MB")
                
    def cleanup_all(self, force: bool = False) -> None:
        """Cleanup all modules"""
        logger.info("Running global memory cleanup")
        
        for module_name in self.module_trackers:
            self.cleanup_module(module_name, force)
            
        # Global garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics for all modules"""
        stats = {
            'total_budget_gb': self.total_budget_bytes / 1024**3,
            'system_memory_gb': self.system_memory / 1024**3,
            'process_memory_mb': self.process.memory_info().rss / 1024**2,
            'modules': {}
        }
        
        total_allocated = 0
        for name, tracker in self.module_trackers.items():
            module_stats = {
                'budget_mb': tracker.budget_bytes / 1024**2,
                'allocated_mb': tracker.stats.allocated_bytes / 1024**2,
                'peak_mb': tracker.stats.peak_bytes / 1024**2,
                'utilization': tracker.stats.allocated_bytes / tracker.budget_bytes,
                'n_allocations': tracker.stats.n_allocations,
                'n_cleanups': tracker.stats.n_cleanups,
                'tensors_tracked': tracker.stats.tensors_tracked
            }
            stats['modules'][name] = module_stats
            total_allocated += tracker.stats.allocated_bytes
            
        stats['total_allocated_mb'] = total_allocated / 1024**2
        stats['total_utilization'] = total_allocated / self.total_budget_bytes
        
        return stats
        
    def _is_memory_critical(self) -> bool:
        """Check if system memory usage is critical"""
        memory_percent = psutil.virtual_memory().percent / 100.0
        process_percent = self.process.memory_info().rss / self.system_memory
        
        return (memory_percent > self.emergency_threshold or 
                process_percent > self.emergency_threshold)
                
    def _cleanup_loop(self) -> None:
        """Background thread for periodic cleanup"""
        while True:
            time.sleep(self.cleanup_interval)
            
            try:
                if self._is_memory_critical():
                    logger.warning("Memory critical, forcing cleanup")
                    self.cleanup_all(force=True)
                elif time.time() - self._last_cleanup > self.cleanup_interval:
                    self.cleanup_all(force=False)
                    self._last_cleanup = time.time()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                
    def create_circular_buffer(self, module_name: str, max_size: int, 
                              dtype: torch.dtype = torch.float32) -> CircularBuffer:
        """Create a circular buffer with memory tracking"""
        # Estimate memory usage
        element_size = torch.tensor([], dtype=dtype).element_size()
        estimated_bytes = max_size * element_size * 64  # assume 64 elements per tensor
        
        # Check if module can allocate
        if not self.allocate(module_name, estimated_bytes):
            raise MemoryError(f"Cannot allocate circular buffer for {module_name}")
            
        return CircularBuffer(max_size, dtype)


# Global singleton instance
_memory_manager: Optional[CentralMemoryManager] = None


def get_memory_manager(total_budget_gb: float = 16.0) -> CentralMemoryManager:
    """Get or create the global memory manager"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = CentralMemoryManager(total_budget_gb)
    return _memory_manager


class MemoryManagedModule(ABC):
    """Base class for memory-managed AGI modules"""
    
    def __init__(self, module_name: str, memory_fraction: float = 0.1):
        self.module_name = module_name
        self.memory_manager = get_memory_manager()
        self.memory_manager.register_module(module_name, memory_fraction)
        
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up module memory - must be implemented by subclasses"""
        pass
        
    def allocate_tensor(self, shape: Tuple[int, ...], 
                       dtype: torch.dtype = torch.float32,
                       device: torch.device = None) -> Optional[torch.Tensor]:
        """Allocate a tensor with memory tracking"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # Calculate size
        numel = 1
        for dim in shape:
            numel *= dim
        size_bytes = numel * torch.tensor([], dtype=dtype).element_size()
        
        # Try to allocate
        if self.memory_manager.allocate(self.module_name, size_bytes):
            tensor = torch.zeros(shape, dtype=dtype, device=device)
            # Track the tensor
            self.memory_manager.module_trackers[self.module_name].tensor_registry[id(tensor)] = tensor
            return tensor
        else:
            logger.error(f"{self.module_name}: Failed to allocate tensor of shape {shape}")
            self.cleanup()  # Try cleanup
            return None
            
    def create_buffer(self, max_size: int) -> CircularBuffer:
        """Create a managed circular buffer"""
        return self.memory_manager.create_circular_buffer(self.module_name, max_size)