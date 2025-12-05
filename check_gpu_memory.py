#!/usr/bin/env python3
"""
GPU Memory Management Helper

Clears GPU cache and displays memory usage before training.
Run this before starting training if you encounter OOM errors.
"""

import torch
import subprocess
import os

def check_gpu_status():
    """Check GPU status and memory usage."""
    print("\n" + "="*70)
    print("GPU STATUS CHECK")
    print("="*70 + "\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - training will run on CPU (slower)")
        return False
    
    print(f"✅ CUDA available")
    print(f"   Device count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n📊 GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Memory info
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        free = total - reserved
        
        print(f"   Total memory:     {total:.2f} GB")
        print(f"   Allocated:        {allocated:.2f} GB")
        print(f"   Reserved:         {reserved:.2f} GB")
        print(f"   Free:             {free:.2f} GB")
        
        # Check if memory is low
        if free < 2.0:
            print(f"   ⚠️  WARNING: Low free memory ({free:.2f} GB)")
            print(f"      Consider clearing cache or reducing batch sizes")
        elif free < 5.0:
            print(f"   ⚠️  Moderate free memory ({free:.2f} GB)")
        else:
            print(f"   ✅ Good free memory ({free:.2f} GB)")
    
    return True


def clear_gpu_cache():
    """Clear PyTorch GPU cache."""
    print("\n" + "="*70)
    print("CLEARING GPU CACHE")
    print("="*70 + "\n")
    
    if not torch.cuda.is_available():
        print("❌ No GPU to clear")
        return
    
    before = torch.cuda.memory_allocated(0) / 1e9
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated(0) / 1e9
    freed = before - after
    
    print(f"✅ Cache cleared")
    print(f"   Before: {before:.2f} GB")
    print(f"   After:  {after:.2f} GB")
    print(f"   Freed:  {freed:.2f} GB")


def show_gpu_processes():
    """Show processes using GPU."""
    print("\n" + "="*70)
    print("GPU PROCESSES")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            print(f"Found {len(lines)} GPU process(es):\n")
            
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 3:
                    pid = parts[0].strip()
                    name = parts[1].strip()
                    mem = parts[2].strip()
                    print(f"   PID {pid}: {name} ({mem})")
            
            print("\n💡 To kill a process: kill -9 <PID>")
            print("   Or kill all Python processes: pkill -9 python")
            
        else:
            print("No GPU processes found or nvidia-smi not available")
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    except subprocess.TimeoutExpired:
        print("⏱️ nvidia-smi timed out")
    except Exception as e:
        print(f"❌ Error: {e}")


def set_memory_config():
    """Set environment variables for memory management."""
    print("\n" + "="*70)
    print("MEMORY CONFIGURATION")
    print("="*70 + "\n")
    
    # Set PyTorch memory allocator settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    
    print("✅ Environment variables set:")
    print("   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512")
    print("\nThis helps reduce memory fragmentation and OOM errors.")


def main():
    """Main memory management workflow."""
    print("\n" + "="*70)
    print("GPU MEMORY MANAGEMENT HELPER")
    print("="*70)
    
    # 1. Check status
    has_gpu = check_gpu_status()
    
    if not has_gpu:
        print("\n⚠️  No GPU available - training will be slow on CPU")
        return
    
    # 2. Show processes
    show_gpu_processes()
    
    # 3. Clear cache
    clear_gpu_cache()
    
    # 4. Set config
    set_memory_config()
    
    # 5. Check status again
    check_gpu_status()
    
    # 6. Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70 + "\n")
    
    free_mem = (torch.cuda.get_device_properties(0).total_memory - 
                torch.cuda.memory_reserved(0)) / 1e9
    
    if free_mem < 5.0:
        print("⚠️  LOW MEMORY DETECTED")
        print("\n   Recommended actions:")
        print("   1. Kill other GPU processes:")
        print("      nvidia-smi")
        print("      kill -9 <PID>")
        print("\n   2. Reduce training batch sizes:")
        print("      - Daycare: batch_size=32, batch_rollout_size=2")
        print("      - Warehouse: batch_size=16, batch_rollout_size=1")
        print("\n   3. Reduce number of agents:")
        print("      - Try with 2-3 agents instead of 4+")
        print("\n   4. Run training with environment variable:")
        print("      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 ...")
    else:
        print("✅ SUFFICIENT MEMORY")
        print(f"\n   Free memory: {free_mem:.2f} GB")
        print("   You should be able to train with:")
        print("   - Daycare: 3-4 agents, batch_size=48")
        print("   - Warehouse: 3-4 agents, batch_size=32")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
