# Curriculum Learning Strategy: Phase 1 Diagnostic

## Executive Summary

✅ **Phase 1 Rewards Validated**  
- Random policy baseline: **26.9 ± 8.4** average return
- Reward signal is **clear and meaningful**
- Environment is **ready for PPO training**

## Phase 1: Simplified Curriculum

**Goal:** Verify PPO can learn on an easy task before tackling full problem

### Configuration
```python
Phase 1 Rewards:
  - coverage:   0.5 (room swept reward)
  - rescue:     5.0 (people rescued reward)
  - hp_loss:    0.0 (OFF - no penalties)
  - time:       0.0 (OFF - no time pressure)
  - redundancy: 0.0 (OFF - no redundancy bonus)
```

### Why This Works

1. **Simplified reward** → Easier for PPO to optimize
2. **No penalties** → Removes conflicting signals
3. **High rescue weight (5.0)** → Strong incentive for the main goal
4. **Small coverage weight (0.5)** → Encourages exploration

### Environment Characteristics
- Standard office layout (11 nodes)
- 2 agents
- 6 people to rescue
- No fire/smoke hazards

### Baseline Performance

Random policy achieves **~27 average return** in 100 steps:
```
Episode 1: 35.00  (rescued people, covered rooms)
Episode 2: 25.00  (good coverage)
Episode 3: 11.33  (minimal sweeping)
Episode 4: 30.83  (good performance)
Episode 5: 32.33  (excellent)
```

This shows the reward function differentiates well between behaviors.

## Next Steps: Full PPO Training

### Training Script
```bash
python3 src/rl/enhanced_training.py
```

### Configuration for Phase 1 Training
```python
PPOConfig(
    scenario="office",
    num_agents=2,
    lr_policy=5e-4,
    lr_value=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    entropy_coef=0.02,
    batch_size=64,
    num_iterations=1000,  # Start with shorter training
    log_interval=10,      # Print diagnostics often
)

RewardShaper.for_scenario(
    scenario="office",
    curriculum_phase=1  # Simplified rewards
)
```

### Success Criteria

**✓ PASS** (PPO learning Phase 1):
- Average return increases over time
- Improvement >5.0 over random
- Agents learn to move and rescue

**⚠ WEAK** (Slow learning):
- Improvement 2-5.0 over random
- May need reward tuning or longer training

**✗ FAIL** (No learning):
- Improvement ≤0 or negative
- Debug: action masking, observations, network gradients

## After Phase 1 Success

### Phase 2: Add Mild Penalties
```python
curriculum_phase=2
# Same as Phase 1 but add:
weight_hp_loss=0.02  # Small penalty for civilian HP loss
```

### Phase 3: Full Realism
```python
curriculum_phase=3
# Add redundancy bonus for high-risk rooms
weight_redundancy=1.0
```

## Technical Infrastructure

### Files Modified
- `reward_shaper.py`: Added `curriculum_phase` parameter
- `enhanced_training.py`: Already supports phase training

### Diagnostics Available
- Random policy baseline (quick verification)
- Greedy policy baseline (simple agent comparison)
- Full PPO training with comprehensive logging

## Key Insights

1. **Reward shaping matters** - Clear signal (coverage + rescue) is better than complex multi-term rewards
2. **Start simple** - Phase 1 removes penalties to eliminate conflicting gradients
3. **Gradual complexity** - If Phase 1 works, add penalties in Phase 2
4. **Validation approach** - Test each phase before moving to next

## Running the Full Pipeline

```bash
# 1. Quick validation (5 min)
python3 src/rl/quick_phase1_diagnostic.py

# 2. Full PPO training on Phase 1 (if step 1 passes)
python3 src/rl/enhanced_training.py

# 3. After Phase 1 works, try Phase 2
# (modify reward_shaper.for_scenario(..., curriculum_phase=2))

# 4. After Phase 2 works, try Phase 3
# (modify reward_shaper.for_scenario(..., curriculum_phase=3))
```

## Monitoring Training

The enhanced trainer will print comprehensive diagnostics every `log_interval` iterations:

```
[Iteration 10]
  Actions: move=24, search=15, wait=11
  Positions: agent0@5, agent1@7
  Loss: policy=0.234, value=0.456, entropy=0.789
  Rewards: mean=22.3, std=5.1, min=8.2, max=35.6
  Advantages: mean=0.05, std=0.23
  Episodes: rescued=4/10, sweep_complete=6/10
```

This lets you see in real-time if PPO is learning.

---

**Status: Phase 1 environment validated and ready for PPO training** ✅
