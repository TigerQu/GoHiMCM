# Bug Fixes Summary

## Overview
All four critical bugs in `src/Env_sim` have been identified, fixed, and validated. The fixes address high-severity issues in node-type encoding, movement time units, configuration handling, and robustness.

---

## Fix #1: NODE_TYPES One-Hot Dimension Mismatch ✅

**Severity:** HIGH  
**Files Modified:** `config.py`, `env.py`

### Problem
- `NODE_TYPES` dictionary contains 4 entries (room:0, hall:1, exit:2, floor:3)
- One-hot encoding was hardcoded to size 3, causing IndexError when 'floor' node type was used
- Feature dimension mismatch would crash when extracting features for PyTorch Geometric

### Solution
- Updated `FEATURE_DIM` in `config.py`: 10 → **11**
- Changed one-hot generation in `env.get_node_features()`:
  ```python
  one_hot = np.zeros(len(NODE_TYPES), dtype=np.float32)  # Dynamic size
  ```
- Updated feature vector assembly indices and documentation

### Validation
✅ Standard office: 11D features  
✅ Babycare (45 nodes): 11D features  
✅ Warehouse (41 nodes): 11D features  

---

## Fix #2: Civilian Traversal Time Unit Mismatch ✅

**Severity:** HIGH  
**File Modified:** `occupants.py`

### Problem
- Agent movement times computed in timesteps (divide seconds by 5.0)
- Person edge costs returned in raw seconds, causing 5× speed difference
- Hazard penalties (100.0, 10.0) were additive absolute values, not relative weights
- Civilians moved unrealistically slow compared to agents

### Solution
- `_edge_cost_for_person()` now converts times to timesteps:
  ```python
  base_time_timesteps = (base_time_seconds * congestion_factor) / 5.0
  ```
- Changed hazard penalties to **multiplicative factors**:
  - Fire hazard: 1.5× cost multiplier
  - Smoke hazard: 1.1× cost multiplier
- Updated docstring to clarify all costs are in timesteps

### Validation
✅ Person/agent time ratio matches speed ratio  
✅ Fire penalty: 1.5x (realistic avoidance)  
✅ Smoke penalty: 1.1x (realistic avoidance)  
✅ Congestion applies multiplicatively  

---

## Fix #3A: Person.effective_speed Uses Per-Environment Config ✅

**Severity:** MEDIUM  
**Files Modified:** `entities.py`, `env.py`

### Problem
- `Person.effective_speed` read from module-level `DEFAULT_CONFIG`
- Custom config overrides passed to `BuildingFireEnvironment(config=...)` were ignored
- Speed multipliers couldn't be tuned per-environment

### Solution
- Added dataclass fields to `Person`:
  ```python
  assisted_multiplier: float = 1.8
  panic_multiplier: float = 0.7
  ```
- `spawn_person()` and `reset()` copy config values to each Person:
  ```python
  assisted_multiplier=self.config.get('assisted_speed_multiplier', 1.8),
  panic_multiplier=self.config.get('panic_speed_reduction', 0.7)
  ```
- `effective_speed` now uses stored multipliers (not global config)

### Validation
✅ Default multipliers: 1.8× assist, 0.7× panic  
✅ Custom config (2.5× assist, 0.5× panic) respected  
✅ Multipliers preserved across reset()  

---

## Fix #4: Hazard Health Degradation Robustness ✅

**Severity:** LOW  
**File Modified:** `hazards.py`

### Problem
- `degrade_health()` did not guard against persons with invalid `node_id`
- Could cause KeyError if node_id was None or missing from nodes dict

### Solution
- Added guard before node lookup:
  ```python
  if person.node_id not in nodes:
      if verbose:
          print(f"Warning: Person {person.pid} has invalid node_id: {person.node_id}")
      continue
  ```

### Validation
✅ No crashes on invalid node_id  
✅ Optional warning message  

---

## Test Results

### Layout Testing (`test_all_layouts.py`)
| Layout | Nodes | Edges | People | Status |
|--------|-------|-------|--------|--------|
| Office | 11 | 10 | 6 | ✅ PASSED |
| Babycare | 45 | 59 | 21 | ✅ PASSED |
| Warehouse | 41 | 100 | 6 | ✅ PASSED |

### Config Override Testing (`test_config_overrides.py`)
- ✅ Default multipliers correctly applied (1.8× assist, 0.7× panic)
- ✅ Custom config overrides respected (2.5× assist, 0.5× panic)
- ✅ Multipliers preserved across reset()
- ✅ effective_speed uses correct values

### Movement Units Testing (`test_movement_units.py`)
- ✅ Edge costs in timesteps (not seconds)
- ✅ Person/agent time ratio matches speed ratio
- ✅ Fire penalty: 1.5x multiplicative
- ✅ Smoke penalty: 1.1x multiplicative
- ✅ Congestion applies correctly

---

## Files Modified

### Core Environment Files
1. **`src/Env_sim/config.py`**
   - FEATURE_DIM: 10 → 11
   - Updated comment explaining new 4-type one-hot

2. **`src/Env_sim/entities.py`**
   - Added `assisted_multiplier` and `panic_multiplier` to Person
   - Updated `effective_speed` property to use stored multipliers

3. **`src/Env_sim/env.py`**
   - `get_node_features()`: one-hot now uses `len(NODE_TYPES)`
   - Feature vector indices updated for 11D format
   - `spawn_person()`: passes config multipliers to Person
   - `reset()`: passes config multipliers when creating missing People

4. **`src/Env_sim/occupants.py`**
   - `_edge_cost_for_person()`: returns timesteps (not seconds)
   - Hazard penalties changed to multiplicative factors (1.5×, 1.1×)
   - Updated docstring with unit clarification

5. **`src/Env_sim/hazards.py`**
   - `degrade_health()`: added guard for invalid node_id

### New Test Files
- `src/scripts/test_all_layouts.py` — Comprehensive layout testing
- `src/scripts/test_config_overrides.py` — Config override validation
- `src/scripts/test_movement_units.py` — Movement time units validation

---

## Recommendations

### Immediate Actions
1. Review and commit these changes to version control
2. Run tests with your RL training setup to verify performance
3. Consider running longer episodes to validate fire spread and civilian behavior

### Optional Future Enhancements
1. Make hazard multipliers (1.5×, 1.1×) configurable in DEFAULT_CONFIG
2. Extend and validate mobility category mappings ('staff', 'infant', etc.)
3. Add benchmark for evacuation times under different scenarios
4. Consider adaptive hazard penalties based on node type (e.g., stairwells)

---

## Summary

✅ **All 4 critical bugs fixed and validated**
- NODE_TYPES encoding: Fixed dimension mismatch
- Movement units: Fixed civilian speed and hazard routing
- Config handling: Fixed per-environment multiplier support
- Robustness: Added guards for edge cases

✅ **Comprehensive testing**
- 3 layouts tested (11-45 nodes each)
- Config overrides validated
- Movement units verified

✅ **Ready for production use**
