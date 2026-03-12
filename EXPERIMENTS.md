# Experiments — Hot Dog / Not Hot Dog

## v1 — 2026-03-11
- **Changes:** Initial run. ViT-B/16 pretrained, Food-101 binary (balanced)
- **Params:** batch=64, lr=3e-4, epochs=10, dropout=0.0, augmentation=none, warmup=0
- **Workload ID:** 369f3cb0-8e32-4150-bcc9-b6008a3ff96b
- **Result:** ❌ FAILED — `AttributeError: total_mem` (should be `total_memory`)
- **Fix:** PyTorch API mismatch. Changed to `.total_memory`.

## v2 — 2026-03-11
- **Changes:** Fix total_mem + lazy dataset loading
- **Workload ID:** 104d1951-faa6-48b2-8a80-95feb9b3ca26
- **Result:** ❌ CANCELLED — same bug as v1 (pushed before fix applied)

## v3 — 2026-03-11
- **Changes:** Fix total_memory + lazy loading applied
- **Params:** batch=64, lr=3e-4, epochs=10
- **Workload ID:** 31cbeca2-d989-4847-b27b-dd457f0e2de0
- **Result:** ❌ FAILED (17s) — `DatasetNotFoundError: 'ethz/food-101'`
- **Fix:** Dataset name is `food101` in datasets==2.16.1, not `ethz/food-101`.

## v4 — 2026-03-11
- **Changes:** Correct dataset name `food101`
- **Params:** batch=64, lr=3e-4, epochs=10, augmentation=none, class_weights=false
- **Workload ID:** 227d8ec5-9d69-454f-9204-9d1f8632e739
- **Result:** 65.2% val acc (hotdog: 60.4%, not_hotdog: 68.4%). Underfitting.
- **Duration:** 777s, 0.216 GPU-hours
- **Diagnosis:** Low LR + no augmentation + no class weights = underfitting on small balanced dataset
- **Next:** Lower LR, enable augmentation + class weights + warmup

## v5 — 2026-03-11 🏆
- **Changes:** lr=1e-4, augmentation=basic, class_weights=true, warmup=2, epochs=15
- **Params:** batch=64, lr=1e-4, epochs=15, dropout=0.0, augmentation=basic, class_weights=true, warmup=2
- **Workload ID:** 4e8eae2f-67a6-4bf4-84d8-d52962e227b5
- **Result:** ✅ **98.0% val acc** (hotdog: 96.8%, not_hotdog: 99.2%) — TARGET HIT
- **Duration:** 1037s (~17 min), 0.288 GPU-hours
- **Total cost:** ~5 iterations, ~0.55 GPU-hours total
- **W&B:** jasonong-chamberai/chamber-hotdog
