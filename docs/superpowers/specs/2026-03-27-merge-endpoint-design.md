# Design: POST /merge Endpoint for ModelData Merging

## Context

The Controller in the control-loop repo maintains a "current" `ModelData` that it passes
to the Optimizer via `SystemData.Spec.Models`. This ModelData starts as static config and
must be updated with EKF-tuned `PerfParms` after each `/tune` call.

Rather than having the Controller perform the merge itself, the Tuner takes on this
responsibility via a new `POST /merge` endpoint. This keeps the Controller light and
leverages the Tuner's internal `ParameterStore` which already holds the tuned parameters.

The Controller's serial loop becomes:

```
1. Collect metrics -> replicaSpecs
2. POST /tune  [replicaSpecs]       -> (tuned-only result, can be ignored)
3. POST /merge [current ModelData]  -> merged ModelData (becomes new "current")
4. SystemData.Spec.Models = merged ModelData
5. POST /optimizeOne(SystemData)    -> Optimizer
```

## Design

### Merge semantics

Given the Controller's current `ModelData` and the Tuner's `ParameterStore`:

1. **Overlay tuned entries**: For each entry in the input ModelData, if the ParameterStore
   has tuned params for that `(Name, Acc)` pair, replace its `PerfParms` (alpha/beta/gamma).
   All other fields (`AccCount`, `MaxBatchSize`, `AtTokens`) remain unchanged.

2. **Append new entries**: For each entry in the ParameterStore that has no matching entry
   in the input ModelData, append a new `ModelAcceleratorPerfData` with tuned `PerfParms`
   and default values for non-parameter fields:
   - `AccCount`: `DefaultAccCount = 1`
   - `MaxBatchSize`: `DefaultMaxBatchSize = 256`
   - `AtTokens`: `DefaultAtTokens = 1024`

### API

```
POST /merge
Content-Type: application/json

{ "models": [ { "name": "llama3-8b", "acc": "A100", "accCount": 2, ... }, ... ] }

200 OK
{ "models": [ ... merged entries ... ] }

400 Bad Request   <- invalid JSON
```

The response has the same `ModelData` shape as the input but with `PerfParms` updated
where the ParameterStore has tuned values, plus any new entries appended.

### Files to modify

| File | Change |
|------|--------|
| `tunerservice/defaults.go` | Add `DefaultAccCount`, `DefaultMaxBatchSize`, `DefaultAtTokens` constants |
| `tunerservice/service.go` | Add `Merge(modelData *optconfig.ModelData) *optconfig.ModelData` method |
| `tunerservice/handlers.go` | Add `handleMerge` handler |
| `tunerservice/server.go` | Register `POST /merge` route |
| `docs/tunerservice-design.md` | Document the new endpoint |
| `demos/tunerservice/main.go` | Add `/merge` call to the demo |

### Service method: `Merge`

```go
func (ts *TunerService) Merge(modelData *optconfig.ModelData) *optconfig.ModelData {
    allParams := ts.paramStore.GetAll()
    matched := make(map[string]bool)

    // Phase 1: overlay tuned PerfParms onto existing entries
    result := make([]optconfig.ModelAcceleratorPerfData, len(modelData.PerfData))
    for i, entry := range modelData.PerfData {
        result[i] = entry
        key := makeKey(entry.Name, entry.Acc)
        if params, ok := allParams[key]; ok {
            result[i].PerfParms = optconfig.PerfParms{
                Alpha: params.Alpha,
                Beta:  params.Beta,
                Gamma: params.Gamma,
            }
            matched[key] = true
        }
    }

    // Phase 2: append new entries from ParameterStore not in input
    for key, params := range allParams {
        if matched[key] {
            continue
        }
        model, acc := splitKey(key)
        result = append(result, optconfig.ModelAcceleratorPerfData{
            Name:         model,
            Acc:          acc,
            AccCount:     DefaultAccCount,
            MaxBatchSize: DefaultMaxBatchSize,
            AtTokens:     DefaultAtTokens,
            PerfParms: optconfig.PerfParms{
                Alpha: params.Alpha,
                Beta:  params.Beta,
                Gamma: params.Gamma,
            },
        })
    }

    return &optconfig.ModelData{PerfData: result}
}
```

### Handler: `handleMerge`

```go
func (ts *TunerServer) handleMerge(c *gin.Context) {
    var modelData optconfig.ModelData
    if err := c.ShouldBindJSON(&modelData); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body: " + err.Error()})
        return
    }
    merged := ts.service.Merge(&modelData)
    c.JSON(http.StatusOK, merged)
}
```

Note: an empty `PerfData` list is valid input (the Controller may have no initial data),
in which case the response contains only entries from the ParameterStore with defaults.

### Route registration

In `server.go`, add to `NewTunerServer`:

```go
router.POST("/merge", ts.handleMerge)
```

### Defaults

In `defaults.go`, add:

```go
const (
    DefaultAccCount     = 1
    DefaultMaxBatchSize = 256
    DefaultAtTokens     = 1024
)
```

## Verification

1. **Build**: `go build ./...`
2. **Run demo**: `CONFIG_DATA_DIR=./config-data TUNER_PORT=8081 go run ./demos/tunerservice/`
   - Demo should call `/tune` with synthetic specs, then call `/merge` with a ModelData
     that has some overlapping and some non-overlapping entries, and print the merged result.
3. **Manual curl test**:
   ```bash
   # After /tune has been called:
   curl -X POST localhost:8081/merge -H 'Content-Type: application/json' \
     -d '{"models":[{"name":"llama3-8b","acc":"A100","accCount":2,"maxBatchSize":128,"atTokens":512,"perfParms":{"alpha":1,"beta":1,"gamma":1}}]}'
   # Verify: perfParms should be updated if tuned, other fields preserved
   ```
4. **Edge cases to verify**:
   - Empty ModelData input -> returns only ParameterStore entries with defaults
   - ModelData with entries not in ParameterStore -> those entries pass through unchanged
   - ParameterStore empty (no /tune called yet) -> returns input ModelData unchanged
