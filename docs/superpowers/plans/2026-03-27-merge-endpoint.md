# Merge Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `POST /merge` endpoint to the tunerservice that accepts the Controller's current `ModelData`, overlays tuned `PerfParms` from the `ParameterStore`, appends any extra ParameterStore entries with defaults, and returns the merged `ModelData`.

**Architecture:** The `TunerService` gains a `Merge` method that performs a two-phase merge: first it overlays tuned `PerfParms` onto matching input entries, then it appends any ParameterStore entries not present in the input (using default values for non-parameter fields). The handler and route follow the existing `/tune` pattern exactly.

**Tech Stack:** Go, Gin, `github.com/llm-inferno/optimizer-light/pkg/config` (ModelData/ModelAcceleratorPerfData/PerfParms types)

---

## File Map

| File | Change |
|------|--------|
| `tunerservice/defaults.go` | Add `DefaultAccCount`, `DefaultMaxBatchSize`, `DefaultAtTokens` constants |
| `tunerservice/service.go` | Add `Merge(*optconfig.ModelData) *optconfig.ModelData` method |
| `tunerservice/handlers.go` | Add `handleMerge` handler |
| `tunerservice/server.go` | Register `POST /merge` route |
| `docs/tunerservice-design.md` | Document new endpoint |

---

### Task 1: Add merge defaults to `defaults.go`

**Files:**
- Modify: `tunerservice/defaults.go`

- [ ] **Step 1: Add the three default constants**

Open `tunerservice/defaults.go`. It currently contains `baseFactor` and server address constants. Add the merge defaults at the end:

```go
// Default field values used when the ParameterStore has a model/accelerator entry
// that is not present in the Controller's current ModelData.
const (
	DefaultAccCount     = 1
	DefaultMaxBatchSize = 256
	DefaultAtTokens     = 1024
)
```

The full file after the edit:

```go
package tunerservice

const (
	// baseFactor is the fraction of ITL assumed to be the baseline iteration overhead (alpha).
	// Used in guessInitState to derive an initial estimate of alpha from observed ITL.
	baseFactor = 0.9
)

// Environment variable names and defaults for the tuner REST server.
const (
	TunerHostEnvName = "TUNER_HOST"
	TunerPortEnvName = "TUNER_PORT"

	DefaultTunerHost = "localhost"
	DefaultTunerPort = "8081"
)

// Default field values used when the ParameterStore has a model/accelerator entry
// that is not present in the Controller's current ModelData.
const (
	DefaultAccCount     = 1
	DefaultMaxBatchSize = 256
	DefaultAtTokens     = 1024
)
```

- [ ] **Step 2: Build to confirm no errors**

```bash
cd /Users/tantawi/Projects/llm-inferno/model-tuner && go build ./...
```

Expected: no output (clean build).

- [ ] **Step 3: Commit**

```bash
cd /Users/tantawi/Projects/llm-inferno/model-tuner
git add tunerservice/defaults.go
git commit -m "feat(tunerservice): add merge default constants for AccCount, MaxBatchSize, AtTokens"
```

---

### Task 2: Add `Merge` method to `TunerService`

**Files:**
- Modify: `tunerservice/service.go`

- [ ] **Step 1: Add the `Merge` method**

Append the following method to `tunerservice/service.go`, after the existing `GetParams` method (line 178):

```go
// Merge accepts the Controller's current ModelData and returns it with PerfParms overlaid
// from the ParameterStore for any matching (name, accelerator) pairs. Entries in the
// ParameterStore that have no match in the input are appended with default non-parameter fields.
func (ts *TunerService) Merge(modelData *optconfig.ModelData) *optconfig.ModelData {
	allParams := ts.paramStore.GetAll()
	matched := make(map[string]bool, len(allParams))

	// Phase 1: overlay tuned PerfParms onto existing entries.
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

	// Phase 2: append ParameterStore entries not present in the input.
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

- [ ] **Step 2: Build to confirm no errors**

```bash
cd /Users/tantawi/Projects/llm-inferno/model-tuner && go build ./...
```

Expected: no output (clean build).

- [ ] **Step 3: Commit**

```bash
cd /Users/tantawi/Projects/llm-inferno/model-tuner
git add tunerservice/service.go
git commit -m "feat(tunerservice): add Merge method to TunerService"
```

---

### Task 3: Add `handleMerge` handler and register route

**Files:**
- Modify: `tunerservice/handlers.go`
- Modify: `tunerservice/server.go`

- [ ] **Step 1: Add `handleMerge` to `handlers.go`**

Append the following handler to `tunerservice/handlers.go`, after the `handleGetParams` function:

```go
// POST /merge
// Request body: config.ModelData (the Controller's current ModelData)
// Response:     config.ModelData with PerfParms overlaid from the ParameterStore;
//               ParameterStore entries absent from the input are appended with defaults.
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

- [ ] **Step 2: Register the route in `server.go`**

In `tunerservice/server.go`, inside `NewTunerServer`, add the `/merge` route after the existing route registrations:

```go
func NewTunerServer(service *TunerService) *TunerServer {
	router := gin.Default()
	ts := &TunerServer{service: service, router: router}
	router.POST("/tune", ts.handleTune)
	router.GET("/getparams", ts.handleGetParams)
	router.POST("/merge", ts.handleMerge)
	return ts
}
```

- [ ] **Step 3: Build to confirm no errors**

```bash
cd /Users/tantawi/Projects/llm-inferno/model-tuner && go build ./...
```

Expected: no output (clean build).

- [ ] **Step 4: Smoke-test the endpoint manually**

Start the server in one terminal:

```bash
cd /Users/tantawi/Projects/llm-inferno/model-tuner
CONFIG_DATA_DIR=./config-data TUNER_PORT=8081 go run ./demos/tunerservice/
```

In a second terminal, first call `/tune` to populate the ParameterStore:

```bash
curl -s -X POST localhost:8081/tune \
  -H 'Content-Type: application/json' \
  -d '[{
    "name": "llama3-8b", "class": "llm", "model": "llama3-8b",
    "maxBatchSize": 256,
    "currentAlloc": {
      "accelerator": "A100", "numReplicas": 1, "maxBatch": 256,
      "ttftAverage": 450.0, "itlAverage": 25.0,
      "load": { "arrivalRate": 10.0, "avgInTokens": 512, "avgOutTokens": 128 }
    }
  }]' | jq .
```

Expected: `200 OK` with ModelData containing one entry for `llama3-8b/A100`.

Now call `/merge` with a ModelData that has the same entry (with stale perfParms) plus an extra entry not in the ParameterStore:

```bash
curl -s -X POST localhost:8081/merge \
  -H 'Content-Type: application/json' \
  -d '{
    "models": [
      {
        "name": "llama3-8b", "acc": "A100",
        "accCount": 2, "maxBatchSize": 128, "atTokens": 512,
        "perfParms": { "alpha": 1.0, "beta": 1.0, "gamma": 1.0 }
      },
      {
        "name": "llama3-70b", "acc": "H100",
        "accCount": 4, "maxBatchSize": 512, "atTokens": 2048,
        "perfParms": { "alpha": 5.0, "beta": 2.0, "gamma": 0.5 }
      }
    ]
  }' | jq .
```

Expected response:
- `llama3-8b/A100`: `accCount=2`, `maxBatchSize=128`, `atTokens=512` preserved; `perfParms` updated to tuned values (alpha/beta/gamma different from 1.0/1.0/1.0)
- `llama3-70b/H100`: all fields unchanged (no tuned params for this pair)

Also test the extra-entry case — call `/merge` with an empty ModelData to verify ParameterStore entries are appended with defaults:

```bash
curl -s -X POST localhost:8081/merge \
  -H 'Content-Type: application/json' \
  -d '{"models": []}' | jq .
```

Expected: `200 OK` with one entry for `llama3-8b/A100` having `accCount=1`, `maxBatchSize=256`, `atTokens=1024`.

- [ ] **Step 5: Commit**

Stop the server (`Ctrl-C`). Then:

```bash
cd /Users/tantawi/Projects/llm-inferno/model-tuner
git add tunerservice/handlers.go tunerservice/server.go
git commit -m "feat(tunerservice): add POST /merge endpoint"
```

---

### Task 4: Update design doc

**Files:**
- Modify: `docs/tunerservice-design.md`

- [ ] **Step 1: Add the `/merge` API section**

In `docs/tunerservice-design.md`, after the `GET /getparams` section (after line 205, before `---`), add:

```markdown
### `POST /merge`

```
POST /merge
Content-Type: application/json

{ "models": [ { "name", "acc", "accCount", "maxBatchSize", "atTokens", "perfParms": { "alpha", "beta", "gamma" } }, ... ] }

200 OK
{ "models": [ ... merged entries ... ] }

400 Bad Request   ← invalid JSON
```

Merges the Controller's current `ModelData` with tuned parameters from the `ParameterStore`:

- For each entry in the input: if the ParameterStore has tuned params for that `(name, acc)` pair, its `PerfParms` (alpha/beta/gamma) are replaced. All other fields (`accCount`, `maxBatchSize`, `atTokens`) are preserved unchanged.
- ParameterStore entries not present in the input are appended as new entries with tuned `PerfParms` and defaults: `accCount=1`, `maxBatchSize=256`, `atTokens=1024`.
- An empty `models` array is valid input; the response will contain only the extra ParameterStore entries.
```

- [ ] **Step 2: Update the architecture diagram comment**

In `docs/tunerservice-design.md`, update the architecture block (lines 15–38) to reflect the `/merge` step in the Controller's loop. Replace the `SystemData.Spec.Models = ModelData` line so the diagram reads:

```
  │  POST /tune  ──[ []ServerSpec (ReplicaSpecs) ]──►  TunerServer
  │  ◄──────────────[ ModelData (tuned pairs only) ]───┘
  │
  │  POST /merge ──[ ModelData (current) ]──────────►  TunerServer
  │  ◄──────────────[ ModelData (merged)  ]────────────┘
  │
  │  SystemData.Spec.Models = merged ModelData
```

- [ ] **Step 3: Build to confirm no errors introduced**

```bash
cd /Users/tantawi/Projects/llm-inferno/model-tuner && go build ./...
```

Expected: no output (clean build).

- [ ] **Step 4: Commit**

```bash
cd /Users/tantawi/Projects/llm-inferno/model-tuner
git add docs/tunerservice-design.md
git commit -m "docs(tunerservice): document POST /merge endpoint"
```
