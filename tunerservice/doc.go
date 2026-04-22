// Package tunerservice is a thin HTTP adapter over [pkg/service.TunerService].
//
// It exposes four endpoints:
//
//	POST /tune
//	  Body:     []config.ServerSpec   (ReplicaSpecs from the Collector)
//	  Response: config.ModelData      (updated alpha/beta/gamma per model/accelerator)
//
//	GET /getparams?model=<name>&accelerator=<acc>
//	  Response: JSON with alpha, beta, gamma, NIS, updateCount, lastUpdated
//
//	GET /warmup
//	  Response: {"warmingUp": bool}
//
//	POST /merge
//	  Body:     config.ModelData
//	  Response: config.ModelData with PerfParms overlaid from the parameter store
//
// All estimation logic lives in pkg/estimator and pkg/service.
package tunerservice
