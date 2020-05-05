/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import * as actions from '../actions';
import {
  ExecutionDigestsResponse,
  GraphOpInfoResponse,
} from '../data_source/tfdbg2_data_source';
import {reducers} from './debugger_reducers';
import {
  DataLoadState,
  Execution,
  StackFramesById,
  AlertType,
  Alert,
} from './debugger_types';
import {
  createAlertsState,
  createDebuggerExecutionsState,
  createDebuggerGraphExecutionsState,
  createDebuggerGraphsState,
  createDebuggerSourceCodeState,
  createDebuggerState,
  createDebuggerStateWithLoadedExecutionDigests,
  createDigestsStateWhileLoadingExecutionDigests,
  createTestExecutionData,
  createTestGraphExecution,
  createTestGraphOpInfo,
  createTestStackFrame,
  createTestInfNanAlert,
} from '../testing';

describe('Debugger reducers', () => {
  describe('alertTypeFocusToggled', () => {
    for (const focusType of [
      AlertType.FUNCTION_RECOMPILE_ALERT,
      AlertType.INF_NAN_ALERT,
      AlertType.TENSOR_SHAPE_ALERT,
    ]) {
      it(`sets correct focusType (${focusType}) from no-focus initial state`, () => {
        const state = createDebuggerState();
        const nextState = reducers(
          state,
          actions.alertTypeFocusToggled({
            alertType: focusType,
          })
        );
        expect(nextState.alerts.focusType).toBe(focusType);
      });
    }

    it('sets focusType to new type from non-empty focus state', () => {
      const state = createDebuggerState({
        alerts: createAlertsState({
          focusType: AlertType.FUNCTION_RECOMPILE_ALERT,
        }),
      });
      const nextState = reducers(
        state,
        actions.alertTypeFocusToggled({
          alertType: AlertType.INF_NAN_ALERT,
        })
      );
      expect(nextState.alerts.focusType).toBe(AlertType.INF_NAN_ALERT);
    });

    it('sets focusType to null from non-empty state', () => {
      const state = createDebuggerState({
        alerts: createAlertsState({
          focusType: AlertType.INF_NAN_ALERT,
        }),
      });
      const nextState = reducers(
        state,
        actions.alertTypeFocusToggled({
          alertType: AlertType.INF_NAN_ALERT,
        })
      );
      expect(nextState.alerts.focusType).toBeNull();
    });

    for (const {displayCount, expectedScrollBegin} of [
      {displayCount: 5, expectedScrollBegin: 8},
      {displayCount: 40, expectedScrollBegin: 0},
    ]) {
      it('scrolls to execution digest corresponding to first alert', () => {
        const state = createDebuggerState({
          alerts: createAlertsState({
            focusType: null,
            executionIndices: {
              [AlertType.INF_NAN_ALERT]: [10, 11],
            },
          }),
          executions: createDebuggerExecutionsState({
            scrollBeginIndex: 0,
            displayCount,
          }),
        });
        const nextState = reducers(
          state,
          actions.alertTypeFocusToggled({
            alertType: AlertType.INF_NAN_ALERT,
          })
        );
        expect(nextState.executions.scrollBeginIndex).toBe(expectedScrollBegin);
      });
    }
  });

  describe('Runs loading', () => {
    it('sets runsLoaded to loading on requesting runs', () => {
      const state = createDebuggerState();
      const nextState = reducers(state, actions.debuggerRunsRequested());
      expect(nextState.runsLoaded.state).toBe(DataLoadState.LOADING);
      expect(nextState.runsLoaded.lastLoadedTimeInMs).toBeNull();
    });

    it('set runsLoad to failed on request failure', () => {
      const state = createDebuggerState({
        runsLoaded: {state: DataLoadState.LOADING, lastLoadedTimeInMs: null},
      });
      const nextState = reducers(state, actions.debuggerRunsRequestFailed());
      expect(nextState.runsLoaded.state).toBe(DataLoadState.FAILED);
      expect(nextState.runsLoaded.lastLoadedTimeInMs).toBeNull();
    });

    it('sets runsLoaded and runs on successful runs loading', () => {
      const state = createDebuggerState();
      const t0 = Date.now();
      const nextState = reducers(
        state,
        actions.debuggerRunsLoaded({
          runs: {
            foo_debugger_run: {
              start_time: 111,
            },
          },
        })
      );
      expect(nextState.runsLoaded.state).toBe(DataLoadState.LOADED);
      expect(nextState.runsLoaded.lastLoadedTimeInMs).toBeGreaterThanOrEqual(
        t0
      );
      expect(nextState.runs).toEqual({
        foo_debugger_run: {
          start_time: 111,
        },
      });
    });

    it('Overrides existing runs on successful runs loading', () => {
      const state = createDebuggerState({
        runs: {
          foo_debugger_run: {
            start_time: 111,
          },
        },
        runsLoaded: {
          state: DataLoadState.LOADED,
          lastLoadedTimeInMs: 0,
        },
      });
      const t0 = Date.now();
      const nextState = reducers(
        state,
        actions.debuggerRunsLoaded({
          runs: {
            bar_debugger_run: {
              start_time: 222,
            },
          },
        })
      );
      expect(nextState.runsLoaded.state).toBe(DataLoadState.LOADED);
      expect(nextState.runsLoaded.lastLoadedTimeInMs).toBeGreaterThanOrEqual(
        t0
      );
      expect(nextState.runs).toEqual({
        bar_debugger_run: {
          start_time: 222,
        },
      });
    });
  });

  it('Overrides activeRunId on debuggerRunsLoaded', () => {
    const state = createDebuggerState();
    const nextState = reducers(
      state,
      actions.debuggerRunsLoaded({
        runs: {
          __default_debugger_run__: {
            start_time: 222,
          },
        },
      })
    );
    expect(nextState.activeRunId).toEqual('__default_debugger_run__');
  });

  it('Updates alert load state on numAlertsAndBreakdownRequested', () => {
    const state = createDebuggerState({
      activeRunId: '__default_debugger_run__',
    });
    const nextState = reducers(state, actions.numAlertsAndBreakdownRequested());
    expect(nextState.alerts.alertsLoaded.state).toEqual(DataLoadState.LOADING);
    expect(nextState.alerts.alertsLoaded.lastLoadedTimeInMs).toBeNull();
  });

  it('Updates on numAlertsAndBreakdownLoaded', () => {
    const state = createDebuggerState({
      activeRunId: '__default_debugger_run__',
      alerts: createAlertsState({
        alertsLoaded: {
          state: DataLoadState.LOADING,
          lastLoadedTimeInMs: null,
        },
      }),
    });
    const nextState = reducers(
      state,
      actions.numAlertsAndBreakdownLoaded({
        numAlerts: 30,
        alertsBreakdown: {
          InfNanAlerts: 29,
          FunctionRecompileAlerts: 1,
        },
      })
    );
    expect(nextState.alerts.alertsLoaded.state).toEqual(DataLoadState.LOADED);
    expect(nextState.alerts.alertsLoaded.lastLoadedTimeInMs).toBeGreaterThan(0);
    expect(nextState.alerts.numAlerts).toBe(30);
    expect(nextState.alerts.alertsBreakdown).toEqual({
      InfNanAlerts: 29,
      FunctionRecompileAlerts: 1,
    });
  });

  describe('alertsOfTypeLoaded', () => {
    for (const {displayCount, expectedScrollBegin} of [
      {
        displayCount: 4,
        expectedScrollBegin: 8,
      },
      {
        displayCount: 30,
        expectedScrollBegin: 0,
      },
    ]) {
      it('Updates alerts data and scrollBeginIndex: empty initial state', () => {
        const firstAlertExecutionIndex = 10;
        const state = createDebuggerState({
          activeRunId: '__default_debugger_run__',
          alerts: createAlertsState({
            alertsLoaded: {
              state: DataLoadState.LOADING,
              lastLoadedTimeInMs: null,
            },
          }),
          executions: createDebuggerExecutionsState({
            displayCount,
            scrollBeginIndex: 0,
          }),
        }); // `alerts` state is in an empty initial state.
        const alert0 = createTestInfNanAlert({
          op_type: 'RealDiv',
          execution_index: firstAlertExecutionIndex,
        });
        const alert1 = createTestInfNanAlert({
          op_type: 'Log',
          execution_index: firstAlertExecutionIndex + 1,
        });
        const nextState = reducers(
          state,
          actions.alertsOfTypeLoaded({
            numAlerts: 2,
            alertsBreakdown: {
              InfNanAlert: 2,
            },
            begin: 0,
            end: 2,
            alertType: 'InfNanAlert',
            alerts: [alert0, alert1],
          })
        );
        expect(nextState.alerts.alertsLoaded.state).toBe(DataLoadState.LOADED);
        expect(
          nextState.alerts.alertsLoaded.lastLoadedTimeInMs
        ).toBeGreaterThan(0);
        expect(nextState.alerts.numAlerts).toBe(2);
        expect(nextState.alerts.alertsBreakdown).toEqual({
          [AlertType.INF_NAN_ALERT]: 2,
        });
        expect(Object.keys(nextState.alerts.alerts)).toEqual([
          AlertType.INF_NAN_ALERT,
        ]);
        const alertsOfType = nextState.alerts.alerts[AlertType.INF_NAN_ALERT];
        expect(Object.keys(alertsOfType).length).toBe(2);
        expect(alertsOfType[0]).toEqual(alert0);
        expect(alertsOfType[1]).toEqual(alert1);
        expect(Object.keys(nextState.alerts.executionIndices)).toEqual([
          AlertType.INF_NAN_ALERT,
        ]);
        const executionIndices =
          nextState.alerts.executionIndices[AlertType.INF_NAN_ALERT];
        expect(executionIndices).toEqual([
          firstAlertExecutionIndex,
          firstAlertExecutionIndex + 1,
        ]);
        // Verify that the first alert is scrolled into view.
        expect(nextState.executions.scrollBeginIndex).toEqual(
          expectedScrollBegin
        );
      });
    }

    it('Updates alerts data: non-empty initial state', () => {
      const alert0 = createTestInfNanAlert({
        op_type: 'RealDiv',
        execution_index: 10,
      });
      const alert1 = createTestInfNanAlert({
        op_type: 'Log',
        execution_index: 11,
      });
      const state = createDebuggerState({
        activeRunId: '__default_debugger_run__',
        alerts: createAlertsState({
          alertsLoaded: {
            state: DataLoadState.LOADING,
            lastLoadedTimeInMs: null,
          },
          numAlerts: 1,
          alertsBreakdown: {[AlertType.INF_NAN_ALERT]: 1},
          alerts: {
            [AlertType.INF_NAN_ALERT]: {0: alert0},
          },
        }),
      }); // `alerts` state is in a non-empty initial state.

      const nextState = reducers(
        state,
        actions.alertsOfTypeLoaded({
          numAlerts: 2,
          alertsBreakdown: {
            InfNanAlert: 2,
          },
          begin: 1,
          end: 2,
          alertType: 'InfNanAlert',
          alerts: [alert1],
        })
      );
      expect(nextState.alerts.alertsLoaded.state).toBe(DataLoadState.LOADED);
      expect(nextState.alerts.alertsLoaded.lastLoadedTimeInMs).toBeGreaterThan(
        0
      );
      expect(nextState.alerts.numAlerts).toBe(2);
      expect(nextState.alerts.alertsBreakdown).toEqual({
        [AlertType.INF_NAN_ALERT]: 2,
      });
      expect(Object.keys(nextState.alerts.alerts)).toEqual([
        AlertType.INF_NAN_ALERT,
      ]);
      const alertsOfType = nextState.alerts.alerts[AlertType.INF_NAN_ALERT];
      expect(Object.keys(alertsOfType).length).toBe(2);
      expect(alertsOfType[0]).toEqual(alert0);
      expect(alertsOfType[1]).toEqual(alert1);
      expect(Object.keys(nextState.alerts.executionIndices)).toEqual([
        AlertType.INF_NAN_ALERT,
      ]);
      const executionIndices =
        nextState.alerts.executionIndices[AlertType.INF_NAN_ALERT];
      expect(executionIndices[1]).toBe(11);
    });

    it('Updates alerts data: existing alert types other than the loaded', () => {
      const alert0 = createTestInfNanAlert({
        op_type: 'RealDiv',
        execution_index: 10,
      });
      const alert1 = createTestInfNanAlert({
        op_type: 'Log',
        execution_index: 11,
      });
      const tensorShapeAlert: Alert = {
        alert_type: AlertType.TENSOR_SHAPE_ALERT,
      };
      const state = createDebuggerState({
        activeRunId: '__default_debugger_run__',
        alerts: createAlertsState({
          alertsLoaded: {
            state: DataLoadState.LOADING,
            lastLoadedTimeInMs: null,
          },
          numAlerts: 2,
          alertsBreakdown: {
            [AlertType.INF_NAN_ALERT]: 1,
            [AlertType.TENSOR_SHAPE_ALERT]: 1,
          },
          alerts: {
            [AlertType.INF_NAN_ALERT]: {0: alert0},
            [AlertType.TENSOR_SHAPE_ALERT]: {0: tensorShapeAlert},
          },
        }),
      });

      const nextState = reducers(
        state,
        actions.alertsOfTypeLoaded({
          numAlerts: 3,
          alertsBreakdown: {
            [AlertType.INF_NAN_ALERT]: 2,
            [AlertType.TENSOR_SHAPE_ALERT]: 1,
          },
          begin: 1,
          end: 2,
          alertType: 'InfNanAlert',
          alerts: [alert1],
        })
      );
      expect(nextState.alerts.numAlerts).toBe(3);
      expect(nextState.alerts.alertsBreakdown).toEqual({
        [AlertType.INF_NAN_ALERT]: 2,
        [AlertType.TENSOR_SHAPE_ALERT]: 1,
      });
      expect(Object.keys(nextState.alerts.alerts).length).toBe(2);
      const infNanAlerts = nextState.alerts.alerts[AlertType.INF_NAN_ALERT];
      expect(Object.keys(infNanAlerts).length).toBe(2);
      expect(infNanAlerts[0]).toEqual(alert0);
      expect(infNanAlerts[1]).toEqual(alert1);
      const tensorShapeAlerts =
        nextState.alerts.alerts[AlertType.TENSOR_SHAPE_ALERT];
      expect(Object.keys(tensorShapeAlerts).length).toBe(1);
      expect(tensorShapeAlerts[0]).toEqual(tensorShapeAlert);
    });
  });

  it('Updates load state on numExecutionsRequested', () => {
    const state = createDebuggerState({
      runs: {
        __default_debugger_run__: {
          start_time: 111,
        },
      },
      runsLoaded: {
        state: DataLoadState.LOADED,
        lastLoadedTimeInMs: 222,
      },
      activeRunId: '__default_debugger_run__',
    });
    const nextState = reducers(state, actions.numExecutionsRequested());
    expect(nextState.executions.numExecutionsLoaded.state).toBe(
      DataLoadState.LOADING
    );
    expect(
      nextState.executions.numExecutionsLoaded.lastLoadedTimeInMs
    ).toBeNull();
  });

  it('Updates states correctly on numExecutionsLoaded: non-empty', () => {
    const state = createDebuggerState({
      runs: {
        __default_debugger_run__: {
          start_time: 111,
        },
      },
      runsLoaded: {
        state: DataLoadState.LOADED,
        lastLoadedTimeInMs: 222,
      },
      activeRunId: '__default_debugger_run__',
      executions: {
        numExecutionsLoaded: {
          state: DataLoadState.LOADING,
          lastLoadedTimeInMs: null,
        },
        executionDigestsLoaded: {
          state: DataLoadState.NOT_LOADED,
          lastLoadedTimeInMs: null,
          pageLoadedSizes: {},
          numExecutions: 0,
        },
        pageSize: 1000,
        displayCount: 50,
        focusIndex: null,
        scrollBeginIndex: 0,
        executionDigests: {},
        executionData: {},
      },
    });
    const t0 = Date.now();
    const nextState = reducers(
      state,
      actions.numExecutionsLoaded({numExecutions: 1337})
    );
    expect(nextState.executions.numExecutionsLoaded.state).toBe(
      DataLoadState.LOADED
    );
    expect(
      nextState.executions.numExecutionsLoaded.lastLoadedTimeInMs
    ).toBeGreaterThanOrEqual(t0);
    expect(nextState.executions.executionDigestsLoaded.numExecutions).toBe(
      1337
    );
    expect(nextState.executions.focusIndex).toBe(0);
  });

  it('Updates states correctly on numExecutionsLoaded: empty', () => {
    const state = createDebuggerState({
      runs: {
        __default_debugger_run__: {
          start_time: 111,
        },
      },
      runsLoaded: {
        state: DataLoadState.LOADED,
        lastLoadedTimeInMs: 222,
      },
      activeRunId: '__default_debugger_run__',
      executions: {
        numExecutionsLoaded: {
          state: DataLoadState.LOADING,
          lastLoadedTimeInMs: null,
        },
        executionDigestsLoaded: {
          state: DataLoadState.NOT_LOADED,
          lastLoadedTimeInMs: null,
          pageLoadedSizes: {},
          numExecutions: 0,
        },
        pageSize: 1000,
        displayCount: 50,
        focusIndex: null,
        scrollBeginIndex: 0,
        executionDigests: {},
        executionData: {},
      },
    });
    const t0 = Date.now();
    const nextState = reducers(
      state,
      actions.numExecutionsLoaded({numExecutions: 0})
    );
    expect(nextState.executions.numExecutionsLoaded.state).toBe(
      DataLoadState.LOADED
    );
    expect(
      nextState.executions.numExecutionsLoaded.lastLoadedTimeInMs
    ).toBeGreaterThanOrEqual(t0);
    expect(nextState.executions.executionDigestsLoaded.numExecutions).toBe(0);
    expect(nextState.executions.focusIndex).toBeNull();
  });

  it('Updates states on executionDigestsRequested', () => {
    const state = createDebuggerState({
      runs: {
        __default_debugger_run__: {
          start_time: 111,
        },
      },
      runsLoaded: {
        state: DataLoadState.LOADED,
        lastLoadedTimeInMs: 222,
      },
      activeRunId: '__default_debugger_run__',
      executions: createDebuggerExecutionsState({
        numExecutionsLoaded: {
          state: DataLoadState.LOADED,
          lastLoadedTimeInMs: 333,
        },
        executionDigestsLoaded: {
          numExecutions: 1337,
          pageLoadedSizes: {},
          state: DataLoadState.NOT_LOADED,
          lastLoadedTimeInMs: null,
        },
      }),
    });
    const nextState = reducers(state, actions.executionDigestsRequested());
    expect(nextState.executions.executionDigestsLoaded.state).toBe(
      DataLoadState.LOADING
    );
    expect(
      nextState.executions.executionDigestsLoaded.lastLoadedTimeInMs
    ).toBeNull();
    expect(nextState.executions.executionDigestsLoaded.numExecutions).toBe(
      1337
    );
    expect(nextState.executions.executionDigestsLoaded.pageLoadedSizes).toEqual(
      {}
    );
  });

  it('On executionDigestsLoaded: correct digests & page sizes updates', () => {
    const pageSize = 100;
    const numExecutions = 1337;
    const state = createDigestsStateWhileLoadingExecutionDigests(
      pageSize,
      numExecutions
    );
    const excutionDigestsResponse: ExecutionDigestsResponse = {
      begin: 0,
      end: pageSize,
      num_digests: numExecutions,
      execution_digests: [],
    };
    for (let i = 0; i < 100; ++i) {
      excutionDigestsResponse.execution_digests.push({
        op_type: `Op${i}`,
        output_tensor_device_ids: [`de${i}`],
      });
    }
    const t0 = Date.now();
    const nextState = reducers(
      state,
      actions.executionDigestsLoaded(excutionDigestsResponse)
    );
    expect(nextState.executions.executionDigestsLoaded.state).toBe(
      DataLoadState.LOADED
    );
    expect(
      nextState.executions.executionDigestsLoaded.lastLoadedTimeInMs
    ).toBeGreaterThanOrEqual(t0);
    expect(nextState.executions.executionDigestsLoaded.numExecutions).toEqual(
      numExecutions
    );
    // The first page is loaded in full.
    expect(nextState.executions.executionDigestsLoaded.pageLoadedSizes).toEqual(
      {0: 100}
    );
    // The detailed digest data should be recorded.
    expect(Object.keys(nextState.executions.executionDigests).length).toBe(100);
    for (let i = 0; i < 100; ++i) {
      expect(nextState.executions.executionDigests[i].op_type).toBe(`Op${i}`);
      expect(
        nextState.executions.executionDigests[i].output_tensor_device_ids
      ).toEqual([`de${i}`]);
    }
  });

  it('On executionDigestsLoaded: Incomplete 1st page --> larger 1st page', () => {
    const pageSize = 100;
    const numExecutions = 4;
    const state = createDigestsStateWhileLoadingExecutionDigests(
      pageSize,
      numExecutions,
      {
        0: {op_type: 'Relu', output_tensor_device_ids: ['a']},
        1: {op_type: 'Identity', output_tensor_device_ids: ['a']},
      },
      {
        0: 2 /* Previously loaded incomplete first page. */,
      }
    );
    const excutionDigestsResponse: ExecutionDigestsResponse = {
      begin: 0,
      end: 4,
      num_digests: numExecutions,
      execution_digests: [
        {op_type: 'MatMul', output_tensor_device_ids: ['a']},
        {op_type: 'BiasAdd', output_tensor_device_ids: ['a']},
        {op_type: 'Relu', output_tensor_device_ids: ['a']},
        {op_type: 'Identity', output_tensor_device_ids: ['a']},
      ],
    };
    const t0 = Date.now();
    const nextState = reducers(
      state,
      actions.executionDigestsLoaded(excutionDigestsResponse)
    );
    expect(nextState.executions.executionDigestsLoaded.state).toBe(
      DataLoadState.LOADED
    );
    expect(
      nextState.executions.executionDigestsLoaded.lastLoadedTimeInMs
    ).toBeGreaterThanOrEqual(t0);
    expect(nextState.executions.executionDigestsLoaded.numExecutions).toBe(
      numExecutions
    );
    // The first page is expanded.
    expect(nextState.executions.executionDigestsLoaded.pageLoadedSizes).toEqual(
      {0: 4}
    );
    // The detailed digest data should be recorded.
    expect(Object.keys(nextState.executions.executionDigests).length).toBe(4);
    expect(nextState.executions.executionDigests[0]).toEqual({
      op_type: 'MatMul',
      output_tensor_device_ids: ['a'],
    });
    expect(nextState.executions.executionDigests[1]).toEqual({
      op_type: 'BiasAdd',
      output_tensor_device_ids: ['a'],
    });
    expect(nextState.executions.executionDigests[2]).toEqual({
      op_type: 'Relu',
      output_tensor_device_ids: ['a'],
    });
    expect(nextState.executions.executionDigests[3]).toEqual({
      op_type: 'Identity',
      output_tensor_device_ids: ['a'],
    });
  });

  it('On executionDigestsLoaded: Adding a new page before existing', () => {
    const pageSize = 2;
    const numExecutions = 4;
    const state = createDigestsStateWhileLoadingExecutionDigests(
      pageSize,
      numExecutions,
      {
        2: {op_type: 'Relu', output_tensor_device_ids: ['a']},
        3: {op_type: 'Identity', output_tensor_device_ids: ['a']},
      },
      {
        1: 2 /* Previously loaded 2nd page. */,
      }
    );
    const excutionDigestsResponse: ExecutionDigestsResponse = {
      begin: 0,
      end: 2,
      num_digests: numExecutions,
      execution_digests: [
        {op_type: 'MatMul', output_tensor_device_ids: ['a']},
        {op_type: 'BiasAdd', output_tensor_device_ids: ['a']},
        {op_type: 'Relu', output_tensor_device_ids: ['a']},
        {op_type: 'Identity', output_tensor_device_ids: ['a']},
      ],
    };
    const t0 = Date.now();
    const nextState = reducers(
      state,
      actions.executionDigestsLoaded(excutionDigestsResponse)
    );
    expect(nextState.executions.executionDigestsLoaded.state).toBe(
      DataLoadState.LOADED
    );
    expect(
      nextState.executions.executionDigestsLoaded.lastLoadedTimeInMs
    ).toBeGreaterThanOrEqual(t0);
    expect(nextState.executions.executionDigestsLoaded.numExecutions).toBe(
      numExecutions
    );
    // The first page is expanded.
    expect(nextState.executions.executionDigestsLoaded.pageLoadedSizes).toEqual(
      {0: 2, 1: 2}
    );
    // The detailed digest data should be recorded.
    expect(Object.keys(nextState.executions.executionDigests).length).toEqual(
      4
    );
    expect(nextState.executions.executionDigests[0]).toEqual({
      op_type: 'MatMul',
      output_tensor_device_ids: ['a'],
    });
    expect(nextState.executions.executionDigests[1]).toEqual({
      op_type: 'BiasAdd',
      output_tensor_device_ids: ['a'],
    });
    expect(nextState.executions.executionDigests[2]).toEqual({
      op_type: 'Relu',
      output_tensor_device_ids: ['a'],
    });
    expect(nextState.executions.executionDigests[3]).toEqual({
      op_type: 'Identity',
      output_tensor_device_ids: ['a'],
    });
  });

  it('On executionDigestsLoaded: Adding a new page after existing', () => {
    const pageSize = 2;
    const numExecutions = 4;
    const state = createDigestsStateWhileLoadingExecutionDigests(
      pageSize,
      numExecutions,
      {
        0: {op_type: 'MatMul', output_tensor_device_ids: ['a']},
        1: {op_type: 'BiasAdd', output_tensor_device_ids: ['a']},
      },
      {
        0: 2 /* Previously loaded 1st page. */,
      }
    );
    const excutionDigestsResponse: ExecutionDigestsResponse = {
      begin: 2,
      end: 4,
      num_digests: numExecutions + 1, // Total execution count has updated.
      execution_digests: [
        {op_type: 'Relu', output_tensor_device_ids: ['a']},
        {op_type: 'Identity', output_tensor_device_ids: ['a']},
      ],
    };
    const t0 = Date.now();
    const nextState = reducers(
      state,
      actions.executionDigestsLoaded(excutionDigestsResponse)
    );
    expect(nextState.executions.executionDigestsLoaded.state).toBe(
      DataLoadState.LOADED
    );
    expect(
      nextState.executions.executionDigestsLoaded.lastLoadedTimeInMs
    ).toBeGreaterThanOrEqual(t0);
    // Update in total execution count should be reflected.
    expect(nextState.executions.executionDigestsLoaded.numExecutions).toBe(
      numExecutions + 1
    );
    // The first page is expanded.
    expect(nextState.executions.executionDigestsLoaded.pageLoadedSizes).toEqual(
      {0: 2, 1: 2}
    );
    // The detailed digest data should be recorded.
    expect(Object.keys(nextState.executions.executionDigests).length).toBe(4);
    expect(nextState.executions.executionDigests[0]).toEqual({
      op_type: 'MatMul',
      output_tensor_device_ids: ['a'],
    });
    expect(nextState.executions.executionDigests[1]).toEqual({
      op_type: 'BiasAdd',
      output_tensor_device_ids: ['a'],
    });
    expect(nextState.executions.executionDigests[2]).toEqual({
      op_type: 'Relu',
      output_tensor_device_ids: ['a'],
    });
    expect(nextState.executions.executionDigests[3]).toEqual({
      op_type: 'Identity',
      output_tensor_device_ids: ['a'],
    });
  });

  it('executionScrollLeft takes no effect due to left bound', () => {
    const scrollBeginIndex = 0;
    const state = createDebuggerStateWithLoadedExecutionDigests(
      scrollBeginIndex
    );
    const nextState = reducers(state, actions.executionScrollLeft());
    expect(nextState.executions.scrollBeginIndex).toBe(0);
  });

  for (const scrollBeginIndex of [1, 50, 100, 999, 1000, 1001, 1234, 1450]) {
    it(`executionScrollLeft takes effect: ${scrollBeginIndex}`, () => {
      const state = createDebuggerStateWithLoadedExecutionDigests(
        scrollBeginIndex
      );
      const nextState = reducers(state, actions.executionScrollLeft());
      expect(nextState.executions.scrollBeginIndex).toBe(scrollBeginIndex - 1);
    });
  }

  for (const scrollBeginIndex of [0, 1, 50, 100, 999, 1000, 1001, 1234, 1449]) {
    it(`executionScrollRight takes effect: ${scrollBeginIndex}`, () => {
      const state = createDebuggerStateWithLoadedExecutionDigests(
        scrollBeginIndex
      );
      const nextState = reducers(state, actions.executionScrollRight());
      expect(nextState.executions.scrollBeginIndex).toBe(scrollBeginIndex + 1);
    });
  }

  it('executionScrollRight takes no effect due to left bound', () => {
    const scrollBeginIndex = 1450;
    const state = createDebuggerStateWithLoadedExecutionDigests(
      scrollBeginIndex
    );
    const nextState = reducers(state, actions.executionScrollRight());
    expect(nextState.executions.scrollBeginIndex).toBe(1450);
  });

  for (const scrollIndex of [0, 1, 20, 50]) {
    it(`executionScrollToIndex sets correct scrollBeginIndex ${scrollIndex}`, () => {
      const scrollBeginIndex = 0;
      const displayCount = 50;
      const opTypes = new Array<string>(100);
      opTypes.fill('FooOp');
      const state = createDebuggerStateWithLoadedExecutionDigests(
        scrollBeginIndex,
        displayCount,
        opTypes
      );
      const nextState = reducers(
        state,
        actions.executionScrollToIndex({index: scrollIndex})
      );
      expect(nextState.executions.scrollBeginIndex).toBe(scrollIndex);
    });
  }

  for (const scrollIndex of [-1, 0.5, 51, 100]) {
    it(
      `Invalid executionScrollToIndex (${scrollIndex}) does not change scrollBeginIndex:` +
        `displayCount < numExecutions`,
      () => {
        const originalScrollBeginIndex = 3;
        const displayCount = 50;
        const opTypes = new Array<string>(100);
        opTypes.fill('FooOp');
        const state = createDebuggerStateWithLoadedExecutionDigests(
          originalScrollBeginIndex,
          displayCount,
          opTypes
        );
        expect(() =>
          reducers(state, actions.executionScrollToIndex({index: scrollIndex}))
        ).toThrow();
      }
    );
  }

  for (const scrollIndex of [-1, 0.5, 1, 2, 20]) {
    // In these tests, `displayCount` is 50 and there are only 20 execution digests
    // (< 50). Hence, the only valid scrolling begin index is 0.
    it(
      `Invalid executionScrollToIndex (${scrollIndex}) does not change scrollBeginIndex:` +
        `displayCount >= numExecutions`,
      () => {
        const originalScrollBeginIndex = 3;
        const displayCount = 50;
        const opTypes = new Array<string>(20);
        opTypes.fill('FooOp');
        const state = createDebuggerStateWithLoadedExecutionDigests(
          originalScrollBeginIndex,
          displayCount,
          opTypes
        );
        expect(() =>
          reducers(state, actions.executionScrollToIndex({index: scrollIndex}))
        ).toThrow();
      }
    );
  }

  it(`Updates states on executionDigestFocused: scrollBeginIndex = 0`, () => {
    const state = createDebuggerState();
    const nextState = reducers(
      state,
      actions.executionDigestFocused({
        displayIndex: 12,
      })
    );
    expect(nextState.executions.focusIndex).toBe(12);
  });

  it(`Updates states on executionDigestFocused: scrollBeginIndex > 0`, () => {
    const state = createDebuggerState({
      executions: {
        numExecutionsLoaded: {
          state: DataLoadState.LOADING,
          lastLoadedTimeInMs: null,
        },
        executionDigestsLoaded: {
          state: DataLoadState.NOT_LOADED,
          lastLoadedTimeInMs: null,
          pageLoadedSizes: {},
          numExecutions: 0,
        },
        pageSize: 1000,
        displayCount: 50,
        focusIndex: null,
        scrollBeginIndex: 100,
        executionDigests: {},
        executionData: {},
      },
    });
    const nextState = reducers(
      state,
      actions.executionDigestFocused({
        displayIndex: 12,
      })
    );
    expect(nextState.executions.focusIndex).toBe(112);
  });

  for (const beginIndex of [0, 50]) {
    for (const numObjects of [1, 2, 3]) {
      it(
        `Updates states on executionDataLoaded: beginIndex=${beginIndex}; ` +
          `${numObjects} objects`,
        () => {
          const state = createDebuggerState({
            activeRunId: '__default_debugger_run__',
          });
          const executionDataObjects: Execution[] = [];
          for (let i = 0; i < numObjects; ++i) {
            executionDataObjects.push(
              createTestExecutionData({
                op_type: `OpType${numObjects}`,
                input_tensor_ids: [i * 10],
                output_tensor_ids: [i * 10 + 1],
              })
            );
          }

          const nextState = reducers(
            state,
            actions.executionDataLoaded({
              begin: beginIndex,
              end: beginIndex + numObjects,
              executions: executionDataObjects,
            })
          );
          expect(Object.keys(nextState.executions.executionData).length).toBe(
            numObjects
          );
          for (let i = beginIndex; i < numObjects; ++i) {
            expect(nextState.executions.executionData[i]).toEqual(
              executionDataObjects[i]
            );
          }
        }
      );
    }
  }

  it(`Updates states on stackFramesLoaded with all new content`, () => {
    const state = createDebuggerState({
      activeRunId: '__default_debugger_run__',
    });
    const stackFrames: StackFramesById = {
      aaa: createTestStackFrame(),
      bbb: createTestStackFrame(),
      ccc: createTestStackFrame(),
    };
    const nextState = reducers(state, actions.stackFramesLoaded({stackFrames}));
    expect(nextState.stackFrames).toEqual(stackFrames);
  });

  it(`updates source-file list load state on sourceFileListRequested`, () => {
    const state = createDebuggerState();
    const nextState = reducers(state, actions.sourceFileListRequested());
    expect(nextState.sourceCode.sourceFileListLoaded.state).toBe(
      DataLoadState.LOADING
    );
    expect(
      nextState.sourceCode.sourceFileListLoaded.lastLoadedTimeInMs
    ).toBeNull();
  });

  it(`updates source-file list and load state sourceFileListLoaded`, () => {
    const state = createDebuggerState({
      sourceCode: createDebuggerSourceCodeState({
        sourceFileList: [
          {
            host_name: 'foo_host',
            file_path: '/tmp/main.py',
          },
        ],
      }),
    });
    const nextState = reducers(
      state,
      actions.sourceFileListLoaded({
        sourceFiles: [
          {
            host_name: 'foo_host',
            file_path: '/tmp/main.py',
          },
          {
            host_name: 'bar_host',
            file_path: '/tmp/train.py',
          },
        ],
      })
    );
    expect(nextState.sourceCode.sourceFileListLoaded.state).toBe(
      DataLoadState.LOADED
    );
    expect(
      nextState.sourceCode.sourceFileListLoaded.lastLoadedTimeInMs
    ).toBeGreaterThan(0);
    expect(nextState.sourceCode.sourceFileList).toEqual([
      {
        host_name: 'foo_host',
        file_path: '/tmp/main.py',
      },
      {
        host_name: 'bar_host',
        file_path: '/tmp/train.py',
      },
    ]);
    expect(nextState.sourceCode.fileContents).toEqual([
      {
        loadState: DataLoadState.NOT_LOADED,
        lines: null,
      },
      {
        loadState: DataLoadState.NOT_LOADED,
        lines: null,
      },
    ]);
  });

  it('updates focused line spec on sourceLineFocused', () => {
    const state = createDebuggerState();
    let nextState = reducers(
      state,
      actions.sourceLineFocused({
        sourceLineSpec: {
          host_name: 'localhost',
          file_path: '/tmp/main.py',
          lineno: 20,
        },
      })
    );
    expect(nextState.sourceCode.focusLineSpec).toEqual({
      host_name: 'localhost',
      file_path: '/tmp/main.py',
      lineno: 20,
    });
    nextState = reducers(
      state,
      actions.sourceLineFocused({
        sourceLineSpec: {
          host_name: 'localhost',
          file_path: '/tmp/train.py',
          lineno: 30,
        },
      })
    );
    expect(nextState.sourceCode.focusLineSpec).toEqual({
      host_name: 'localhost',
      file_path: '/tmp/train.py',
      lineno: 30,
    });
  });

  it(`updates file load state on sourceFileRequested: known file`, () => {
    const state = createDebuggerState({
      sourceCode: createDebuggerSourceCodeState({
        sourceFileList: [
          {
            host_name: 'foo_host',
            file_path: '/tmp/main.py',
          },
          {
            host_name: 'foo_host',
            file_path: '/tmp/model.py',
          },
        ],
        fileContents: [
          {
            loadState: DataLoadState.NOT_LOADED,
            lines: null,
          },
          {
            loadState: DataLoadState.NOT_LOADED,
            lines: null,
          },
        ],
      }),
    });
    const nextState = reducers(
      state,
      actions.sourceFileRequested({
        host_name: 'foo_host',
        file_path: '/tmp/model.py',
      })
    );

    expect(nextState.sourceCode.fileContents).toEqual([
      {
        loadState: DataLoadState.NOT_LOADED,
        lines: null,
      },
      {
        loadState: DataLoadState.LOADING,
        lines: null,
      },
    ]);
  });

  it(`throws error on sourceFileRequested: unknown file`, () => {
    const state = createDebuggerState({
      sourceCode: createDebuggerSourceCodeState({
        sourceFileList: [
          {
            host_name: 'foo_host',
            file_path: '/tmp/main.py',
          },
          {
            host_name: 'foo_host',
            file_path: '/tmp/model.py',
          },
        ],
        fileContents: [
          {
            loadState: DataLoadState.NOT_LOADED,
            lines: null,
          },
          {
            loadState: DataLoadState.NOT_LOADED,
            lines: null,
          },
        ],
      }),
    });
    expect(() =>
      reducers(
        state,
        actions.sourceFileRequested({
          host_name: 'foo_host',
          file_path: '/tmp/i_am_unknown.py',
        })
      )
    ).toThrowError(
      /Cannot find the following file.*\"\/tmp\/i_am_unknown\.py\"/
    );
    expect(state.sourceCode.fileContents).toEqual([
      {
        loadState: DataLoadState.NOT_LOADED,
        lines: null,
      },
      {
        loadState: DataLoadState.NOT_LOADED,
        lines: null,
      },
    ]);
  });

  it(`updates file load state & content on sourceFileLoaded: known file`, () => {
    const state = createDebuggerState({
      sourceCode: createDebuggerSourceCodeState({
        sourceFileList: [
          {
            host_name: 'foo_host',
            file_path: '/tmp/main.py',
          },
          {
            host_name: 'foo_host',
            file_path: '/tmp/model.py',
          },
        ],
        fileContents: [
          {
            loadState: DataLoadState.NOT_LOADED,
            lines: null,
          },
          {
            loadState: DataLoadState.LOADING,
            lines: null,
          },
        ],
      }),
    });
    const nextState = reducers(
      state,
      actions.sourceFileLoaded({
        host_name: 'foo_host',
        file_path: '/tmp/model.py',
        lines: [
          'import tensorflow as tf',
          '',
          'model = tf.keras.applications.MobilNetV2()',
        ],
      })
    );

    expect(nextState.sourceCode.fileContents).toEqual([
      {
        loadState: DataLoadState.NOT_LOADED,
        lines: null,
      },
      {
        loadState: DataLoadState.LOADED,
        lines: [
          'import tensorflow as tf',
          '',
          'model = tf.keras.applications.MobilNetV2()',
        ],
      },
    ]);
  });

  it(`throws error on sourceFileLoaded: unknown file`, () => {
    const state = createDebuggerState({
      sourceCode: createDebuggerSourceCodeState({
        sourceFileList: [
          {
            host_name: 'foo_host',
            file_path: '/tmp/main.py',
          },
          {
            host_name: 'foo_host',
            file_path: '/tmp/model.py',
          },
        ],
        fileContents: [
          {
            loadState: DataLoadState.NOT_LOADED,
            lines: null,
          },
          {
            loadState: DataLoadState.LOADING,
            lines: null,
          },
        ],
      }),
    });
    expect(() =>
      reducers(
        state,
        actions.sourceFileLoaded({
          host_name: 'foo_host',
          file_path: '/tmp/i_am_unknown.py',
          lines: ['print("Mysterious")'],
        })
      )
    ).toThrowError(
      /Cannot find the following file.*\"\/tmp\/i_am_unknown\.py\"/
    );
    expect(state.sourceCode.fileContents).toEqual([
      {
        loadState: DataLoadState.NOT_LOADED,
        lines: null,
      },
      {
        loadState: DataLoadState.LOADING,
        lines: null,
      },
    ]);
  });

  describe('numGraphExecutionsRequested', () => {
    it('updates load state', () => {
      const state = createDebuggerState({
        activeRunId: '__default_debugger_run__',
      });
      const nextState = reducers(state, actions.numGraphExecutionsRequested());
      expect(nextState.graphExecutions.numExecutionsLoaded.state).toBe(
        DataLoadState.LOADING
      );
    });
  });

  describe('numGraphExecutionsLoaded', () => {
    it('updates load state', () => {
      const state = createDebuggerState({
        activeRunId: '__default_debugger_run__',
        graphExecutions: createDebuggerGraphExecutionsState({
          numExecutionsLoaded: {
            state: DataLoadState.LOADING,
            lastLoadedTimeInMs: null,
          },
        }),
      });
      const t0 = Date.now();
      const nextState = reducers(
        state,
        actions.numGraphExecutionsLoaded({numGraphExecutions: 12345})
      );
      expect(nextState.graphExecutions.numExecutionsLoaded.state).toBe(
        DataLoadState.LOADED
      );
      expect(
        nextState.graphExecutions.numExecutionsLoaded.lastLoadedTimeInMs
      ).toBeGreaterThanOrEqual(t0);
      expect(
        nextState.graphExecutions.executionDigestsLoaded.numExecutions
      ).toEqual(12345);
    });
  });

  describe('graphExecutionDataRequested', () => {
    it('updates loading pages by adding a new one', () => {
      const state = createDebuggerState({
        activeRunId: '__default_debugger_run__',
        graphExecutions: createDebuggerGraphExecutionsState({
          graphExecutionDataLoadingPages: [2222, 7777],
        }),
      });
      const nextState = reducers(
        state,
        actions.graphExecutionDataRequested({pageIndex: 4321})
      );
      expect(nextState.graphExecutions.graphExecutionDataLoadingPages).toEqual([
        2222,
        7777,
        4321,
      ]);
    });
  });

  describe('graphExecutionDataLoaded', () => {
    it('with new data, updates loading pages, loaded pages and data', () => {
      const state = createDebuggerState({
        activeRunId: '__default_debugger_run__',
        graphExecutions: createDebuggerGraphExecutionsState({
          pageSize: 2,
          graphExecutionDataLoadingPages: [1, 2],
          graphExecutionDataPageLoadedSizes: {0: 2},
          graphExecutionData: {
            0: createTestGraphExecution({op_name: 'TestOp_0'}),
            1: createTestGraphExecution({op_name: 'TestOp_1'}),
          },
        }),
      });
      const nextState = reducers(
        state,
        actions.graphExecutionDataLoaded({
          begin: 2,
          end: 4,
          graph_executions: [
            createTestGraphExecution({op_name: 'TestOp_2'}),
            createTestGraphExecution({op_name: 'TestOp_3'}),
          ],
        })
      );
      expect(nextState.graphExecutions.graphExecutionDataLoadingPages).toEqual([
        2,
      ]);
      expect(
        nextState.graphExecutions.graphExecutionDataPageLoadedSizes
      ).toEqual({
        0: 2,
        1: 2,
      });
      expect(nextState.graphExecutions.graphExecutionData).toEqual({
        0: createTestGraphExecution({op_name: 'TestOp_0'}),
        1: createTestGraphExecution({op_name: 'TestOp_1'}),
        2: createTestGraphExecution({op_name: 'TestOp_2'}),
        3: createTestGraphExecution({op_name: 'TestOp_3'}),
      });
    });

    it('with partly new data, correctly updates pages and data', () => {
      const state = createDebuggerState({
        activeRunId: '__default_debugger_run__',
        graphExecutions: createDebuggerGraphExecutionsState({
          pageSize: 2,
          graphExecutionDataLoadingPages: [1],
          graphExecutionDataPageLoadedSizes: {0: 2, 1: 1},
          graphExecutionData: {
            0: createTestGraphExecution({op_name: 'TestOp_0'}),
            1: createTestGraphExecution({op_name: 'TestOp_1'}),
            2: createTestGraphExecution({op_name: 'TestOp_2'}),
          },
        }),
      });
      const nextState = reducers(
        state,
        actions.graphExecutionDataLoaded({
          begin: 2,
          end: 4,
          graph_executions: [
            createTestGraphExecution({op_name: 'TestOp_2_overwrite'}),
            createTestGraphExecution({op_name: 'TestOp_3_overwrite'}),
          ],
        })
      );
      expect(nextState.graphExecutions.graphExecutionDataLoadingPages).toEqual(
        []
      );
      expect(
        nextState.graphExecutions.graphExecutionDataPageLoadedSizes
      ).toEqual({
        0: 2,
        1: 2,
      });
      expect(nextState.graphExecutions.graphExecutionData).toEqual({
        0: createTestGraphExecution({op_name: 'TestOp_0'}),
        1: createTestGraphExecution({op_name: 'TestOp_1'}),
        2: createTestGraphExecution({op_name: 'TestOp_2_overwrite'}),
        3: createTestGraphExecution({op_name: 'TestOp_3_overwrite'}),
      });
    });
  });

  describe('graphExecutionScrollToIndex', () => {
    it('updates graph-execution scrollBeginIndex', () => {
      const state = createDebuggerState({
        graphExecutions: createDebuggerGraphExecutionsState({
          scrollBeginIndex: 0,
        }),
      });
      const nextState = reducers(
        state,
        actions.graphExecutionScrollToIndex({index: 1337})
      );
      expect(nextState.graphExecutions.scrollBeginIndex).toBe(1337);
    });

    for (const index of [-1, 8.8, Infinity, NaN]) {
      it(`throws error for invalid scroll index: ${index}`, () => {
        const state = createDebuggerState();
        expect(() =>
          reducers(state, actions.graphExecutionScrollToIndex({index}))
        ).toThrowError(/.*scroll.*negative or non-integer/);
      });
    }
  });

  describe('graphOpInfoLoaded', () => {
    it('updates self op, 1 input op and 1 consumer op', () => {
      const opInfo0 = createTestGraphOpInfo();
      const state = createDebuggerState({
        graphs: createDebuggerGraphsState({
          ops: {
            g0: {[opInfo0.op_name]: opInfo0},
          },
        }),
      });

      const opInfo1 = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
      });
      delete opInfo1.consumer_names;
      const opInfo2 = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
      });
      opInfo1.consumer_names = [[opInfo2.op_name]];
      opInfo2.input_names = [opInfo1.op_name];
      const opInfo3 = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
      });
      delete opInfo2.consumer_names;
      delete opInfo3.consumer_names;
      opInfo2.consumer_names = [[opInfo3.op_name]];
      opInfo3.input_names = [opInfo2.op_name];
      const opInfo2Response: GraphOpInfoResponse = {
        ...opInfo2,
        inputs: [opInfo1],
        consumers: [[opInfo3]],
      };
      const nextState = reducers(
        state,
        actions.graphOpInfoLoaded({graphOpInfoResponse: opInfo2Response})
      );

      expect(nextState.graphs.ops).toEqual({
        // Verify the old graph op data hasn't changed.
        g0: {[opInfo0.op_name]: opInfo0},
        // 'g2' is the immediately-enclosing graph of the three ops.
        g2: {
          [opInfo1.op_name]: opInfo1,
          [opInfo2.op_name]: opInfo2,
          [opInfo3.op_name]: opInfo3,
        },
      });
    });

    it('updates self op, 1 input op and 1 consumer op', () => {
      const opInfo0 = createTestGraphOpInfo();
      const state = createDebuggerState({
        graphs: createDebuggerGraphsState({
          ops: {
            g0: {[opInfo0.op_name]: opInfo0},
          },
        }),
      });

      const opInfo1a = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
      });
      delete opInfo1a.consumer_names;
      const opInfo1b = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
      });
      delete opInfo1b.consumer_names;
      const opInfo2 = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
      });
      opInfo1a.consumer_names = [[opInfo2.op_name]];
      opInfo1b.consumer_names = [[opInfo2.op_name]];
      opInfo2.input_names = [opInfo1a.op_name, opInfo1b.op_name];
      const opInfo3a = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
      });
      const opInfo3b = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
      });
      delete opInfo3a.consumer_names;
      delete opInfo3b.consumer_names;
      opInfo2.consumer_names = [[opInfo3a.op_name, opInfo3b.op_name]];
      opInfo3a.input_names = [opInfo2.op_name];
      opInfo3b.input_names = [opInfo2.op_name];
      const opInfo2Response: GraphOpInfoResponse = {
        ...opInfo2,
        inputs: [opInfo1a, opInfo1b],
        consumers: [[opInfo3a, opInfo3b]],
      };
      const nextState = reducers(
        state,
        actions.graphOpInfoLoaded({graphOpInfoResponse: opInfo2Response})
      );

      expect(nextState.graphs.ops).toEqual({
        // Verify the old graph op data hasn't changed.
        g0: {[opInfo0.op_name]: opInfo0},
        // 'g2' is the immediately-enclosing graph of the three ops.
        g2: {
          [opInfo1a.op_name]: opInfo1a,
          [opInfo1b.op_name]: opInfo1b,
          [opInfo2.op_name]: opInfo2,
          [opInfo3a.op_name]: opInfo3a,
          [opInfo3b.op_name]: opInfo3b,
        },
      });
    });

    it('updates self op and input op: no consumer op', () => {
      const opInfo0 = createTestGraphOpInfo();
      const state = createDebuggerState({
        graphs: createDebuggerGraphsState({
          ops: {
            g0: {[opInfo0.op_name]: opInfo0},
          },
        }),
      });

      const opInfo1 = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
      });
      delete opInfo1.consumer_names;
      const opInfo2 = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
        consumer_names: [],
      });
      opInfo1.consumer_names = [[opInfo2.op_name]];
      opInfo2.input_names = [opInfo1.op_name];
      const opInfo2Response: GraphOpInfoResponse = {
        ...opInfo2,
        inputs: [opInfo1],
        consumers: [],
      };
      const nextState = reducers(
        state,
        actions.graphOpInfoLoaded({graphOpInfoResponse: opInfo2Response})
      );

      expect(nextState.graphs.ops).toEqual({
        g0: {[opInfo0.op_name]: opInfo0},
        g2: {
          [opInfo1.op_name]: opInfo1,
          [opInfo2.op_name]: opInfo2,
        },
      });
    });

    it('updates self op and consumer ops: no input ops', () => {
      const opInfo0 = createTestGraphOpInfo();
      const state = createDebuggerState({
        graphs: createDebuggerGraphsState({
          ops: {
            g0: {[opInfo0.op_name]: opInfo0},
          },
        }),
      });

      const opInfo1 = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
        input_names: null,
      });
      delete opInfo1.consumer_names;
      const opInfo2 = createTestGraphOpInfo({
        graph_ids: ['g1', 'g2'],
      });
      opInfo1.consumer_names = [[opInfo2.op_name]];
      const opInfo1Response: GraphOpInfoResponse = {
        ...opInfo1,
        inputs: null,
        consumers: [[opInfo2]],
      };
      const nextState = reducers(
        state,
        actions.graphOpInfoLoaded({graphOpInfoResponse: opInfo1Response})
      );

      expect(nextState.graphs.ops).toEqual({
        g0: {[opInfo0.op_name]: opInfo0},
        g2: {
          [opInfo1.op_name]: opInfo1,
          [opInfo2.op_name]: opInfo2,
        },
      });
    });
  });
});
