/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

import {
  getAlertsBreakdown,
  getAlertsFocusType,
  getAlertsLoaded,
  getFocusedExecutionData,
  getFocusedExecutionIndex,
  getFocusedExecutionStackFrames,
  getFocusedSourceFileContent,
  getFocusedSourceFileIndex,
  getGraphExecutionDigests,
  getGraphExecutionDigestsLoaded,
  getGraphExecutionPageSize,
  getGraphExecutionScrollBeginIndex,
  getLoadedAlertsOfFocusedType,
  getNumAlerts,
  getNumAlertsOfFocusedType,
  getNumGraphExecutions,
  getNumGraphExecutionsLoaded,
  getFocusAlertTypesOfVisibleExecutionDigests,
  getSourceFileList,
  getSourceFileListLoaded,
} from './debugger_selectors';
import {
  AlertType,
  DataLoadState,
  DEBUGGER_FEATURE_KEY,
  StackFrame,
} from './debugger_types';
import {
  createAlertsState,
  createDebuggerGraphExecutionsState,
  createDebuggerSourceCodeState,
  createDebuggerState,
  createState,
  createTestExecutionData,
  createTestExecutionDigest,
  createTestInfNanAlert,
} from '../testing';

describe('debugger selectors', () => {
  describe('getAlertsLoaded', () => {
    it('returns correct NOT_LOADED state', () => {
      const state = createState(createDebuggerState());
      const alertsLoaded = getAlertsLoaded(state);
      expect(alertsLoaded.state).toBe(DataLoadState.NOT_LOADED);
      expect(alertsLoaded.lastLoadedTimeInMs).toBe(null);
    });

    it('returns correct LOADING state', () => {
      const state = createState(
        createDebuggerState({
          alerts: createAlertsState({
            alertsLoaded: {
              state: DataLoadState.LOADING,
              lastLoadedTimeInMs: null,
            },
          }),
        })
      );
      const alertsLoaded = getAlertsLoaded(state);
      expect(alertsLoaded.state).toBe(DataLoadState.LOADING);
      expect(alertsLoaded.lastLoadedTimeInMs).toBe(null);
    });
  });

  describe('getAlertsFocusType', () => {
    it('returns correct null state', () => {
      const state = createState(createDebuggerState());
      expect(getAlertsFocusType(state)).toBeNull();
    });

    it('returns correct non-null state', () => {
      const state = createState(
        createDebuggerState({
          alerts: createAlertsState({
            focusType: AlertType.INF_NAN_ALERT,
          }),
        })
      );
      expect(getAlertsFocusType(state)).toBe(AlertType.INF_NAN_ALERT);
    });
  });

  describe('getNumAlertsOfFocusedType', () => {
    for (const {focusType, expectedNumAlertsOfFocusedType} of [
      {
        focusType: null,
        expectedNumAlertsOfFocusedType: 0,
      },
      {
        focusType: AlertType.INF_NAN_ALERT,
        expectedNumAlertsOfFocusedType: 2,
      },
    ]) {
      it(`returns correct number of alerts in focus: focusType=${focusType}`, () => {
        const state = createState(
          createDebuggerState({
            alerts: createAlertsState({
              numAlerts: 2,
              focusType,
              alertsBreakdown: {
                [AlertType.INF_NAN_ALERT]: 2,
              },
              // NOTE: `alerts` is left blank here, to test that the return value of
              // getNumAlertsOfFocusedType() shouldn't depend on `alerts`, it should
              // depend on only `alertsBreakdown`.
            }),
          })
        );
        const numAlertsOfFocusedType = getNumAlertsOfFocusedType(state);
        expect(numAlertsOfFocusedType).toBe(expectedNumAlertsOfFocusedType);
      });
    }
  });

  describe('getLoadedAlertsOfFocusedType', () => {
    const alert0 = createTestInfNanAlert();
    const alert1 = createTestInfNanAlert();

    it('returns correct null when there is no focus', () => {
      const state = createState(
        createDebuggerState({
          alerts: createAlertsState({
            numAlerts: 2,
            focusType: null,
            alerts: {
              [AlertType.INF_NAN_ALERT]: {
                0: alert0,
                1: alert1,
              },
            },
          }),
        })
      );
      const loadedAlertsOfFocus = getLoadedAlertsOfFocusedType(state);
      expect(loadedAlertsOfFocus).toBeNull();
    });

    it('returns correct result when focus and data both exist', () => {
      const state = createState(
        createDebuggerState({
          alerts: createAlertsState({
            numAlerts: 2,
            focusType: AlertType.INF_NAN_ALERT,
            alerts: {
              [AlertType.INF_NAN_ALERT]: {
                0: alert0,
                1: alert1,
              },
            },
          }),
        })
      );
      const loadedAlertsOfFocus = getLoadedAlertsOfFocusedType(state);
      expect(loadedAlertsOfFocus).toEqual({
        0: alert0,
        1: alert1,
      });
    });
  });

  describe('getNumAlerts', () => {
    it('Returns correct zero numAlerts', () => {
      const state = createState(createDebuggerState());
      expect(getNumAlerts(state)).toBe(0);
    });

    it('Returns correct non-zero numAlerts', () => {
      const state = createState(
        createDebuggerState({
          alerts: createAlertsState({
            numAlerts: 95,
          }),
        })
      );
      expect(getNumAlerts(state)).toBe(95);
    });
  });

  describe('getAlertsBreakdown', () => {
    it('Returns correct empty object for initial state', () => {
      const state = createState(createDebuggerState());
      expect(getAlertsBreakdown(state)).toEqual({});
    });

    it('Returns correct non-empty map', () => {
      const state = createState(
        createDebuggerState({
          alerts: createAlertsState({
            numAlerts: 95,
            alertsBreakdown: {
              InfNanAlert: 50,
              FooAlert: 30,
              BarAlert: 15,
            },
          }),
        })
      );
      expect(getAlertsBreakdown(state)).toEqual({
        InfNanAlert: 50,
        FooAlert: 30,
        BarAlert: 15,
      });
    });
  });

  describe('getAlertTypesOfVisibleExecutionDigests', () => {
    it('returns all-null array when there is no focused alert type', () => {
      const state = createState(
        createDebuggerState({
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
            displayCount: 3,
            focusIndex: null,
            scrollBeginIndex: 0,
            executionDigests: {
              0: createTestExecutionDigest(),
              1: createTestExecutionDigest(),
              2: createTestExecutionDigest(),
            },
            executionData: {},
          },
        })
      );

      const alertTypes = getFocusAlertTypesOfVisibleExecutionDigests(state);
      expect(alertTypes).toEqual([null, null, null]);
    });

    it('returns correct non-null array when there is focused alert type', () => {
      const state = createState(
        createDebuggerState({
          activeRunId: '__default_debugger_run__',
          alerts: createAlertsState({
            focusType: AlertType.INF_NAN_ALERT,
            executionIndices: {
              [AlertType.INF_NAN_ALERT]: [0, 2, 3],
            },
          }),
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
            displayCount: 3,
            focusIndex: null,
            scrollBeginIndex: 0,
            executionDigests: {
              0: createTestExecutionDigest(),
              1: createTestExecutionDigest(),
              2: createTestExecutionDigest(),
              3: createTestExecutionDigest(),
            },
            executionData: {},
          },
        })
      );

      const alertTypes = getFocusAlertTypesOfVisibleExecutionDigests(state);
      expect(alertTypes).toEqual([
        AlertType.INF_NAN_ALERT,
        null,
        AlertType.INF_NAN_ALERT,
      ]);
    });
  });

  describe('getFocusedExecutionIndex', () => {
    for (const focusIndex of [null, 0, 1]) {
      it(`returns null correctly: focusIndex=${focusIndex}`, () => {
        const state = createState(
          createDebuggerState({
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
              focusIndex,
              scrollBeginIndex: 0,
              executionDigests: {},
              executionData: {},
            },
          })
        );
        expect(getFocusedExecutionIndex(state)).toBe(focusIndex);
      });
    }
  });

  describe('getFocusedExecutionData', () => {
    it('returns correct execution data in focus: null', () => {
      const state = createState(
        createDebuggerState({
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
        })
      );
      expect(getFocusedExecutionData(state)).toBe(null);
    });

    it('returns correct execution data in focus: data present, non-null', () => {
      const executionData = createTestExecutionData({op_type: 'FocusedOp'});
      const state = {
        [DEBUGGER_FEATURE_KEY]: createDebuggerState({
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
              numExecutions: 10,
            },
            pageSize: 1000,
            displayCount: 50,
            focusIndex: 1,
            scrollBeginIndex: 0,
            executionDigests: {},
            executionData: {
              0: createTestExecutionData(),
              1: executionData,
            },
          },
        }),
      };
      expect(getFocusedExecutionData(state)).toEqual(executionData);
    });

    it('returns correct execution data in focus: null due to data missing', () => {
      const state = createState(
        createDebuggerState({
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
              numExecutions: 10,
            },
            pageSize: 1000,
            displayCount: 50,
            focusIndex: 3,
            scrollBeginIndex: 0,
            executionDigests: {},
            executionData: {
              0: createTestExecutionData(),
            },
          },
        })
      );
      expect(getFocusedExecutionData(state)).toBe(null);
    });
  });

  describe('getFocusedExecutionStackFrames', () => {
    it('returns correct stack frames when there is no focus', () => {
      const state = createState(
        createDebuggerState({
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
        })
      );
      expect(getFocusedExecutionStackFrames(state)).toBe(null);
    });

    it('returns correct stack frames when there is no focus', () => {
      const stackFrame1: StackFrame = ['localhost', '/tmp/main.py', 10, 'main'];
      const stackFrame2: StackFrame = [
        'localhost',
        '/tmp/model.py',
        20,
        'initialize',
      ];
      const stackFrame3: StackFrame = [
        'localhost',
        '/tmp/model.py',
        30,
        'create_weight',
      ];

      const state = createState(
        createDebuggerState({
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
            focusIndex: 1,
            scrollBeginIndex: 0,
            executionDigests: {},
            executionData: {
              1: createTestExecutionData({
                stack_frame_ids: ['a1', 'a3'],
              }),
            },
          },
          stackFrames: {
            a1: stackFrame1,
            a2: stackFrame2,
            a3: stackFrame3,
          },
        })
      );
      expect(getFocusedExecutionStackFrames(state)).toEqual([
        stackFrame1,
        stackFrame3,
      ]);
    });

    it('returns null when subset of frames is missing', () => {
      const state = createState(
        createDebuggerState({
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
            focusIndex: 1,
            scrollBeginIndex: 0,
            executionDigests: {},
            executionData: {
              1: createTestExecutionData({
                stack_frame_ids: ['a1', 'a3'],
              }),
            },
          },
          stackFrames: {
            a1: ['localhost', '/tmp/main.py', 10, 'main'],
            a2: ['localhost', '/tmp/model.py', 20, 'initialize'],
          },
        })
      );
      expect(getFocusedExecutionStackFrames(state)).toBeNull();
    });
  });

  describe('getSourceFileListLoaded', () => {
    it('returns correct NOT_LOADED state', () => {
      const state = createState(createDebuggerState());
      const loaded = getSourceFileListLoaded(state);
      expect(loaded.state).toBe(DataLoadState.NOT_LOADED);
      expect(loaded.lastLoadedTimeInMs).toBeNull();
    });

    it('returns correct LOADING state', () => {
      const state = createState(
        createDebuggerState({
          sourceCode: createDebuggerSourceCodeState({
            sourceFileListLoaded: {
              state: DataLoadState.LOADING,
              lastLoadedTimeInMs: 4321,
            },
          }),
        })
      );
      const loaded = getSourceFileListLoaded(state);
      expect(loaded.state).toBe(DataLoadState.LOADING);
      expect(loaded.lastLoadedTimeInMs).toBe(4321);
    });

    it('returns correct LOADED state', () => {
      const state = createState(
        createDebuggerState({
          sourceCode: createDebuggerSourceCodeState({
            sourceFileListLoaded: {
              state: DataLoadState.LOADED,
              lastLoadedTimeInMs: 8888,
            },
          }),
        })
      );
      const loaded = getSourceFileListLoaded(state);
      expect(loaded.state).toBe(DataLoadState.LOADED);
      expect(loaded.lastLoadedTimeInMs).toBe(8888);
    });
  });

  describe('getSourceFileList', () => {
    it('returns correct empty array state', () => {
      const state = createState(createDebuggerState());
      const sourceFileList = getSourceFileList(state);
      expect(sourceFileList).toEqual([]);
    });

    it('returns correct non-empty array state', () => {
      const state = createState(
        createDebuggerState({
          sourceCode: createDebuggerSourceCodeState({
            sourceFileList: [
              {
                host_name: 'worker0',
                file_path: '/tmp/main.py',
              },
              {
                host_name: 'worker1',
                file_path: '/tmp/eval.py',
              },
            ],
          }),
        })
      );
      const sourceFileList = getSourceFileList(state);
      expect(sourceFileList).toEqual([
        {
          host_name: 'worker0',
          file_path: '/tmp/main.py',
        },
        {
          host_name: 'worker1',
          file_path: '/tmp/eval.py',
        },
      ]);
    });
  });

  describe('getFocusedSourceFileIndex', () => {
    it('returns correct -1 for no-focus initial state', () => {
      const state = createState(createDebuggerState());
      expect(getFocusedSourceFileIndex(state)).toBe(-1);
    });

    it('returns correct -1 for no-focus state with file list', () => {
      const state = createState(
        createDebuggerState({
          sourceCode: createDebuggerSourceCodeState({
            sourceFileList: [
              {
                host_name: 'worker0',
                file_path: '/tmp/main.py',
              },
            ],
            focusLineSpec: null,
          }),
        })
      );
      expect(getFocusedSourceFileIndex(state)).toBe(-1);
    });

    it('returns correct >=0 value', () => {
      const state = createState(
        createDebuggerState({
          sourceCode: createDebuggerSourceCodeState({
            sourceFileList: [
              {
                host_name: 'worker0',
                file_path: '/tmp/main.py',
              },
              {
                host_name: 'worker1',
                file_path: '/tmp/eval.py',
              },
            ],
            focusLineSpec: {
              host_name: 'worker1',
              file_path: '/tmp/eval.py',
              lineno: 100,
            },
          }),
        })
      );
      expect(getFocusedSourceFileIndex(state)).toBe(1);
    });
  });

  describe('getFocusedSourceFileContent', () => {
    it('returns correct null for no-focus initial state', () => {
      const state = createState(createDebuggerState());
      expect(getFocusedSourceFileContent(state)).toBeNull();
    });

    it('returns correct >=0 value', () => {
      const state = createState(
        createDebuggerState({
          sourceCode: createDebuggerSourceCodeState({
            sourceFileList: [
              {
                host_name: 'worker0',
                file_path: '/tmp/main.py',
              },
              {
                host_name: 'worker1',
                file_path: '/tmp/eval.py',
              },
            ],
            fileContents: [
              {
                loadState: DataLoadState.NOT_LOADED,
                lines: null,
              },
              {
                loadState: DataLoadState.LOADED,
                lines: ['', 'import tensorflow as tf'],
              },
            ],
            focusLineSpec: {
              host_name: 'worker1',
              file_path: '/tmp/eval.py',
              lineno: 100,
            },
          }),
        })
      );
      expect(getFocusedSourceFileContent(state)).toEqual({
        loadState: DataLoadState.LOADED,
        lines: ['', 'import tensorflow as tf'],
      });
    });
  });

  describe('getNumGraphExecutionsLoaded', () => {
    it('returns correct NOT_LOADED state', () => {
      const state = createState(createDebuggerState());
      const loaded = getNumGraphExecutionsLoaded(state);
      expect(loaded.state).toBe(DataLoadState.NOT_LOADED);
      expect(loaded.lastLoadedTimeInMs).toBe(null);
    });

    it('returns correct LOADING state', () => {
      const state = createState(
        createDebuggerState({
          graphExecutions: createDebuggerGraphExecutionsState({
            numExecutionsLoaded: {
              state: DataLoadState.LOADING,
              lastLoadedTimeInMs: null,
            },
          }),
        })
      );
      const loaded = getNumGraphExecutionsLoaded(state);
      expect(loaded.state).toBe(DataLoadState.LOADING);
      expect(loaded.lastLoadedTimeInMs).toBe(null);
    });

    it('returns correct LOADED state', () => {
      const state = createState(
        createDebuggerState({
          graphExecutions: createDebuggerGraphExecutionsState({
            numExecutionsLoaded: {
              state: DataLoadState.LOADED,
              lastLoadedTimeInMs: 1234,
            },
          }),
        })
      );
      const loaded = getNumGraphExecutionsLoaded(state);
      expect(loaded.state).toBe(DataLoadState.LOADED);
      expect(loaded.lastLoadedTimeInMs).toBe(1234);
    });
  });

  describe('getNumGraphExecutionsLoaded', () => {
    it('returns correct initial NOT_LOADED state', () => {
      const state = createState(createDebuggerState());
      const loaded = getGraphExecutionDigestsLoaded(state);
      expect(loaded.state).toBe(DataLoadState.NOT_LOADED);
      expect(loaded.lastLoadedTimeInMs).toBe(null);
      expect(loaded.pageLoadedSizes).toEqual({});
      expect(loaded.numExecutions).toEqual(0);
    });

    it('returns correct LOADING state', () => {
      const state = createState(
        createDebuggerState({
          graphExecutions: createDebuggerGraphExecutionsState({
            executionDigestsLoaded: {
              state: DataLoadState.LOADING,
              lastLoadedTimeInMs: null,
              pageLoadedSizes: {},
              numExecutions: 10,
            },
          }),
        })
      );
      const loaded = getGraphExecutionDigestsLoaded(state);
      expect(loaded.state).toBe(DataLoadState.LOADING);
      expect(loaded.lastLoadedTimeInMs).toBe(null);
      expect(loaded.pageLoadedSizes).toEqual({});
      expect(loaded.numExecutions).toEqual(10);
    });

    it('returns correct LOADED state', () => {
      const state = createState(
        createDebuggerState({
          graphExecutions: createDebuggerGraphExecutionsState({
            executionDigestsLoaded: {
              state: DataLoadState.LOADED,
              lastLoadedTimeInMs: 1234,
              pageLoadedSizes: {0: 10},
              numExecutions: 10,
            },
          }),
        })
      );
      const loaded = getGraphExecutionDigestsLoaded(state);
      expect(loaded.state).toBe(DataLoadState.LOADED);
      expect(loaded.lastLoadedTimeInMs).toBe(1234);
      expect(loaded.pageLoadedSizes).toEqual({0: 10});
      expect(loaded.numExecutions).toEqual(10);
    });
  });

  describe('getNumGraphExecutions', () => {
    it('returns correct initial zero state', () => {
      const state = createState(createDebuggerState());
      expect(getNumGraphExecutions(state)).toBe(0);
    });

    it('returns correct non-zero state', () => {
      const state = createState(
        createDebuggerState({
          graphExecutions: createDebuggerGraphExecutionsState({
            executionDigestsLoaded: {
              state: DataLoadState.LOADING,
              lastLoadedTimeInMs: null,
              pageLoadedSizes: {},
              numExecutions: 10,
            },
          }),
        })
      );
      expect(getNumGraphExecutions(state)).toBe(10);
    });
  });

  describe('getGraphExecutionScrollBeginIndex', () => {
    it('returns initial zero value', () => {
      const state = createState(createDebuggerState());
      expect(getGraphExecutionScrollBeginIndex(state)).toBe(0);
    });

    it('returns non-zero value', () => {
      const state = createState(
        createDebuggerState({
          graphExecutions: createDebuggerGraphExecutionsState({
            scrollBeginIndex: 1337,
          }),
        })
      );
      expect(getGraphExecutionScrollBeginIndex(state)).toBe(1337);
    });
  });

  describe('getGraphExecutionPageSize', () => {
    it('returns correct page size', () => {
      const state = createState(
        createDebuggerState({
          graphExecutions: createDebuggerGraphExecutionsState({
            pageSize: 2048,
          }),
        })
      );
      expect(getGraphExecutionPageSize(state)).toBe(2048);
    });
  });

  describe('getGraphExecutionDigests', () => {
    it('returns correct initial tempty state', () => {
      const state = createState(createDebuggerState());
      expect(getGraphExecutionDigests(state)).toEqual({});
    });

    it('returns correct graph execution digests', () => {
      const state = createState(
        createDebuggerState({
          graphExecutions: createDebuggerGraphExecutionsState({
            graphExecutionDigests: {
              20: {
                op_name: 'FooOp_1',
                op_type: 'FooOp',
                output_slot: 0,
                graph_id: 'deadbeef',
              },
            },
          }),
        })
      );
      expect(getGraphExecutionDigests(state)).toEqual({
        20: {
          op_name: 'FooOp_1',
          op_type: 'FooOp',
          output_slot: 0,
          graph_id: 'deadbeef',
        },
      });
    });
  });
});
