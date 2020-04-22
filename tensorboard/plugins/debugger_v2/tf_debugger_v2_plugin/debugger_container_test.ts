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
/**
 * Unit tests for the Debugger Container.
 */
import {CommonModule} from '@angular/common';
import {TestBed} from '@angular/core/testing';
import {By} from '@angular/platform-browser';

import {Store} from '@ngrx/store';
import {provideMockStore, MockStore} from '@ngrx/store/testing';

import {
  debuggerLoaded,
  executionScrollLeft,
  executionScrollRight,
  executionScrollToIndex,
  sourceLineFocused,
} from './actions';
import {DebuggerComponent} from './debugger_component';
import {DebuggerContainer} from './debugger_container';
import {DataLoadState, State, AlertType} from './store/debugger_types';
import {
  createAlertsState,
  createDebuggerState,
  createState,
  createDebuggerExecutionsState,
  createDebuggerStateWithLoadedExecutionDigests,
  createTestExecutionData,
  createTestStackFrame,
} from './testing';
import {AlertsModule} from './views/alerts/alerts_module';
import {ExecutionDataModule} from './views/execution_data/execution_data_module';
import {GraphExecutionsModule} from './views/graph_executions/graph_executions_module';
import {InactiveModule} from './views/inactive/inactive_module';
import {TimelineContainer} from './views/timeline/timeline_container';
import {SourceFilesModule} from './views/source_files/source_files_module';
import {StackTraceContainer} from './views/stack_trace/stack_trace_container';
import {StackTraceModule} from './views/stack_trace/stack_trace_module';
import {TimelineModule} from './views/timeline/timeline_module';

/** @typehack */ import * as _typeHackStore from '@ngrx/store';

describe('Debugger Container', () => {
  let store: MockStore<State>;
  let dispatchSpy: jasmine.Spy;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [DebuggerComponent, DebuggerContainer],
      imports: [
        AlertsModule,
        CommonModule,
        ExecutionDataModule,
        GraphExecutionsModule,
        InactiveModule,
        SourceFilesModule,
        StackTraceModule,
        TimelineModule,
      ],
      providers: [
        provideMockStore({
          initialState: createState(createDebuggerState()),
        }),
        DebuggerContainer,
        TimelineContainer,
      ],
    }).compileComponents();
    store = TestBed.get(Store);
    dispatchSpy = spyOn(store, 'dispatch');
  });

  it('renders debugger component initially with inactive component', () => {
    const fixture = TestBed.createComponent(DebuggerContainer);
    fixture.detectChanges();

    const inactiveElement = fixture.debugElement.query(
      By.css('tf-debugger-v2-inactive')
    );
    expect(inactiveElement).toBeTruthy();
    const alertsElement = fixture.debugElement.query(
      By.css('tf-debugger-v2-alerts')
    );
    expect(alertsElement).toBeNull();
  });

  it('rendering debugger component dispatches debuggeRunsRequested', () => {
    const fixture = TestBed.createComponent(DebuggerContainer);
    fixture.detectChanges();
    expect(dispatchSpy).toHaveBeenCalledWith(debuggerLoaded());
  });

  it('updates the UI to hide inactive component when store has 1 run', () => {
    const fixture = TestBed.createComponent(DebuggerContainer);
    fixture.detectChanges();

    store.setState(
      createState(
        createDebuggerState({
          runs: {
            foo_run: {
              start_time: 111,
            },
          },
          runsLoaded: {
            state: DataLoadState.LOADED,
            lastLoadedTimeInMs: Date.now(),
          },
        })
      )
    );
    fixture.detectChanges();

    const inactiveElement = fixture.debugElement.query(
      By.css('tf-debugger-v2-inactive')
    );
    expect(inactiveElement).toBeNull();
    const alertsElement = fixture.debugElement.query(
      By.css('tf-debugger-v2-alerts')
    );
    expect(alertsElement).toBeTruthy();
  });

  it('updates the UI to hide inactive component when store has no run', () => {
    const fixture = TestBed.createComponent(DebuggerContainer);
    fixture.detectChanges();

    store.setState(
      createState(
        createDebuggerState({
          runs: {},
          runsLoaded: {
            state: DataLoadState.LOADED,
            lastLoadedTimeInMs: Date.now(),
          },
        })
      )
    );
    fixture.detectChanges();

    const inactiveElement = fixture.debugElement.query(
      By.css('tf-debugger-v2-inactive')
    );
    expect(inactiveElement).toBeTruthy();
    const alertsElement = fixture.debugElement.query(
      By.css('tf-debugger-v2-alerts')
    );
    expect(alertsElement).toBeNull();
  });

  describe('Timeline module', () => {
    it('shows loading number of executions', () => {
      const fixture = TestBed.createComponent(TimelineContainer);
      fixture.detectChanges();

      store.setState(
        createState(
          createDebuggerState({
            runs: {},
            runsLoaded: {
              state: DataLoadState.LOADED,
              lastLoadedTimeInMs: Date.now(),
            },
            executions: createDebuggerExecutionsState({
              numExecutionsLoaded: {
                state: DataLoadState.LOADING,
                lastLoadedTimeInMs: null,
              },
            }),
          })
        )
      );
      fixture.detectChanges();

      const loadingElements = fixture.debugElement.queryAll(
        By.css('.loading-num-executions')
      );
      expect(loadingElements.length).toEqual(1);
    });

    it('hides loading number of executions', () => {
      const fixture = TestBed.createComponent(TimelineContainer);
      fixture.detectChanges();

      store.setState(
        createState(
          createDebuggerState({
            runs: {},
            runsLoaded: {
              state: DataLoadState.LOADED,
              lastLoadedTimeInMs: Date.now(),
            },
            executions: createDebuggerExecutionsState({
              numExecutionsLoaded: {
                state: DataLoadState.LOADED,
                lastLoadedTimeInMs: 111,
              },
            }),
          })
        )
      );
      fixture.detectChanges();

      const loadingElements = fixture.debugElement.queryAll(
        By.css('.loading-num-executions')
      );
      expect(loadingElements.length).toEqual(0);
    });

    it('shows correct display range for executions', () => {
      const fixture = TestBed.createComponent(TimelineContainer);
      fixture.detectChanges();

      const scrollBeginIndex = 977;
      const dislpaySize = 100;
      store.setState(
        createState(
          createDebuggerStateWithLoadedExecutionDigests(
            scrollBeginIndex,
            dislpaySize
          )
        )
      );
      fixture.detectChanges();

      const navigationPositionInfoElement = fixture.debugElement.query(
        By.css('.navigation-position-info')
      );
      expect(navigationPositionInfoElement.nativeElement.innerText).toBe(
        'Execution: 977 ~ 1076 of 1500'
      );
    });

    it('left-button click dispatches executionScrollLeft action', () => {
      const fixture = TestBed.createComponent(TimelineContainer);
      fixture.detectChanges();
      store.setState(
        createState(createDebuggerStateWithLoadedExecutionDigests(0, 50))
      );
      fixture.detectChanges();

      const leftBUtton = fixture.debugElement.query(
        By.css('.navigation-button-left')
      );
      leftBUtton.nativeElement.click();
      fixture.detectChanges();
      expect(dispatchSpy).toHaveBeenCalledWith(executionScrollLeft());
    });

    it('right-button click dispatches executionScrollRight action', () => {
      const fixture = TestBed.createComponent(TimelineContainer);
      fixture.detectChanges();
      store.setState(
        createState(createDebuggerStateWithLoadedExecutionDigests(0, 50))
      );
      fixture.detectChanges();

      const rightButton = fixture.debugElement.query(
        By.css('.navigation-button-right')
      );
      rightButton.nativeElement.click();
      fixture.detectChanges();
      expect(dispatchSpy).toHaveBeenCalledWith(executionScrollRight());
    });

    it('displays correct op names', () => {
      const fixture = TestBed.createComponent(TimelineContainer);
      fixture.detectChanges();
      const scrollBeginIndex = 100;
      const displayCount = 40;
      const opTypes: string[] = [];
      for (let i = 0; i < 200; ++i) {
        opTypes.push(`${i}Op`);
      }
      store.setState(
        createState(
          createDebuggerStateWithLoadedExecutionDigests(
            scrollBeginIndex,
            displayCount,
            opTypes
          )
        )
      );
      fixture.detectChanges();

      const executionDigests = fixture.debugElement.queryAll(
        By.css('.execution-digest')
      );
      expect(executionDigests.length).toEqual(40);
      const strLen = 1;
      for (let i = 0; i < 40; ++i) {
        expect(executionDigests[i].nativeElement.innerText).toEqual(
          opTypes[i + 100].slice(0, strLen)
        );
      }
    });

    it('displays correct InfNanAlert alert type', () => {
      const fixture = TestBed.createComponent(TimelineContainer);
      fixture.detectChanges();
      const scrollBeginIndex = 5;
      const displayCount = 4;
      const opTypes: string[] = [];
      for (let i = 0; i < 200; ++i) {
        opTypes.push(`${i}Op`);
      }
      const debuggerState = createDebuggerStateWithLoadedExecutionDigests(
        scrollBeginIndex,
        displayCount,
        opTypes
      );
      debuggerState.alerts = createAlertsState({
        focusType: AlertType.INF_NAN_ALERT,
        executionIndices: {
          [AlertType.INF_NAN_ALERT]: [
            4, // Outside the viewing window.
            6, // Inside the viewing window; same below.
            8,
          ],
        },
      });
      store.setState(createState(debuggerState));
      fixture.detectChanges();

      const digestsWithAlert = fixture.debugElement.queryAll(
        By.css('.execution-digest.InfNanAlert')
      );
      expect(digestsWithAlert.length).toBe(2);
      expect(digestsWithAlert[0].nativeElement.innerText).toEqual('6');
      expect(digestsWithAlert[1].nativeElement.innerText).toEqual('8');
    });

    for (const numExecutions of [1, 9, 10]) {
      it(`hides slider if # of executions ${numExecutions} <= display count`, () => {
        const fixture = TestBed.createComponent(TimelineContainer);
        fixture.detectChanges();
        const scrollBeginIndex = 0;
        const displayCount = 10;
        const opTypes: string[] = new Array<string>(numExecutions);
        opTypes.fill('MatMul');
        const debuggerState = createDebuggerStateWithLoadedExecutionDigests(
          scrollBeginIndex,
          displayCount,
          opTypes
        );
        store.setState(createState(debuggerState));
        fixture.detectChanges();

        const sliders = fixture.debugElement.queryAll(
          By.css('.timeline-slider')
        );
        expect(sliders.length).toBe(0);
      });
    }

    for (const numExecutions of [11, 12, 20]) {
      const displayCount = 10;
      for (const scrollBeginIndex of [0, 1, numExecutions - displayCount]) {
        it(
          `shows slider if # of executions ${numExecutions} > display count, ` +
            `scrollBeginIndex = ${scrollBeginIndex}`,
          () => {
            const fixture = TestBed.createComponent(TimelineContainer);
            fixture.detectChanges();
            const opTypes: string[] = new Array<string>(numExecutions);
            opTypes.fill('MatMul');
            const debuggerState = createDebuggerStateWithLoadedExecutionDigests(
              scrollBeginIndex,
              displayCount,
              opTypes
            );
            store.setState(createState(debuggerState));
            fixture.detectChanges();

            const sliders = fixture.debugElement.queryAll(
              By.css('.timeline-slider')
            );
            expect(sliders.length).toBe(1);
            const [slider] = sliders;
            expect(slider.attributes['aria-valuemin']).toBe('0');
            expect(slider.attributes['aria-valuemax']).toBe(
              String(numExecutions - displayCount)
            );
            expect(slider.attributes['aria-valuenow']).toBe(
              String(scrollBeginIndex)
            );
          }
        );
      }
    }
  });

  for (const scrollBeginIndex of [0, 1, 5]) {
    it(`changes slider dispatches executionToScrollIndex (${scrollBeginIndex})`, () => {
      const fixture = TestBed.createComponent(TimelineContainer);
      fixture.detectChanges();
      const numExecutions = 10;
      const displayCount = 5;
      const opTypes: string[] = new Array<string>(numExecutions);
      opTypes.fill('MatMul');
      const debuggerState = createDebuggerStateWithLoadedExecutionDigests(
        scrollBeginIndex,
        displayCount,
        opTypes
      );
      store.setState(createState(debuggerState));
      fixture.detectChanges();

      const slider = fixture.debugElement.query(By.css('.timeline-slider'));

      slider.triggerEventHandler('change', {value: scrollBeginIndex});
      fixture.detectChanges();
      expect(dispatchSpy).toHaveBeenCalledWith(
        executionScrollToIndex({index: scrollBeginIndex})
      );
    });
  }

  describe('Stack Trace module', () => {
    it('Shows non-empty stack frames correctly', () => {
      const fixture = TestBed.createComponent(StackTraceContainer);
      fixture.detectChanges();

      const stackFrame0 = createTestStackFrame();
      const stackFrame1 = createTestStackFrame();
      const stackFrame2 = createTestStackFrame();
      store.setState(
        createState(
          createDebuggerState({
            executions: {
              numExecutionsLoaded: {
                state: DataLoadState.LOADED,
                lastLoadedTimeInMs: 111,
              },
              executionDigestsLoaded: {
                state: DataLoadState.LOADED,
                lastLoadedTimeInMs: 222,
                pageLoadedSizes: {0: 100},
                numExecutions: 1000,
              },
              executionDigests: {},
              pageSize: 100,
              displayCount: 50,
              scrollBeginIndex: 90,
              focusIndex: 98,
              executionData: {
                98: createTestExecutionData({
                  stack_frame_ids: ['a0', 'a1', 'a2'],
                }),
              },
            },
            stackFrames: {
              a0: stackFrame0,
              a1: stackFrame1,
              a2: stackFrame2,
            },
          })
        )
      );
      fixture.detectChanges();

      const hostNameElement = fixture.debugElement.query(
        By.css('.stack-trace-host-name')
      );
      expect(hostNameElement.nativeElement.innerText).toEqual('(on localhost)');
      const stackFrameContainers = fixture.debugElement.queryAll(
        By.css('.stack-frame-container')
      );
      expect(stackFrameContainers.length).toEqual(3);

      const filePathElements = fixture.debugElement.queryAll(
        By.css('.stack-frame-file-path')
      );
      expect(filePathElements.length).toEqual(3);
      expect(filePathElements[0].nativeElement.innerText).toEqual(
        stackFrame0[1].slice(stackFrame0[1].lastIndexOf('/') + 1)
      );
      expect(filePathElements[0].nativeElement.title).toEqual(stackFrame0[1]);
      expect(filePathElements[1].nativeElement.innerText).toEqual(
        stackFrame1[1].slice(stackFrame1[1].lastIndexOf('/') + 1)
      );
      expect(filePathElements[1].nativeElement.title).toEqual(stackFrame1[1]);
      expect(filePathElements[2].nativeElement.innerText).toEqual(
        stackFrame2[1].slice(stackFrame2[1].lastIndexOf('/') + 1)
      );
      expect(filePathElements[2].nativeElement.title).toEqual(stackFrame2[1]);

      const linenoElements = fixture.debugElement.queryAll(
        By.css('.stack-frame-lineno')
      );
      expect(linenoElements.length).toEqual(3);
      expect(linenoElements[0].nativeElement.innerText).toEqual(
        `Line ${stackFrame0[2]}`
      );
      expect(linenoElements[1].nativeElement.innerText).toEqual(
        `Line ${stackFrame1[2]}`
      );
      expect(linenoElements[2].nativeElement.innerText).toEqual(
        `Line ${stackFrame2[2]}`
      );

      const functionElements = fixture.debugElement.queryAll(
        By.css('.stack-frame-function')
      );
      expect(functionElements.length).toEqual(3);
      expect(functionElements[0].nativeElement.innerText).toEqual(
        stackFrame0[3]
      );
      expect(functionElements[1].nativeElement.innerText).toEqual(
        stackFrame1[3]
      );
      expect(functionElements[2].nativeElement.innerText).toEqual(
        stackFrame2[3]
      );
    });

    it('Shows loading state when stack-trace data is unavailable', () => {
      const fixture = TestBed.createComponent(StackTraceContainer);
      fixture.detectChanges();

      store.setState(
        createState(
          createDebuggerState({
            executions: {
              numExecutionsLoaded: {
                state: DataLoadState.LOADED,
                lastLoadedTimeInMs: 111,
              },
              executionDigestsLoaded: {
                state: DataLoadState.LOADED,
                lastLoadedTimeInMs: 222,
                pageLoadedSizes: {0: 100},
                numExecutions: 1000,
              },
              executionDigests: {},
              pageSize: 100,
              displayCount: 50,
              scrollBeginIndex: 90,
              focusIndex: 98,
              executionData: {
                98: createTestExecutionData({
                  stack_frame_ids: ['a0', 'a1', 'a2'],
                }),
              },
            },
            stackFrames: {}, // Note the empty stackFrames field.
          })
        )
      );
      fixture.detectChanges();

      const stackFrameContainers = fixture.debugElement.queryAll(
        By.css('.stack-frame-container')
      );
      expect(stackFrameContainers.length).toEqual(0);
    });

    it('Emits sourceLineFocused when line number is clicked', () => {
      const fixture = TestBed.createComponent(StackTraceContainer);
      fixture.detectChanges();

      const stackFrame0 = createTestStackFrame();
      const stackFrame1 = createTestStackFrame();
      const stackFrame2 = createTestStackFrame();
      store.setState(
        createState(
          createDebuggerState({
            executions: {
              numExecutionsLoaded: {
                state: DataLoadState.LOADED,
                lastLoadedTimeInMs: 111,
              },
              executionDigestsLoaded: {
                state: DataLoadState.LOADED,
                lastLoadedTimeInMs: 222,
                pageLoadedSizes: {0: 100},
                numExecutions: 1000,
              },
              executionDigests: {},
              pageSize: 100,
              displayCount: 50,
              scrollBeginIndex: 90,
              focusIndex: 98,
              executionData: {
                98: createTestExecutionData({
                  stack_frame_ids: ['a0', 'a1', 'a2'],
                }),
              },
            },
            stackFrames: {
              a0: stackFrame0,
              a1: stackFrame1,
              a2: stackFrame2,
            },
          })
        )
      );
      fixture.detectChanges();

      const linenoElements = fixture.debugElement.queryAll(
        By.css('.stack-frame-lineno')
      );
      linenoElements[1].nativeElement.click();
      fixture.detectChanges();
      expect(dispatchSpy).toHaveBeenCalledWith(
        sourceLineFocused({
          sourceLineSpec: {
            host_name: stackFrame1[0],
            file_path: stackFrame1[1],
            lineno: stackFrame1[2],
          },
        })
      );
    });
  });
});
