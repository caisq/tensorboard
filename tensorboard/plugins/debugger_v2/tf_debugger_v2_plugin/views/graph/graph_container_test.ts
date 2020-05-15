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
/**
 * Unit tests for the the graph structure component and container.
 */
import {CommonModule} from '@angular/common';
import {TestBed} from '@angular/core/testing';
import {By} from '@angular/platform-browser';

import {Store} from '@ngrx/store';
import {MockStore, provideMockStore} from '@ngrx/store/testing';

import {DebuggerComponent} from '../../debugger_component';
import {DebuggerContainer} from '../../debugger_container';
import {
  State,
  GraphExecution,
  TensorDebugMode,
} from '../../store/debugger_types';
import {
  getFocusedGraphOpConsumers,
  getFocusedGraphOpInfo,
  getFocusedGraphOpInputs,
} from '../../store';
import {
  createDebuggerState,
  createState,
  createTestGraphExecution,
  createTestGraphOpInfo,
} from '../../testing';
import {GraphContainer} from './graph_container';
import {GraphModule} from './graph_module';

/** @typehack */ import * as _typeHackStore from '@ngrx/store';

fdescribe('Graph Container', () => {
  let store: MockStore<State>;

  const graphExecutionData: {[index: number]: GraphExecution} = {};
  for (let i = 0; i < 120; ++i) {
    graphExecutionData[i] = createTestGraphExecution({
      op_name: `TestOp_${i}`,
      op_type: `OpType_${i}`,
      tensor_debug_mode: TensorDebugMode.CONCISE_HEALTH,
      debug_tensor_value: [i, 100, 0, 0, 0],
    });
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [DebuggerComponent, DebuggerContainer],
      imports: [CommonModule, GraphModule],
      providers: [
        provideMockStore({
          initialState: createState(createDebuggerState()),
        }),
        DebuggerContainer,
      ],
    }).compileComponents();
    store = TestBed.inject<Store<State>>(Store) as MockStore<State>;
  });

  it('renders no-op-selected element and nothing else when no op is selected', () => {
    const fixture = TestBed.createComponent(GraphContainer);
    store.overrideSelector(getFocusedGraphOpInfo, null);
    fixture.detectChanges();

    const noOpFocused = fixture.debugElement.query(By.css('.no-op-focused'));
    expect(noOpFocused).not.toBeNull();
    expect(noOpFocused.nativeElement.innerText).toMatch(/No graph op selected/);
    const inputsContainer = fixture.debugElement.query(
      By.css('.inputs-container')
    );
    expect(inputsContainer).toBeNull();
    const selfOpContainer = fixture.debugElement.query(
      By.css('.self-op-container')
    );
    expect(selfOpContainer).toBeNull();
    const consumersContainer = fixture.debugElement.query(
      By.css('.consumers-container')
    );
    expect(consumersContainer).toBeNull();
  });

  for (const inputOutputSlot of [0, 1]) {
    for (const consumerInputSlot of [0, 1]) {
      it(
        `renders op with 1 input tensor and 1 consumer: data available; ` +
          `consumerInputSlot=${consumerInputSlot}]`,
        () => {
          const fixture = TestBed.createComponent(GraphContainer);
          const op1 = createTestGraphOpInfo({
            op_name: 'op1',
            op_type: 'InputOp',
          });
          const op2 = createTestGraphOpInfo({
            op_name: 'op2',
            op_type: 'SelfOp',
          });
          const op3 = createTestGraphOpInfo({
            op_name: 'op3',
            op_type: 'ConsumerOp',
          });
          op1.consumers = [
            [
              {
                op_name: 'op2',
                input_slot: 0,
              },
            ],
          ];
          op2.inputs = [
            {
              op_name: 'op1',
              output_slot: inputOutputSlot,
            },
          ];
          op2.consumers = [
            [
              {
                op_name: 'op3',
                input_slot: consumerInputSlot,
              },
            ],
          ];
          op3.inputs = [
            {
              op_name: 'op2',
              output_slot: 0,
            },
          ];
          store.overrideSelector(getFocusedGraphOpInfo, op2);
          store.overrideSelector(getFocusedGraphOpInputs, [
            {
              ...op2.inputs[0],
              data: op1,
            },
          ]);
          store.overrideSelector(getFocusedGraphOpConsumers, [
            [
              {
                ...op2.consumers[0][0],
                data: op3,
              },
            ],
          ]);

          fixture.detectChanges();

          const noOpFocused = fixture.debugElement.query(
            By.css('.no-op-focused')
          );
          expect(noOpFocused).toBeNull();
          // Check self op section.
          const selfOpContainer = fixture.debugElement.query(
            By.css('.self-op-container')
          );
          const selfOpName = selfOpContainer.query(By.css('.self-op-name'));
          expect(selfOpName.nativeElement.innerText).toBe('op2');
          const selfOpType = selfOpContainer.query(By.css('.op-type'));
          expect(selfOpType.nativeElement.innerText).toBe('SelfOp');
          // Check inputs section.
          const inputsContainer = fixture.debugElement.query(
            By.css('.inputs-container')
          );
          const inputSlotHeaders = inputsContainer.queryAll(
            By.css('.input-slot-header')
          );
          expect(inputSlotHeaders.length).toBe(1);
          expect(inputSlotHeaders[0].nativeElement.innerText).toBe(
            'Input slot 0:'
          );
          const inputOpNames = inputsContainer.queryAll(By.css('.op-name'));
          expect(inputOpNames.length).toBe(1);
          expect(inputOpNames[0].nativeElement.innerText).toBe('op1');
          const inputOutputSlots = inputsContainer.queryAll(By.css('.slot'));
          expect(inputOutputSlots.length).toBe(1);
          expect(inputOutputSlots[0].nativeElement.innerText).toBe(
            `Output slot: ${inputOutputSlot}`
          );
          const inputOpTypes = inputsContainer.queryAll(By.css('.op-type'));
          expect(inputOpTypes.length).toBe(1);
          expect(inputOpTypes[0].nativeElement.innerText).toBe('InputOp');
          // Check consumers section.
          const consumersContainer = fixture.debugElement.query(
            By.css('.consumers-container')
          );
          const slotConsumersContainers = consumersContainer.queryAll(
            By.css('.slot-consumers-container')
          );
          expect(slotConsumersContainers.length).toBe(1);
          const slotConsumersContainer = slotConsumersContainers[0];
          const slotConsumersHeader = slotConsumersContainer.queryAll(
            By.css('.slot-consumers-header')
          );
          expect(slotConsumersHeader.length).toBe(1);
          expect(slotConsumersHeader[0].nativeElement.innerText).toBe(
            'Output slot 0: (1 consumer)'
          );
          const consumerOpNames = slotConsumersContainer.queryAll(
            By.css('.op-name')
          );
          expect(consumerOpNames.length).toBe(1);
          expect(consumerOpNames[0].nativeElement.innerText).toEqual('op3');
          const consumerInputSlots = slotConsumersContainer.queryAll(
            By.css('.slot')
          );
          expect(consumerInputSlots.length).toBe(1);
          expect(consumerInputSlots[0].nativeElement.innerText).toBe(
            `Input slot: ${consumerInputSlot}`
          );
          const consumerOpTypes = slotConsumersContainer.queryAll(
            By.css('.op-type')
          );
          expect(consumerOpTypes.length).toBe(1);
          expect(consumerOpTypes[0].nativeElement.innerText).toEqual(
            'ConsumerOp'
          );
        }
      );
    }
  }
});
