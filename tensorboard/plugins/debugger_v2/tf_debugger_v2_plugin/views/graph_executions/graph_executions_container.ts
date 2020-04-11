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
import {Component} from '@angular/core';
import {createSelector, select, Store} from '@ngrx/store';

import {getGraphExecutionDigests, getNumGraphExecutions} from '../../store';
import {GraphExecutionDigest, State} from '../../store/debugger_types';

/** @typehack */ import * as _typeHackRxjs from 'rxjs';

@Component({
  selector: 'tf-debugger-v2-graph-executions',
  template: `
    <graph-executions-component
      [numGraphExecutions]="numGraphExecutions$ | async"
      [graphExecutionDigests]="graphExecutionDigests$ | async"
      [graphExecutionIndices]="graphExecutionIndices$ | async"
    ></graph-executions-component>
  `,
})
export class GraphExecutionsContainer {
  readonly numGraphExecutions$ = this.store.pipe(select(getNumGraphExecutions));

  readonly graphExecutionDigests$ = this.store.pipe(
    select(getGraphExecutionDigests)
  );

  // readonly opNames$ = this.store.pipe(
  //   select(
  //     createSelector(
  //       getGraphExecutionDigests,
  //       getNumGraphExecutions,
  //       (graphExecutionDigests: {[index: number]: GraphExecutionDigest}, numGraphExecution: number): string[] => {
  //         console.log('numGraphExecution=', numGraphExecution);  // DEBUG
  //         const output: string[] = new Array<string>(numGraphExecution);
  //         for (let i = 0; i < numGraphExecution; ++i) {
  //           output[i] = graphExecutionDigests[i] === undefined ? '' : graphExecutionDigests[i].op_name;
  //         }
  //         console.log('output:', output);  // DEBUG
  //         return output;
  //       }
  //     )
  //   )
  // )

  readonly graphExecutionIndices$ = this.store.pipe(
    select(
      createSelector(
        getNumGraphExecutions,
        (numGraphExecution: number): number[] | null => {
          if (numGraphExecution === 0) {
            return null;
          }
          return Array.from({length: numGraphExecution}).map((_, i) => i);
        }
      )
    )
  )

  constructor(private readonly store: Store<State>) {}
}
