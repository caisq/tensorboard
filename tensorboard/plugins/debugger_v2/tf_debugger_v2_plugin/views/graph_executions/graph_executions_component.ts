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

import {ChangeDetectionStrategy, Component, Input} from '@angular/core';

import {GraphExecutionDigest} from '../../store/debugger_types';

@Component({
  selector: 'graph-executions-component',
  templateUrl: './graph_executions_component.ng.html',
  styleUrls: ['./graph_executions_component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GraphExecutionsComponent {
  @Input()
  numGraphExecutions: number | null = null;

  @Input()
  graphExecutionDigests: {[index: number]: GraphExecutionDigest} = {};

  // @Input()
  // opNames: string[] | null = ["Mul", "Mul", "Mul", "Mul", "Mul", "Mul"];

  @Input()
  graphExecutionIndices: number[] | null = null;

  // opNames1: string[] = Array.from({length: this.numGraphExecutions!}).map((_, i) => {
  //   const graphExecutionDigest = this.graphExecutionDigests[i];
  //   return graphExecutionDigest === undefined
  //     ? null
  //     : graphExecutionDigest.op_name;
  // });

  // items = Array.from({length: 1000 * 1000}).map((_, i) => `Item #${i}`);

  // items = Array.from({length: this.numGraphExecutions!}).map((_, i) => {
  //   const graphExecutionDigest = this.graphExecutionDigests[i];
  //   return graphExecutionDigest === undefined
  //     ? null
  //     : graphExecutionDigest.op_name;
  // });

  scrolledIndexChange(scrollIndex: number) {
    console.log('scrolledIndexChange:', scrollIndex); // DEBUG
  }

  // constructor() {
    // console.log('GraphExecutionsComponent: numGraphExecutions =', this.numGraphExecutions); // DEBUG
    // console.log('GraphExecutionsComponent: items =', this.items); // DEBUG
  // }
}
