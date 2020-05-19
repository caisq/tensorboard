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

import {Component, EventEmitter, Input, Output} from '@angular/core';

import {GraphOpInfo} from '../../store/debugger_types';

@Component({
  selector: 'graph-op',
  templateUrl: 'graph_op_component.ng.html',
  styleUrls: ['./graph_op_component.css'],
})
export class GraphOpComponent {
  @Input()
  kind!: 'self' | 'input' | 'consumer';

  // ID of the immediately-enclosing graph.
  @Input()
  graphId!: string;

  @Input()
  opName!: string;

  // Must be `undefined` if kind is 'self'.
  // Must be a valid number if kind is 'input' or 'consumer'.
  @Input()
  slot!: number | undefined;

  @Input()
  opData!: GraphOpInfo | undefined;

  @Output()
  onOpNameClick = new EventEmitter<{graph_id: string; op_name: string}>();
}
