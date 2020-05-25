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

import {CodeLocationType} from '../../store/debugger_types';

export interface StackFrameForDisplay {
  host_name: string;
  file_path: string;
  concise_file_path: string;
  lineno: number;
  function_name: string;
  // Whether the stack frame is a part of the focused file.
  // Being a part of the focused file is a necessary but insufficient
  // condition for being the focused stack frame (see `focused` below).
  belongsToFocusedFile: boolean;
  // Whether the stack frame is the one being focused on (e.g.,
  // being viewed in the source code viewer). If this field is `true`,
  // `belongsToFocusedFile` must also be `true`.
  focused: boolean;
  // Is this stack frame the topmost in the focused file.
  // N.B.: Python stack frames are printed from bottommost to topmost by
  // default, a "topmost" stack frame is actually the one that appears
  // the last in a typical stack trace printed from Python.
  topmostInFocusedFile: boolean;
}

@Component({
  selector: 'stack-trace-component',
  templateUrl: './stack_trace_component.ng.html',
  styleUrls: ['./stack_trace_component.css'],
})
export class StackTraceComponent {
  @Input()
  stackTraceType!: CodeLocationType | null;

  @Input()
  originOpInfo!: {
    // The name of the op that the stack trace is about.
    // E.g., 'Dense_2/MatMul'.
    // For eager execution, this is null.
    opName: string | null;
    // The type of the op that the stack trace is about.
    // E.g., 'MatMul'.
    opType: string;
  } | null;

  @Input()
  stackFramesForDisplay: StackFrameForDisplay[] | null = null;

  @Output()
  onSourceLineClicked = new EventEmitter<{
    host_name: string;
    file_path: string;
    lineno: number;
  }>();

  toggleTopmostFrameInFile() {
    console.log('toggleTopmostFrameInFile():'); // DEBUG
  }

  CodeLocationType = CodeLocationType;
}
