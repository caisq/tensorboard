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
  // Whether this frame should be automatically focused on even though the
  // user may have focused on a different frame in the same file.
  // This is applicable only if the `stickToBottommostFrameInFocusedFile` is
  // `true` in the ngrx store.
  autoFocus: boolean;
}

@Component({
  selector: 'stack-trace-component',
  templateUrl: './stack_trace_component.ng.html',
  styleUrls: ['./stack_trace_component.css'],
})
export class StackTraceComponent {
  @Input()
  codeLocationType!: CodeLocationType | null;

  @Input()
  opType!: string | null;

  // Index of eager (top-level) execution, not `null` only for
  // `CodeLocationType.GRAPH_OP_CREATION`.
  @Input()
  opName!: string | null;

  // Index of eager (top-level) execution, not `null` only for
  // `CodeLocationType.EXECUTION`.
  @Input()
  executionIndex!: number | null;

  @Input()
  stickToBottommostFrameInFocusedFile!: boolean;

  @Input()
  stackFramesForDisplay: StackFrameForDisplay[] | null = null;

  @Output()
  onSourceLineClicked = new EventEmitter<{
    host_name: string;
    file_path: string;
    lineno: number;
  }>();

  @Output()
  onToggleBottommostFrameInFile = new EventEmitter<boolean>();

  CodeLocationType = CodeLocationType;
}
