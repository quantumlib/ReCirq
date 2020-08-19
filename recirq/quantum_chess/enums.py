# Copyright 2020 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import enum


class MoveType(enum.Enum):
    NULL_TYPE = 0
    UNSPECIFIED_STANDARD = 1
    JUMP = 2
    SLIDE = 3
    SPLIT_JUMP = 4
    SPLIT_SLIDE = 5
    MERGE_JUMP = 6
    MERGE_SLIDE = 7
    PAWN_STEP = 8
    PAWN_TWO_STEP = 9
    PAWN_CAPTURE = 10
    PAWN_EP = 11
    KS_CASTLE = 12
    QS_CASTLE = 13


class MoveVariant(enum.Enum):
    UNSPECIFIED = 0
    BASIC = 1
    EXCLUDED = 2
    CAPTURE = 3


class ErrorMitigation(enum.Enum):
    Nothing = 0  # No error mitigation
    Error = 1  # Throw ValueError (for debugging)
    Correct = 2  # Correct for errors through post-selection
