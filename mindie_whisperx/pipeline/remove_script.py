# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import stat

import asteroid_filterbanks


def add_comment_before_function(file_path, target_function):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if i > 0 and target_function in line:
            lines[i - 1] = '#' + lines[i - 1]

    flags = os.O_WRONLY | os.O_TRUNC
    mode = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(file_path, flags, mode), 'w') as file:
        for line in lines:
            file.write(line)


if __name__ == "__main__":
    asteroid_filterbanks_path = asteroid_filterbanks.__path__
    enc_dec_path = f'{asteroid_filterbanks_path[0]}/enc_dec.py'
    function_str = 'def multishape_conv1d('
    add_comment_before_function(enc_dec_path, function_str)