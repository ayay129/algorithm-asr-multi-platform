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
import sys
import pyannote.audio


def main():
    pyannote_audio_path = pyannote.audio.__path__
    pyannote_audio_version = pyannote.audio.__version__

    if pyannote_audio_version != '3.1.1':
        sys.exit("Expectation pyannote.audio==3.1.1")
    os.system(f'patch -p0 {pyannote_audio_path[0]}/models/segmentation/PyanNet.py PyanNet.patch')
    os.system(f'patch -p0 {pyannote_audio_path[0]}/models/blocks/sincnet.py sincnet.patch')
    os.system(f'patch -p0 {pyannote_audio_path[0]}/pipelines/voice_activity_detection.py voice_activity_detection.patch')
    os.system(f'patch -p0 {pyannote_audio_path[0]}/core/inference.py inference.patch')

if __name__ == '__main__':
    main()