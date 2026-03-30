# 多平台模型编译与测试工作台 MVP 说明

## 1. 目标

做一个统一的模型工具，用来管理和执行以下任务：

- 多平台模型编译
- 准确率测试
- 性能测试
- 服务部署
- 结果归档与导出

第一阶段先覆盖当前已有能力：

- ASR
  - `mindie_whisperx`，昇腾 910B4
  - `whisperx`，NVIDIA CUDA
- CV
  - 先按通用模型任务抽象，不绑定具体前端展示样式
  - 后续逐步接入 OCR、Detection、Feature、Face 等

这份文档的目标不是定义视觉稿，而是定义：

- 前端页面结构
- 交互流程
- 核心对象
- 后端接口契约
- 任务状态和数据格式

前端可以先按这份协议做壳子，后面我直接接后端。

## 2. 设计原则

- 不为某一种模型单独做系统，统一抽象成“模型 + 平台 + 任务 + 数据集 + 产物”。
- 前端只负责配置、触发、展示，不负责推理逻辑。
- 编译、测试、部署全部抽象成任务。
- 任务参数必须结构化，不允许只靠自由文本。
- 前端不写死平台逻辑，平台差异由后端模板和插件返回。

## 3. MVP 范围

MVP 必做：

- 模型管理
- 数据集管理
- 新建任务
- 任务列表
- 任务详情
- 日志查看
- 指标查看
- 产物下载

MVP 可延后：

- 在线可视化波形/图像对比
- 多用户权限
- WebSocket 实时推送
- 复杂报表设计器
- 集群资源调度面板
- 精细化告警系统

## 4. 核心对象

### 4.1 Model

表示一个模型条目，不等于某个平台编译产物。

建议字段：

- `id`
- `name`
- `task_type`
  - `asr`
  - `ocr`
  - `detection`
  - `classification`
  - `embedding`
  - `face`
- `framework`
  - `whisperx`
  - `pytorch`
  - `onnx`
  - `tensorrt`
  - `mindie`
  - `openvino`
- `source_format`
  - `hf`
  - `pt`
  - `onnx`
  - `engine`
  - `om`
- `source_path`
- `repo_path`
- `description`
- `tags`
- `created_at`
- `updated_at`

### 4.2 BackendProfile

表示一个运行/编译目标环境。

建议字段：

- `id`
- `name`
- `vendor`
  - `nvidia`
  - `ascend`
  - `intel`
- `runtime`
  - `cuda`
  - `tensorrt`
  - `mindie`
  - `acl`
  - `openvino`
  - `onnxruntime`
- `device_model`
  - 例如 `A10`、`4090`、`910B4`
- `host`
- `container_image`
- `enabled`
- `capabilities`
  - `compile`
  - `accuracy_eval`
  - `benchmark`
  - `serve`

### 4.3 Dataset

表示一个测试数据集定义。

建议字段：

- `id`
- `name`
- `task_type`
- `source_type`
  - `local`
  - `modelscope`
  - `huggingface`
- `path_or_name`
- `namespace`
- `split`
- `language_map`
- `audio_column`
- `text_columns`
- `image_column`
- `label_column`
- `meta`

说明：

- ASR 数据集通常需要 `language_map`、`audio_column`、`text_columns`
- CV 数据集通常需要图片列、标签列、标注格式说明

### 4.4 Job

系统中的一切执行动作都抽象为 Job。

建议字段：

- `id`
- `job_type`
  - `compile`
  - `accuracy_eval`
  - `benchmark`
  - `serve`
  - `export_report`
- `model_id`
- `backend_profile_id`
- `dataset_id`
- `params`
- `status`
- `progress`
- `created_at`
- `started_at`
- `finished_at`
- `created_by`
- `error_message`

### 4.5 Artifact

任务执行结果文件。

建议字段：

- `id`
- `job_id`
- `type`
  - `compiled_model`
  - `metrics_json`
  - `metrics_csv`
  - `detail_csv`
  - `report_md`
  - `log`
  - `docker_image_ref`
- `name`
- `path`
- `size`
- `mime_type`
- `created_at`

### 4.6 Metric

结构化指标展示对象。

建议字段：

- `job_id`
- `group`
  - `overall`
  - `language`
  - `class`
  - `scenario`
- `name`
  - 例如 `wer`、`cer`、`rtf`、`latency_ms`、`qps`、`map`
- `value`
- `unit`
- `dimension`
  - 例如 `zh`、`en`、`fr`、`ar`
- `display_order`

## 5. 任务类型定义

### 5.1 编译任务

用途：

- 模型从原始格式编译到目标运行格式

典型例子：

- WhisperX 到 TensorRT 引擎
- Whisper/MindIE 模型编图
- ONNX 到 OM

### 5.2 准确率任务

用途：

- 跑标准数据集并输出结构化指标

ASR 常见指标：

- `WER`
- `CER`
- `句级准确率`
- `RTF`
- `长音频稳定性`

CV 常见指标：

- `Top1`
- `Recall`
- `mAP`
- `Precision`

### 5.3 性能任务

用途：

- 跑吞吐、时延、显存、内存、CPU/NPU/GPU 利用率

常见指标：

- `avg_latency_ms`
- `p95_latency_ms`
- `throughput_qps`
- `rtf`
- `gpu_util`
- `npu_util`
- `gpu_mem_mb`
- `cpu_percent`

### 5.4 部署任务

用途：

- 启服务
- 停服务
- 记录部署配置和镜像

## 6. 前端页面结构

### 6.1 模型列表页

用途：

- 查看所有模型
- 搜索和筛选
- 新建模型
- 进入任务页

字段建议：

- 名称
- 任务类型
- 框架
- 来源格式
- 最近更新时间
- 标签

操作：

- 新建
- 编辑
- 创建任务
- 查看历史任务

### 6.2 数据集列表页

用途：

- 管理准确率/性能测试数据集定义

字段建议：

- 名称
- 任务类型
- 数据源类型
- split
- 最近更新时间

操作：

- 新建
- 编辑
- 预览配置

### 6.3 新建任务页

用途：

- 从模型、平台、数据集出发创建任务

推荐交互：

1. 选择模型
2. 选择任务类型
3. 选择平台
4. 按后端模板渲染动态参数表单
5. 提交任务

表单必须分区：

- 基础信息
- 平台参数
- 数据集参数
- 高级参数

### 6.4 任务列表页

用途：

- 查看任务执行状态

筛选项：

- 任务类型
- 状态
- 模型
- 平台
- 时间范围

状态：

- `pending`
- `queued`
- `running`
- `success`
- `failed`
- `canceled`

### 6.5 任务详情页

用途：

- 这是 MVP 的核心页面

页面分区建议：

- 基本信息
- 参数快照
- 状态时间线
- 日志
- 指标
- 产物

日志要求：

- 支持分页或增量加载
- 支持复制
- 支持搜索关键字

指标要求：

- 支持表格视图
- 支持按语言/类别切换
- 支持直接复制为 Markdown/CSV

产物要求：

- 支持下载
- 支持显示文件大小和创建时间

### 6.6 平台配置页

MVP 可以先做简单版。

用途：

- 管理可用后端节点
- 查看平台能力

字段建议：

- 名称
- 厂商
- runtime
- 设备型号
- 是否启用
- 支持的任务类型

## 7. 页面交互约束

- 所有任务提交前必须显示“参数快照预览”
- 所有运行中的任务必须有状态刷新
- 所有失败任务必须显示错误信息和日志入口
- 所有成功任务必须显示结构化指标和产物入口
- 不允许任务成功但页面只显示一段纯文本

## 8. 后端接口契约

接口统一前缀建议：

- `/api/v1`

### 8.1 模型

#### `GET /api/v1/models`

返回模型列表。

#### `POST /api/v1/models`

创建模型。

请求示例：

```json
{
  "name": "whisperx-large-v3",
  "task_type": "asr",
  "framework": "whisperx",
  "source_format": "hf",
  "source_path": "openai/whisper-large-v3",
  "description": "NVIDIA WhisperX model",
  "tags": ["asr", "cuda", "whisperx"]
}
```

### 8.2 数据集

#### `GET /api/v1/datasets`

#### `POST /api/v1/datasets`

ASR 数据集示例：

```json
{
  "name": "google-fleurs-test",
  "task_type": "asr",
  "source_type": "modelscope",
  "path_or_name": "google/fleurs",
  "split": "test",
  "language_map": {
    "zh": "cmn_hans_cn",
    "en": "en_us",
    "fr": "fr_fr",
    "ar": "ar_eg"
  },
  "audio_column": "audio",
  "text_columns": ["transcription", "text", "sentence"]
}
```

### 8.3 平台模板

#### `GET /api/v1/backend-profiles`

返回平台列表。

#### `GET /api/v1/job-templates`

根据 `task_type + framework + runtime` 返回动态表单模板。

请求参数示例：

- `task_type=accuracy_eval`
- `framework=whisperx`
- `runtime=cuda`

返回示例：

```json
{
  "sections": [
    {
      "title": "基础参数",
      "fields": [
        {
          "name": "batch_size",
          "type": "number",
          "required": true,
          "default": 16
        },
        {
          "name": "language",
          "type": "string",
          "required": false
        }
      ]
    }
  ]
}
```

### 8.4 任务

#### `POST /api/v1/jobs`

创建任务。

请求示例：

```json
{
  "job_type": "accuracy_eval",
  "model_id": "model_001",
  "backend_profile_id": "backend_cuda_01",
  "dataset_id": "dataset_fleurs_test",
  "params": {
    "batch_size": 16,
    "device_index": 0,
    "compute_type": "float16",
    "align": true,
    "diarize": false
  }
}
```

#### `GET /api/v1/jobs`

支持筛选：

- `status`
- `job_type`
- `model_id`
- `backend_profile_id`

#### `GET /api/v1/jobs/{job_id}`

返回任务详情。

#### `POST /api/v1/jobs/{job_id}/cancel`

取消任务。

#### `GET /api/v1/jobs/{job_id}/logs`

返回日志片段。

建议支持：

- `offset`
- `limit`

#### `GET /api/v1/jobs/{job_id}/metrics`

返回结构化指标。

#### `GET /api/v1/jobs/{job_id}/artifacts`

返回产物列表。

### 8.5 下载

#### `GET /api/v1/artifacts/{artifact_id}/download`

下载文件。

## 9. 返回结构建议

统一响应格式建议：

```json
{
  "code": 0,
  "message": "ok",
  "data": {}
}
```

错误响应：

```json
{
  "code": 40001,
  "message": "invalid params",
  "data": null
}
```

## 10. 任务状态流转

统一状态：

- `pending`
- `queued`
- `running`
- `success`
- `failed`
- `canceled`

状态流转规则：

- 新建后进入 `pending`
- 被调度后进入 `queued`
- Worker 开始执行进入 `running`
- 完成进入 `success`
- 出错进入 `failed`
- 用户取消进入 `canceled`

## 11. 插件式后端抽象

后端不要写死在前端里，建议按插件实现。

统一插件接口建议：

- `prepare(job)`
- `validate(job)`
- `run(job)`
- `collect_metrics(job)`
- `collect_artifacts(job)`

第一批插件建议：

- `adapter_whisperx_cuda`
- `adapter_mindie_whisperx_ascend`
- `adapter_cv_generic_onnx`

## 12. ASR 专项要求

前端必须支持下面这些参数展示：

- 模型名
- 语言
- batch size
- 设备索引
- 对齐开关
- diarization 开关
- 计算精度
- 数据集映射

ASR 指标展示至少支持：

- `WER`
- `CER`
- `句级准确率`
- `RTF`
- `长音频稳定性`

指标展示必须支持：

- overall
- 按语言拆分

## 13. CV 通用要求

前端不要为某个单一 CV 任务写死页面。

CV 指标区必须支持以下结构：

- `overall`
- `per_class`
- `per_scenario`

常见指标举例：

- `top1`
- `precision`
- `recall`
- `map`
- `latency_ms`
- `qps`

## 14. 报告导出要求

至少支持导出：

- `metrics.json`
- `metrics.csv`
- `detail.csv`
- `report.md`

后续可再补：

- `docx`
- `xlsx`

## 15. 推荐前端实现边界

前端现在只需要先做：

- 路由和页面壳子
- 表格和筛选
- 动态表单
- 任务详情页
- 日志/指标/产物展示

前端现在不要做：

- 编译逻辑
- 推理逻辑
- 本地文件直接处理
- 平台差异判断

这些都放后端。

## 16. 推荐前端目录结构

仅作为参考，不是强制。

```text
src/
  pages/
    models/
    datasets/
    jobs/
    backends/
  components/
    job-form/
    metric-table/
    artifact-list/
    log-viewer/
  services/
    api/
  types/
  hooks/
```

## 17. 实施顺序建议

第一步：

- 先把模型列表、数据集列表、任务列表、任务详情页做出来

第二步：

- 接动态表单和任务提交

第三步：

- 接日志、指标、产物接口

第四步：

- 接平台管理页

## 18. 当前仓库落地建议

这个仓库里当前已经有两条后端线：

- [mindie_whisperx](/Users/rangers/DevelopServices/app/algo-asr-multi-plat/mindie_whisperx)
- [whisperx](/Users/rangers/DevelopServices/app/algo-asr-multi-plat/whisperx)

后端第一阶段建议只接这两条：

- `whisperx` 作为 NVIDIA ASR backend
- `mindie_whisperx` 作为 Ascend ASR backend

这样前端可以先验证通用工作流，后面再加 CV 模型插件。

## 19. 交付说明

这份文档是前后端契约草案。

你可以先让别人按它做前端，不必等后端实现完。

后续如果前端完成了页面壳子，我接后端时只需要对齐：

- 路由
- 请求结构
- 状态字段
- 指标格式
- 产物格式

就可以直接落。
