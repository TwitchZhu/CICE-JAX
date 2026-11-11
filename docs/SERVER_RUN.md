# CICE-JAX 服务器直接运行：以 gx3_5day 为例

执行顺序：准备源码和输入数据 → 创建 Python 环境 → 检查 GPU → 创建 case → 运行 → 查看结果 → 可选 Fortran 对比。

本流程在可以直接运行程序的 NVIDIA GPU 节点上执行，不包含作业提交脚本。JAX 使用一个 Python 进程运行完整网格，不需要编译 Fortran，也不使用 `mpirun`。

## 1. 准备目录和数据

将包含本次修改的 CICE-JAX 源码上传或克隆到服务器，采用下面的目录布局：

```text
工作目录/
  CICE-JAX/
    cice.setup
    cice_setup.py
    environment.yml
    configuration/
    cicecore/
    icepack/
    docs/
  CICE/
    input/
      CICE_data/
        ic/gx3/iced_gx3_v5.nc
        grid/gx3/
        forcing/gx3/
  cice-dirs/
    cases/                         后续统一在这里执行命令
      gx3_5day_jax/                 创建命令将生成此目录
```

输入使用与此前 GNU/Intel 五天试验相同的数据，包含完整的 gx3 网格、初始场和 2005 年 JRA55 forcing。创建命令检查初始场、grid/kmt 文件和 forcing 目录是否存在，不下载数据，也不逐个校验全部 forcing 文件。

`CICE/` 在这个流程中只提供输入数据，不要求安装其编译环境。数据也可以存放在其他位置，创建 case 时修改 `--inputdata` 即可。该参数指向 **包含 `CICE_data/` 的目录**，例如 `CICE/input/`，不是 `CICE_data/` 本身。

从上述“工作目录”进入一次：

```bash
mkdir -p cice-dirs/cases
cd cice-dirs/cases
```

**下面所有命令都在 `cice-dirs/cases/` 执行，无需进入源码目录或 case 目录。** 已有这个目录时直接进入即可。

## 2. 创建并激活 GPU 环境

首次安装时执行，要求服务器已有可用的 Conda 和 NVIDIA 驱动，且安装时能访问 Conda/PyPI 或相应镜像：

```bash
conda env create -f ../../CICE-JAX/environment.yml
conda activate cice_jax
```

环境使用 Python 3.10、NumPy、netCDF4 和 `jax[cuda12]==0.6.2`。CUDA 运行时由 pip 依赖安装，宿主机 NVIDIA 驱动仍需支持它。环境已创建时只执行 `conda activate cice_jax`；已有兼容环境也可直接激活使用。

## 3. 确认当前节点可以使用 GPU

```bash
nvidia-smi
JAX_PLATFORMS=cuda python -c "import jax; jax.config.update('jax_enable_x64', True); print('JAX:', jax.__version__); print('Devices:', jax.devices()); print('x64:', jax.config.x64_enabled)"
```

应看到 CUDA GPU，且 `x64: True`。先解决这里的驱动或环境错误，再运行模型。生成的 `run.py` 同样要求 CUDA，GPU 初始化失败时会报错，不会悄悄改用 CPU。

本例使用默认的第一张可见 GPU。如果节点有多张卡，可以在确认可用卡号后设置 `CUDA_VISIBLE_DEVICES`；已有 GPU 可见性限制时沿用现有设置。本版本没有实现跨多张 GPU 的网格分解。

## 4. 创建 gx3 五天 case

```bash
python ../../CICE-JAX/cice.setup \
  --case gx3_5day_jax \
  --preset gx3_5day \
  --inputdata ../../CICE/input
```

等价入口是 `python ../../CICE-JAX/cice_setup.py`，参数相同。`--case` 和 `--inputdata` 都相对于执行命令的目录解析，也接受用户提供的绝对路径；生成的文件内部保存相对路径。

当前内置预设为 `gx3_5day`。配置从随仓库分发的 CICE 6.6.3 namelist 快照按“基础 → gx3 → diag1 → run5day”生成，对应 `quick_suite` 的 `smoke gx3 8x2 diag1,run5day` 项，再改为 JAX 的单进程完整网格布局。不读取个人 `~/.cice_set`。它不是执行完整的官方 test suite，也不创建 Fortran 案例。

主要设置：100×116 网格，5 个厚度类别、7 个冰层、1 个雪层；`dt=3600` 秒，2005-01-01 至 2005-01-06，共 120 步；保留 mushy、dEdd、EVP、remap 和 ridging 配置。`8x2` 是 Fortran 对照的并行布局，不是 JAX 的启动进程数。

生成内容：

```text
gx3_5day_jax/
  ice_in                 模型配置，数据路径相对于这个 case
  case.json              源码/数据相对路径、预设来源和初始配置校验值
  run.py                 运行入口，复制自仓库的运行模板
  compare_all.py         本五天试验的 GNU/Intel/JAX 比较脚本
  history/               history 输出
  restart/               restart 输出
```

case 可以建在源码树之外。`case.json` 中的 `source` 指向此次创建命令所在的 CICE-JAX 源码；运行时直接使用这份源码，不另外复制模型实现。若整体迁移目录，请保留相对布局；单独移动 case 后需调整 `case.json` 的 `source` 和 `ice_in` 的输入路径。`case.json` 的 `inputdata` 是创建记录，模型实际读取的是 `ice_in`。

已有同名 case 时，**创建命令**停止，避免覆盖配置；需要新试验时换一个 `--case` 名称。已有 case 重新运行则直接执行下一步。当前不支持旧移植稿中的 `-m/-e/-p/-g/-s` 或 `--suite/--test` 参数；其他网格及一年试验需另行加入并验证预设。

## 5. 直接运行

```bash
python gx3_5day_jax/run.py
```

程序会自动进入 case 目录读写文件。第一次运行需要 JIT 编译，首步明显慢于稳定时间步。终端首先显示日志路径；详细进度写入 `gx3_5day_jax/cice.runlog.<时间戳>`，完成时终端显示 `Completed gx3_5day_jax`。

性能评估以日志中的 `TIMING_LOOP completed=True reason=ok` 为准：默认先推进 7 个真实时间步预热，然后对剩余完整循环整体计时。`wall_seconds` 是这段循环的总耗时，`measured_steps` 是计入的步数，`seconds_per_step` 是两者相除。120 步的试验默认测量后 113 步。预热属于原有模拟步数，不重复执行或重置模型状态；从 restart 启动时也按本次实际执行的步数计数。

计时开始前和结束时等待模型状态、history 数组及 JAX effects 完成。起点位于第一个被测时间步的 forcing 准备之前，包含 forcing 读取和上传、物理计算、诊断、history/restart、日志以及现有阶段计时的开销；不含初始化、预热和循环结束后的 finalize。它是完整循环的墙钟时间，不是纯 GPU kernel 时间，也不依赖 `jax.monitoring` 的编译监控接口，适用于当前 CentOS 7 的 JAX 0.5.3。

模型同步使用逐数组等待：本地 CentOS 7 / JAX 0.5.3 / WSL CUDA 组合在对完整状态同时调用 `jax.block_until_ready` 时曾卡在 CUDA stream 创建；逐数组等待可通过该位置。同步位置及等待完成的数组范围保持一致，不将未完成的 GPU 工作排除在计时之外。

可调整预热步数（正整数）：

```bash
python gx3_5day_jax/run.py --warmup-steps 10
```

如果总步数不超过预热步数，输出 `reason=no_steps_after_warmup`，耗时为 `unavailable`；循环异常退出时输出 `completed=False reason=aborted`，不把部分运行作为完整性能结果。`TIMING_LOOP` 的完成标志仅表示时间步循环完成，整次运行仍需检查后面的成功标志。

固定步数预热不能保证之后没有新分支触发 JIT；后续编译会如实计入循环耗时。必要时另做一次 `JAX_LOG_COMPILES=1` 的诊断运行观察编译日志，并调整预热长度。比较 SIF 与原生 conda 时，使用相同节点/GPU、软件版本、输入、预热长度及输出设置，关闭 profiler，重复运行比较结果。仅更换 JAX/cuDNN 版本也可能改变性能，不能将其差异全部归因于容器。

新建 case 的 `run.py` 支持上述参数。已生成的 case 不会自动更新入口脚本；仍可使用默认 7 步计时（需指向更新后的源码），若要通过参数调整，可将 `configuration/scripts/run_case.py` 中的入口改动同步到已有 `run.py`，保留自己的配置。

原有 `TIMING_TOTAL`、`TIMING_OUTPUT` 保留作为辅助诊断：`compile_seconds`、扣除编译的 `run_seconds` 等分项依赖 JAX 监控接口，接口不可用时仍明确输出 `unavailable`。不要把这些分项与 `TIMING_LOOP` 相加或混用；后者直接测量预热后的完整循环。

默认关闭 GPU profiler，仍记录阶段计时；第 1–7 步记录细分物理计时。需要与本地优化过程一致的第 5–10 步 GPU 采样时，改为运行：

```bash
python gx3_5day_jax/run.py --profile
```

这两个命令是两种运行方式，通常只选其中一个。启用采样会增加开销，并在 `profiles/cice.runlog.<时间戳>/` 保存记录。

入口保留当前经过五天试验的优化设置：x64、Picard 分块收敛、therm2/ridge 批量 JIT 等。短波计算继续沿用源码中局部关闭 CUDA command buffer 的兼容设置。只有遇到 command buffer 兼容问题时，再尝试全局关闭选项：

```bash
python gx3_5day_jax/run.py --disable-cuda-command-buffer
```

本机 CentOS 7 / glibc 2.17、JAX 0.5.3、WSL + RTX 3060 Ti 的五天实测需要此选项，并使用上述逐数组同步修复。保留默认 command buffer 时曾在第 3 步发生 `libcuda.so` 段错误；关闭后完整通过 120 步。这个结论针对该本机组合，近思需用其实际驱动验证。

**重新运行会覆盖同名 history/restart 输出，日志以时间戳区分。** 如需同时保留两组输出，创建另一个 case；无需为继续运行逐项删除旧日志或目录。

## 6. 查看进度和最终输出

运行过程中可在另一个终端激活环境并进入同一个 `cice-dirs/cases/`，查看最新日志：

```bash
tail -f "$(ls -t gx3_5day_jax/cice.runlog.* | head -n 1)"
```

也可以单独查看 GPU：

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1
```

运行结束后检查：

```bash
tail -n 20 "$(ls -t gx3_5day_jax/cice.runlog.* | head -n 1)"
ls -lh gx3_5day_jax/restart/iced.2005-01-06-00000.nc
ls -lh gx3_5day_jax/history/
```

日志应有 `CICE COMPLETED SUCCESSFULLY` 和 `Final calendar: istep1=121, npt=120`。结束日志中的 121 是循环结束后的计数；最终 restart 内的 `istep1` 应为 120，日期为 2005-01-06 00:00。

五天配置沿用月平均 history，并写初始场；不能期待五个每日 history 文件。最终 `restart/iceh_rm.*` 保存的是 history 累积恢复信息，不是五天的每日历史场。

## 7. 可选：与已有 GNU/Intel 结果比较

先将同一个五天试验的 GNU/Intel 结果放到如下目录。这里只收集结果，不需要重新编译 Fortran：

```text
gx3_5day_jax/
  gnu/
    logs/cice.runlog.<时间戳>
    restart/iced.2005-01-06-00000.nc
    history/*.nc
  intel/
    logs/cice.runlog.<时间戳>
    restart/iced.2005-01-06-00000.nc
    history/*.nc
```

三端均完成后执行：

```bash
python gx3_5day_jax/compare_all.py
```

报告写入本 case 的 `comparison_logs/<时间戳>/`，包含 GNU–Intel、GNU–JAX、Intel–JAX。脚本检查最新日志的完成标志、restart 的 120 步和最终日期，然后报告变量误差、掩膜差异及非有限值；不设置所谓官方浮点误差阈值。`compare_all.py` 当前针对这个五天预设，修改模拟时间后不能原样用于其他试验。

已有 Fortran 结果尚未收集时，可先完成 JAX 运行，之后再比较；创建和运行 JAX case 不依赖 `gnu/`、`intel/` 目录。
