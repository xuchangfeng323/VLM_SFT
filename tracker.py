"""SwanLab 实验记录封装。

设计原则：

1. 默认只在主进程（rank0）上报，多卡时不会重复写一份实验；
2. swanlab 没装 / 没登录 / 网络不通 / 配置里没开，全部降级成空操作，
   只打印一行提示，绝不因为记录实验而中断训练；
3. 只上报数值，字符串（比如 val_summary）会被过滤掉——swanlab 遇到
   字符串会报错并丢掉这条记录。

常用配置（写在 args/arg1.json 里）：

    "use_swanlab": true,
    "swanlab_project": "VLM_SFT-GMNER",
    "swanlab_experiment_name": "qwen2.5-vl-3b-lora-r16",   // 不填则自动命名
    "swanlab_mode": null,        // null=云端(online) / "offline" / "local" / "disabled"
    "swanlab_logdir": null,      // offline/local 模式的本地目录，默认 ./swanlab/
    "swanlab_log_interval": 10   // 每多少个优化步记一次 train/*

云端模式需要先 `swanlab login`，或者设置环境变量 SWANLAB_API_KEY。
"""

from typing import Any, Dict, Mapping, Optional

_NUMERIC = (int, float)


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """把嵌套指标拍平成 swanlab 的 key，并丢掉非数值项。

    {"f1": 0.7, "per_type": {"loc": {"f1": 0.5}}, "summary": "..."}
      -> {"prefix/f1": 0.7, "prefix/per_type/loc/f1": 0.5}
    """
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_metrics(value, f"{name}/"))
        elif isinstance(value, bool) or not isinstance(value, _NUMERIC):
            continue  # 字符串/None 之类直接跳过
        else:
            flat[name] = value
    return flat


class SwanLabTracker:
    """swanlab 的空操作安全封装：没开启时所有方法都是 no-op。"""

    def __init__(self, run: Any = None, log_interval: int = 10):
        self.run = run
        self.enabled = run is not None
        self.log_interval = max(1, int(log_interval))
        self._broken = False

    def log(self, data: Optional[Mapping[str, Any]], step: Optional[int] = None) -> None:
        """上报一批数值指标；任何异常都只在第一次提示，然后停用上报。"""
        if not self.enabled or not data or self._broken:
            return
        # swanlab 遇到 None / 裸字符串会报错并丢掉整条记录，这里先过滤掉
        cleaned = {
            key: value
            for key, value in data.items()
            if value is not None and not isinstance(value, str)
        }
        if not cleaned:
            return
        try:
            self.run.log(cleaned, step=step)
        except Exception as exc:  # noqa: BLE001 - 记录失败不该影响训练
            self._broken = True
            print(f"[swanlab] 上报失败，本次训练后续不再记录：{exc}")

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        prefix: str = "",
        step: Optional[int] = None,
    ) -> None:
        """上报一份指标字典（自动拍平嵌套 + 过滤非数值）。"""
        self.log(flatten_metrics(metrics, prefix), step=step)

    def finish(self) -> None:
        if not self.enabled:
            return
        self.enabled = False
        try:
            import swanlab

            swanlab.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[swanlab] 结束实验时出错（可以忽略）：{exc}")


def build_tracker(args: Any, is_main_process: bool = True) -> SwanLabTracker:
    """按配置初始化 swanlab，返回 tracker（失败时返回空操作对象）。"""
    if not getattr(args, "use_swanlab", False):
        return SwanLabTracker()
    if not is_main_process:
        return SwanLabTracker()

    try:
        import swanlab
    except ImportError:
        print("[swanlab] 没装 swanlab，跳过实验记录（pip install swanlab 后可用）")
        return SwanLabTracker()

    name = getattr(args, "swanlab_experiment_name", None) or None
    project = getattr(args, "swanlab_project", None) or "VLM_SFT-GMNER"
    try:
        run = swanlab.init(
            project=project,
            workspace=getattr(args, "swanlab_workspace", None) or None,
            name=name,
            description=getattr(args, "swanlab_description", None) or None,
            config=dict(getattr(args, "args_dict", {}) or {}),
            log_dir=getattr(args, "swanlab_logdir", None) or None,
            mode=getattr(args, "swanlab_mode", None) or None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[swanlab] 初始化失败，本次训练不上报（训练照常进行）：{exc}")
        print("         云端模式需要先 `swanlab login` 或设置 SWANLAB_API_KEY；"
              "没有网络可以在配置里把 swanlab_mode 设成 offline 或 local")
        return SwanLabTracker()

    print(f"[swanlab] 已开启实验记录：project={project}, name={name or '(自动命名)'}")
    return SwanLabTracker(run, log_interval=getattr(args, "swanlab_log_interval", 10))
