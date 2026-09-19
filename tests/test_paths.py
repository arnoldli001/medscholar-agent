"""项目根目录与数据目录的解析。

**这些测试是为一次真实事故写的**：把 `medscholar/config.py` 搬到
`medscholar/platform/config.py` 之后，`project_root()` 里写死的
`Path(__file__).resolve().parent.parent` 少了一层 —— 源码树判定失败，
`data_home()` 悄悄退回 `~/.medscholar`，用户打开应用看到"知识库中还没有文献"。
仓库里的库好好的（几百篇），但程序读的是另一个空库，而且**没有任何报错**。

教训：
1. 不要用"数层数"的方式推导项目根 —— 文件一旦搬家就会失效；
2. 路径推导出错的后果往往是**静默读到另一个空库**，所以必须有测试守着；
3. 这类回归只有"真的跑一次应用"才会暴露，单测覆盖不到的话就轮到用户发现。
"""

from __future__ import annotations

from pathlib import Path

from medscholar.platform.config import (
    ROOT_MARKERS,
    _is_source_tree,
    config_path,
    data_home,
    project_root,
)


class TestProjectRoot:
    def test_root_contains_marker_files(self):
        """根目录必须真的含标记文件（否则源码树判定会失败）。"""
        root = project_root()
        assert root.is_dir(), f"项目根不是目录：{root}"
        assert any((root / marker).exists() for marker in ROOT_MARKERS), (
            f"{root} 里找不到任何一个标记文件 {ROOT_MARKERS} —— "
            "说明 project_root() 又数错层数了"
        )

    def test_root_is_not_the_package_directory(self):
        """回归：曾经返回的是 `medscholar/` 包目录而不是仓库根。"""
        root = project_root()
        assert root.name != "medscholar", "项目根不应该是包目录本身"
        assert (root / "medscholar").is_dir(), "仓库根下应当有 medscholar 包"

    def test_root_is_stable_regardless_of_module_depth(self):
        """根目录推导不能依赖"本文件在第几层"。"""
        root = project_root()
        # medscholar/platform/config.py → 需要向上 3 层；写死 parent.parent 会得到 medscholar/
        assert root == root.resolve()
        assert (root / "medscholar" / "platform" / "config.py").is_file()

    def test_is_source_tree_true_for_repo_root(self):
        assert _is_source_tree(project_root()) is True

    def test_is_source_tree_false_for_random_directory(self, tmp_path: Path):
        assert _is_source_tree(tmp_path) is False


class TestDataHome:
    def test_env_var_wins(self, monkeypatch, tmp_path: Path):
        """`MEDSCHOLAR_HOME` 是显式指定，优先级最高（测试与打包都靠它隔离）。"""
        target = tmp_path / "custom"
        monkeypatch.setenv("MEDSCHOLAR_HOME", str(target))
        assert data_home() == target.resolve()

    def test_source_tree_uses_repo_data_dir(self, monkeypatch):
        """在源码树里运行时，数据目录必须是仓库下的 `data/`。

        这一条就是事故的回归测试：如果 `project_root()` 数错层数，
        `_is_source_tree()` 会返回 False，数据目录会变成 `~/.medscholar`，
        用户的库"消失"。
        """
        monkeypatch.delenv("MEDSCHOLAR_HOME", raising=False)
        home = data_home()
        assert _is_source_tree(project_root()), "本测试假定运行在源码树里"
        assert home == project_root() / "data"
        assert home.is_relative_to(project_root()), "数据目录必须在项目根之内"

    def test_config_path_defaults_into_project(self, monkeypatch):
        monkeypatch.delenv("MEDSCHOLAR_HOME", raising=False)
        monkeypatch.delenv("MEDSCHOLAR_CONFIG", raising=False)
        path = config_path()
        assert path.name == "config.yaml"
        assert path.parent in {data_home(), project_root()}
