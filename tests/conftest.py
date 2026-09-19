"""pytest 共享夹具。

**所有测试都不联网、都不碰真实知识库**：API 客户端只测解析器（用固定样本），
需要 LLM/嵌入的地方用 ``hashing`` 提供方或直接跳过。

两处隔离是**强制覆盖**而不是 ``setdefault``，都是踩过的坑：

1. ``MEDSCHOLAR_HOME`` 指向**带 PID 的临时目录**。
   * 原来用固定的 ``medscholar_pytest`` 目录 + 会话级 ``rmtree``：两个 pytest 进程
     （比如并行跑、或 IDE 里同时跑两个文件）会互相删掉对方的临时库，
     表现为随机的 "no such table" —— 这类 flaky 极难定位。加 PID 后互不干扰。
   * 原来用 ``setdefault``：开发机上只要 shell 里导出过 ``MEDSCHOLAR_HOME``
     （``run.bat`` / ``scripts/e2e.py`` 都会设），测试就会**写真实知识库**。
     数据安全高于"尊重环境变量"，所以这里强制覆盖。
2. ``MEDSCHOLAR_OFFLINE`` / ``MEDSCHOLAR_EMBED_PROVIDER`` 同样强制：
   测试必须离线且确定性，不能因为开发者环境的变量而变成联网跑。

运行::

    .python\\python.exe -m pytest -q
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 在导入 medscholar 之前把数据目录指向临时位置，避免污染真实知识库。
# 带 PID：并发/嵌套的 pytest 进程各用各的目录，互不删除。
_TEST_HOME = Path(tempfile.gettempdir()) / f"medscholar_pytest_{os.getpid()}"
os.environ["MEDSCHOLAR_HOME"] = str(_TEST_HOME)
# 用确定性哈希嵌入，完全不依赖 Ollama；强制离线，绝不联网
os.environ["MEDSCHOLAR_EMBED_PROVIDER"] = "hashing"
os.environ["MEDSCHOLAR_OFFLINE"] = "true"


@pytest.fixture(scope="session", autouse=True)
def _clean_home():
    if _TEST_HOME.exists():
        shutil.rmtree(_TEST_HOME, ignore_errors=True)
    _TEST_HOME.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(_TEST_HOME, ignore_errors=True)


@pytest.fixture()
def config():
    from medscholar.config import AppConfig

    return AppConfig(data_dir=str(_TEST_HOME / "db"), offline=True)


@pytest.fixture()
def db(config, tmp_path):
    """每个测试一个独立的 SQLite 文件。"""
    from medscholar.db.connect import Database

    database = Database(tmp_path / "test.db", config=config)
    yield database
    database.close()


@pytest.fixture()
def sample_papers():
    from medscholar.models import Paper

    return [
        Paper(
            title="Accelerated rTMS for post-stroke depression: a randomized trial",
            abstract=(
                "BACKGROUND: Post-stroke depression is common. METHODS: 60 patients were "
                "randomly assigned. RESULTS: HAMD scores decreased significantly (P<0.01). "
                "CONCLUSION: Accelerated rTMS is effective."
            ),
            authors=["Zhang Wei", "Li Ming", "Chen Hua"],
            journal="Brain Stimulation",
            pub_year=2023,
            source="pubmed",
            pmid="37123456",
            doi="10.1016/j.brs.2023.001",
            mesh_terms=["Depression", "Stroke", "Transcranial Magnetic Stimulation"],
            cited_by_count=42,
            is_open_access=True,
            publication_type="Journal Article",
            volume="16",
            issue="2",
            pages="100-108",
        ),
        Paper(
            title="加速rTMS治疗卒中后抑郁的临床疗效观察",
            abstract=(
                "目的：探讨加速重复经颅磁刺激治疗卒中后抑郁的临床疗效。"
                "方法：将60例患者随机分为两组。结果：治疗组HAMD评分显著降低。"
            ),
            authors=["王伟", "李静", "张强", "赵敏"],
            journal="中国康复医学杂志",
            pub_year=2022,
            source="openalex",
            volume="37",
            issue="4",
            pages="512-516",
            keywords=["卒中后抑郁", "重复经颅磁刺激"],
        ),
        Paper(
            title="Deep brain stimulation for treatment-resistant depression",
            abstract="DBS of the subcallosal cingulate showed a sustained antidepressant response.",
            authors=["Malone DA"],
            journal="Biological Psychiatry",
            pub_year=2019,
            source="europepmc",
            doi="10.1016/j.biopsych.2019.05.011",
            cited_by_count=310,
            publication_type="Journal Article",
        ),
    ]
