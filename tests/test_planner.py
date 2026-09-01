# -*- coding: utf-8 -*-
"""Planner 的 JSON 解析与子任务解析单元测试。"""

from __future__ import annotations

import pytest

from app.core.agent.planner import _extract_json_object, _parse_subtasks


def test_extract_json_from_plain_text():
    data = _extract_json_object('{"subtasks": []}')
    assert data == {"subtasks": []}


def test_extract_json_from_code_fence():
    text = '```json\n{"subtasks": [{"id": "1"}]}\n```'
    data = _extract_json_object(text)
    assert data["subtasks"][0]["id"] == "1"


def test_extract_json_raises_without_json():
    with pytest.raises(ValueError):
        _extract_json_object("模型没有输出 JSON")


def test_parse_subtasks_returns_objects():
    data = {
        "subtasks": [
            {
                "id": "t1",
                "title": "检索知识",
                "description": "查找高血压饮食建议",
                "expected_tools": ["rag_search"],
            }
        ]
    }
    subtasks = _parse_subtasks(data)
    assert len(subtasks) == 1
    assert subtasks[0].id == "t1"
    assert subtasks[0].expected_tools == ["rag_search"]


def test_parse_subtasks_skips_invalid_items():
    data = {"subtasks": [{"id": "ok"}, "bad", None, 42]}
    subtasks = _parse_subtasks(data)
    assert len(subtasks) == 1
    assert subtasks[0].id == "ok"


def test_parse_subtasks_empty_when_missing():
    assert _parse_subtasks({}) == []
