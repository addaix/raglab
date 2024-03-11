from raglab.app import *
import pytest

def test_can_list_app():
    lst = app_list(1)
    assert len(lst["names"]) == 1


def test_return_forbidden_when_app_list_not_auth():
    lst = app_list(-1)
    assert len(lst["names"]) == 0


def test_can_get_template_list():
    lst = get_template_list(1)
    assert len(lst) == 1


def test_return_forbidden_when_get_template_list_not_auth():
    lst = get_template_list(-1)
    assert len(lst) == 0
