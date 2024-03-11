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
    assert len(lst) > 0


def test_return_forbidden_when_get_template_list_not_auth():
    lst = get_template_list(-1)
    assert len(lst) == 0


def test_add_new_then_update_then_delete_template():
    request = SavePromptTemplateRequest(
        name="test_template_1", template="test_template_template_1"
    )
    assert save_prompt(request, 1).status_code == 200

    response = get_editor(request.name)
    assert response.status_code == 200

    body = json.loads(response.body)
    assert (
        body["prompt"]["name"] == request.name
        and body["prompt"]["template"] == request.template
    )

    response = delete_prompt_template(["test_template_1"], 1)
    assert response.status_code == 200

    response = get_editor(request.name)
    assert response.status_code == 404


def test_update_template():
    request = SavePromptTemplateRequest(
        name="test_template_1", template="test_template_template_1"
    )
    save_prompt(request, 1)
    request.template = "new_template"
    save_prompt(request, 1)

    response = get_editor(request.name)
    assert response.status_code == 200

    body = json.loads(response.body)
    assert (
        body["prompt"]["name"] == request.name
        and body["prompt"]["template"] == request.template
    )

    response = delete_prompt_template(["test_template_1"], 1)
    assert response.status_code == 200
