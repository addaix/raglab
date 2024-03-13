from raglab.app import *
import pytest

ADMIN_USER = 1
UNKNOWN_USER = -1

def test_can_list_app():
    lst = app_list(ADMIN_USER)
    assert len(lst["names"]) == 1


def test_return_forbidden_when_app_list_not_auth():
    lst = app_list(UNKNOWN_USER)
    assert len(lst["names"]) == 0


def test_can_get_template_list():
    lst = get_template_list(ADMIN_USER)
    assert len(lst) > 0


def test_return_forbidden_when_get_template_list_not_auth():
    lst = get_template_list(UNKNOWN_USER)
    assert len(lst) == 0


def test_add_new_then_update_then_delete_template():
    request = SavePromptTemplateRequest(
        name="test_template_1", template="test_template_template_1"
    )
    assert save_prompt(request, ADMIN_USER).status_code == 200

    response = get_editor(request.name)
    assert response.status_code == 200

    body = json.loads(response.body)
    assert (
        body["prompt"]["name"] == request.name
        and body["prompt"]["template"] == request.template
    )

    response = delete_prompt_template(["test_template_1"], ADMIN_USER)
    assert response.status_code == 200

    response = get_editor(request.name)
    assert response.status_code == 404


def test_update_template():
    request = SavePromptTemplateRequest(
        name="test_template_1", template="test_template_template_1"
    )
    save_prompt(request, ADMIN_USER)
    request.template = "new_template"
    save_prompt(request, ADMIN_USER)

    response = get_editor(request.name)
    assert response.status_code == 200

    body = json.loads(response.body)
    assert (
        body["prompt"]["name"] == request.name
        and body["prompt"]["template"] == request.template
    )

    response = delete_prompt_template(["test_template_1"], ADMIN_USER)
    assert response.status_code == 200

def test_get_permissions() :
    permissions = get_permissions(ADMIN_USER)

    assert permissions.status_code == 200

    