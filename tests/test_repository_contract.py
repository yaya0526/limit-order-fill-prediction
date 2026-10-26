from scripts.check_repository import validate_repository


def test_published_repository_contract():
    assert validate_repository() == []
