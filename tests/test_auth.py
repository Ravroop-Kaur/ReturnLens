from src.auth.service import AuthService, hash_password, verify_password


def test_password_never_stored_in_plaintext():
    stored = hash_password("correct horse battery staple")
    assert "correct horse battery staple" not in stored
    assert "$" in stored


def test_verify_password_round_trip():
    stored = hash_password("hunter2")
    assert verify_password("hunter2", stored)
    assert not verify_password("wrong", stored)


def test_successful_login_returns_session():
    auth = AuthService()
    auth.register(email="merchant@example.com", password="pw123456", organization_id="org_1")
    result = auth.login("merchant@example.com", "pw123456")
    assert result.success
    assert result.session.organization_id == "org_1"


def test_failed_login_wrong_password():
    auth = AuthService()
    auth.register(email="merchant@example.com", password="pw123456", organization_id="org_1")
    result = auth.login("merchant@example.com", "wrongpassword")
    assert not result.success
    assert result.session is None


def test_failed_login_unknown_email_same_error_as_wrong_password():
    auth = AuthService()
    auth.register(email="merchant@example.com", password="pw123456", organization_id="org_1")
    r1 = auth.login("nobody@example.com", "whatever")
    r2 = auth.login("merchant@example.com", "wrongpassword")
    assert r1.error == r2.error


def test_protected_route_requires_valid_session():
    auth = AuthService()
    auth.register(email="a@example.com", password="pw123456", organization_id="org_1")
    login_result = auth.login("a@example.com", "pw123456")
    token = login_result.session.token

    ok = auth.require_session(token)
    assert ok.success

    bad = auth.require_session("not-a-real-token")
    assert not bad.success


def test_logout_invalidates_session():
    auth = AuthService()
    auth.register(email="a@example.com", password="pw123456", organization_id="org_1")
    token = auth.login("a@example.com", "pw123456").session.token
    assert auth.logout(token)
    result = auth.require_session(token)
    assert not result.success


def test_expired_session_rejected():
    auth = AuthService(session_ttl_seconds=-1)  # already expired the instant it's created
    auth.register(email="a@example.com", password="pw123456", organization_id="org_1")
    token = auth.login("a@example.com", "pw123456").session.token
    result = auth.require_session(token)
    assert not result.success
