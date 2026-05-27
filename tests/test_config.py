from app.core.config import Settings


def test_settings_coerces_release_debug_flag_to_false():
    settings = Settings(DEBUG="release")

    assert settings.debug is False
