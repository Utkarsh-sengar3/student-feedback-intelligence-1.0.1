from src.clean_data import clean_text

def test_clean_text():
    assert clean_text("Hello!!!  WORLD") == "hello world"
