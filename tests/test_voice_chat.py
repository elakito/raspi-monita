from raspi_monita.voice_chat import update_config


def test_update_config():
    conf = {"a": "1",
            "b": "2",
            "e": {"x": "10", "y": "20"}}
    update_config("a", "5", "-1", conf)
    update_config("b", None, "-2", conf)
    update_config("c", None, "-3", conf)
    update_config("d", "8", "-4", conf)
    update_config("e", None, {"x": "-10", "y": "-20"}, conf)
    update_config("f", None, {"z": "-30"}, conf)

    assert conf["a"] == "5"
    assert conf["b"] == "2"
    assert conf["c"] == "-3"
    assert conf["d"] == "8"
    assert conf["e"] == {"x": "10", "y": "20"}
    assert conf["f"] == {"z": "-30"}

