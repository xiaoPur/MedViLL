from pathlib import PurePath


def make_json_compatible(value):
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {key: make_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_compatible(item) for item in value]
    return value
