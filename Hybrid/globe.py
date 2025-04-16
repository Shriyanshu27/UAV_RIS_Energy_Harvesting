# globe.py

# Initialize global variable storage
def _init():
    global _global_dict
    _global_dict = {}

def set_value(key, value):
    """Set a global variable"""
    _global_dict[key] = value

def get_value(key, defValue=None):
    """Get a global variable, return default if not found"""
    try:
        return _global_dict[key]
    except KeyError:
        print(f"[Globe Warning] Key '{key}' not found. Returning default: {defValue}")
        return defValue

def has_key(key):
    """Check if key exists"""
    return key in _global_dict

def keys():
    """Return all stored keys"""
    return list(_global_dict.keys())

def clear():
    """Clear all global variables"""
    _global_dict.clear()
