from pathlib import Path
from dotenv import dotenv_values

def load_apikeys(env_path=None):
    if env_path is None:
        env_path = Path.cwd() / 'config' / '.env'
    if not env_path.exists():
        print(f"[ERROR] No file at: {env_path.absolute()}")
        return {}
    config = dotenv_values(str(env_path))
    return dict(config)






















