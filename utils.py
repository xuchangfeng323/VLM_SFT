class Arguments:
    def __init__(self, args_path: str=""):
        self.args_dict = self._load_json_config(args_path)
        for key, value in self.args_dict.items():
            setattr(self, key, value)

    def _load_json_config(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
