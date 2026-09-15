class Trainer:
    def __init__(self, args: Arguments):
        self.args = args
        self.model = QwenVlModel(args)
        self.dataset = GroundedMNER(args.train_data_path, args.image_root)
        