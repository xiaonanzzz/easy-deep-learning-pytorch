from easydl.config import get_config_from_cmd, TrainingConfig



if __name__ == '__main__':

    train_cfg = TrainingConfig()
    train_cfg.is_norm = True

    train_cfg.update_values_from_cmd()
    print(train_cfg.__dict__)