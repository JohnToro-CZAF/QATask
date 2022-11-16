from .sqlite import SQLiteDatabase
def build_database(cfg, dataset_path, database_path):
    if cfg.type == "sqlite":
        return SQLiteDatabase(cfg, dataset_path, database_path)
    else:
        assert cfg.type == "sqlite", "NotImplemented database{}".format(cfg.type)
